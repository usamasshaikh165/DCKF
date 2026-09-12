"""
Accuracy-vs-gallery-scale evaluation with REAL distractors (extends E4/E6).

Protocol: the standard Flickr30k 1k-test queries (5,000 captions), ranked
against galleries of growing size:
  1,000 (E5 protocol) -> 31,783 (full Flickr30k, E4) -> 100k -> 500k -> 1M
where everything beyond 31,783 is real web-image CLIP features from DataComp
(see fetch_distractors.py). The ground-truth image is always present; ranks
can only get worse as distractors join — this measures how both the baseline
and the KG-expanded config (E7 config D) degrade with scale.

Two phases in one script:
  1. torch/CLIP: encode baseline + config-D query features (cached to npz)
  2. numpy only: one streaming pass over the distractor memmap, accumulating
     per-query "how many gallery items outscore the true image" counts,
     snapshotting recall at each scale point. No FAISS (exact ranking).

Usage:
  python src/scale_eval.py
  python src/scale_eval.py --scales 1000,31783,100000,1000000
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
QFEATS = ROOT / "results" / "scale_query_feats.npz"
REAL_FEATS = ROOT / "results" / "image_features_31783.npy"
OUT_CSV = ROOT / "results" / "scale_accuracy.csv"


# --------------------------------------------------------------------------
# phase 1 — query features (torch; skipped when the npz cache exists)
# --------------------------------------------------------------------------
def build_query_feats():
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import baseline_retrieval as br
    from ablations import content_len, load_queries
    from advanced_expansion import REWEIGHTED_PRIORS, build_text, rerank
    from kg_expanded_eval import DATASETS

    cfg = DATASETS["flickr30k"]
    br.set_dataset(str(cfg["img_dir"]), str(cfg["captions"]), cfg["tag"])
    model, preprocess = br.load_model()
    names, _ = br.encode_gallery(model, preprocess, limit=0)   # full 31,783
    assert len(names) == 31783
    queries, gt = load_queries(cfg, names[:1000], 5)  # 1k-test protocol;
    # names are the sorted prefix, so gt indices are valid in the full gallery
    qhash = hashlib.md5("\n".join(queries).encode()).hexdigest()[:10]

    cache = ROOT / "results" / "expansions_flickr30k_1000_5.json"
    blob = json.loads(cache.read_text())
    assert blob["qhash"] == qhash, "expansion cache doesn't match query set"
    expansions = blob["expansions"]

    f_base = br.encode_texts(model, queries).float()
    re_concepts = [rerank(e, REWEIGHTED_PRIORS) for e in expansions]
    texts = [build_text(q, c) for q, c in zip(queries, re_concepts)]
    f_exp = br.encode_texts(model, texts).float()
    lens = np.array([content_len(q) for q in queries])
    alpha = torch.tensor(np.clip(0.70 + 0.04 * lens, 0.75, 0.95),
                         dtype=torch.float32, device=f_base.device)[:, None]
    f_D = alpha * f_base + (1 - alpha) * f_exp
    f_D /= f_D.norm(dim=-1, keepdim=True)

    np.savez(QFEATS, qhash=qhash,
             f_base=f_base.cpu().numpy(), f_D=f_D.cpu().numpy(),
             gt=np.array(gt, dtype=np.int64))
    print(f"query features cached -> {QFEATS.name}")


# --------------------------------------------------------------------------
# phase 2 — streaming exact ranking (numpy only)
# --------------------------------------------------------------------------
def paired_bootstrap(ranks_a, ranks_b, k=1, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    hits_a, hits_b = (ranks_a < k).astype(float), (ranks_b < k).astype(float)
    idx = rng.integers(0, len(hits_a), (n_boot, len(hits_a)))
    delta = hits_b[idx].mean(axis=1) - hits_a[idx].mean(axis=1)
    return delta.mean(), np.percentile(delta, [2.5, 97.5]), (delta <= 0).mean()


def count_higher(Q, gallery_block, gt_sim, qchunk=1024, gt_cols=None):
    """Per query: how many columns of gallery_block outscore its true image.

    When gt_cols is given (real-gallery pass), gt_sim is READ from the matmul
    output instead of being precomputed — elementwise dot products differ from
    BLAS matmul by ~1 ulp, which made the true image outscore itself and
    silently turn every rank-0 into rank-1.
    """
    counts = np.zeros(len(Q), dtype=np.int64)
    G = gallery_block.astype(np.float32).T          # [512, g]
    for i in range(0, len(Q), qchunk):
        sims = Q[i:i + qchunk] @ G
        if gt_cols is not None:
            gt_sim[i:i + qchunk] = sims[np.arange(len(sims)),
                                        gt_cols[i:i + qchunk]]
        counts[i:i + qchunk] = (sims > gt_sim[i:i + qchunk, None]).sum(axis=1)
    return counts


def metrics(ranks):
    return {**{f"R@{k}": float((ranks < k).mean()) for k in (1, 5, 10)},
            "MedR": float(np.median(ranks) + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="1000,31783,100000,500000,1000000")
    ap.add_argument("--distractors", default="")
    args = ap.parse_args()
    scales = sorted(int(s) for s in args.scales.split(","))

    if not QFEATS.exists():
        build_query_feats()
    q = np.load(QFEATS)
    f_base, f_D, gt = q["f_base"], q["f_D"], q["gt"]
    n_q = len(gt)

    real = np.asarray(np.load(REAL_FEATS, mmap_mode="r"), dtype=np.float32)
    n_real = len(real)

    dpath = (Path(args.distractors) if args.distractors else
             next(iter(sorted((ROOT / "results").glob(
                 "distractors_datacomp_*.fp16.npy"))), None))
    mpath = dpath.with_suffix(".meta.json") if dpath else None
    dmeta = (json.loads(mpath.read_text())
             if mpath and mpath.exists() else {"filled": 0})
    n_dis = dmeta["filled"]
    dis = np.load(dpath, mmap_mode="r") if n_dis else None
    print(f"{n_q} queries | {n_real:,} real gallery | "
          f"{n_dis:,} distractors available")
    scales = [s for s in scales if s <= n_real + n_dis]

    # stack both configs -> one streaming pass
    Q = np.concatenate([f_base, f_D]).astype(np.float32)     # [2*n_q, 512]
    gt2 = np.tile(gt, 2)
    gt_sim = np.empty(len(Q), dtype=np.float32)

    # ranks within the real gallery (first 1000 cols = the 1k-test gallery)
    ranks = {}
    if 1000 in scales:
        ranks[1000] = count_higher(Q, real[:1000], gt_sim, gt_cols=gt2)
    counts_real = count_higher(Q, real, gt_sim, gt_cols=gt2)
    if n_real in scales:
        ranks[n_real] = counts_real.copy()

    # stream distractors, snapshot at each scale point
    acc = counts_real.copy()
    done = n_real
    for s in [s for s in scales if s > n_real]:
        for i in range(done - n_real, s - n_real, 100_000):
            j = min(i + 100_000, s - n_real)
            acc += count_higher(Q, dis[i:j], gt_sim)
            print(f"  streamed {n_real + j:,}/{s:,}", flush=True)
        done = s
        ranks[s] = acc.copy()

    # report
    rows = []
    print(f"\n{'gallery':>10} {'config':<12} {'R@1':>8} {'R@5':>8} "
          f"{'R@10':>8} {'MedR':>6}   dR@1  bootstrap")
    for s in scales:
        ra, rd = ranks[s][:n_q], ranks[s][n_q:]
        ma, md = metrics(ra), metrics(rd)
        mean_d, (lo, hi), p = paired_bootstrap(ra, rd)
        for tag, m in (("baseline", ma), ("D adaptive", md)):
            d = "" if tag == "baseline" else (
                f"{(md['R@1'] - ma['R@1']) * 100:+.2f}  "
                f"CI[{lo * 100:+.2f},{hi * 100:+.2f}] p={p:.4f}")
            print(f"{s:>10,} {tag:<12} {m['R@1']:8.2%} {m['R@5']:8.2%} "
                  f"{m['R@10']:8.2%} {m['MedR']:6.0f}   {d}")
            rows.append((s, tag, m, mean_d, lo, hi, p))

    with open(OUT_CSV, "w") as f:
        f.write("gallery,config,R@1,R@5,R@10,MedR,dR1_mean,ci_lo,ci_hi,p\n")
        for s, tag, m, mean_d, lo, hi, p in rows:
            f.write(f"{s},{tag},{m['R@1']:.4f},{m['R@5']:.4f},"
                    f"{m['R@10']:.4f},{m['MedR']:.0f},"
                    f"{mean_d:.4f},{lo:.4f},{hi:.4f},{p:.4f}\n")
    print(f"\nSaved {OUT_CSV}")


if __name__ == "__main__":
    main()

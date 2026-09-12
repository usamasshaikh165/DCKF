"""
Adaptive, relation-reweighted expansion — the configs findings F3-F5 point at.

All configs re-rank the CACHED expansion candidates (results/expansions_*.json,
which store relation + sim per concept), so no new KG/expansion pass is needed.

Configs:
  A  baseline
  B  fused a=0.9, cached concept order (reproduces E5 best)
  C  relation-reweighted (IsA demoted 1.0->0.4, AtLocation/UsedFor promoted),
     T2 template, fused a=0.9
  D  C + adaptive alpha: a(q) = clamp(0.70 + 0.04*content_len(q), 0.75, 0.95)
     (short queries get more expansion signal)
  E  selective: expand only if content_len < 6 AND best concept sim >= 0.6;
     enriched queries fused at a=0.85, others stay pure baseline
Significance: paired bootstrap (10k resamples) on best config vs baseline.

Usage:
  python src/advanced_expansion.py --dataset flickr30k --limit 1000 --captions-per-image 5
  python src/advanced_expansion.py --dataset coco5k --captions-per-image 1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_retrieval as br                       # noqa: E402
from ablations import content_len, load_queries      # noqa: E402
from kg_expanded_eval import DATASETS, recall_table   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

REWEIGHTED_PRIORS = {
    "AtLocation": 1.0, "UsedFor": 1.0, "PartOf": 0.9, "Synonym": 0.8,
    "MadeOf": 0.8, "CapableOf": 0.7, "HasA": 0.7,
    "IsA": 0.4,          # F4: hypernyms dilute descriptive queries
    "RelatedTo": 0.3,
}


def rerank(concepts, priors, min_sim=0.55, k=2):
    scored = [(priors.get(c["relation"], 0.0) * c["sim"], c) for c in concepts
              if c["sim"] >= min_sim]
    scored.sort(key=lambda t: -t[0])
    return [c for s, c in scored[:k] if s > 0]


def build_text(q, concepts):
    if not concepts:
        return q
    return f"{q}, " + ", ".join(c["concept"] for c in concepts)


def ranks_of(txt_feats, img_feats, gt, chunk=2048):
    gt_t = torch.tensor(gt, device=txt_feats.device)
    ranks = torch.empty(len(gt), device=txt_feats.device)
    for i in range(0, len(gt), chunk):
        sims = txt_feats[i:i + chunk] @ img_feats.T
        gt_scores = sims.gather(1, gt_t[i:i + chunk, None])
        ranks[i:i + chunk] = (sims > gt_scores).sum(dim=1).float()
    return ranks.cpu().numpy()


# E40: when CLUSTER_BOOT=1 and a script has registered the per-query image index
# here, the bootstrap resamples IMAGES (clusters of 5 captions) instead of captions.
BOOT_CLUSTERS = None


def set_boot_clusters(gt):
    """Register per-query cluster ids (the ground-truth image index) for E40."""
    global BOOT_CLUSTERS
    import os
    BOOT_CLUSTERS = np.asarray(gt) if os.environ.get("CLUSTER_BOOT") else None


def paired_bootstrap(ranks_a, ranks_b, k=1, n_boot=10_000, seed=0):
    """P(config B beats A on R@k) via paired bootstrap over queries, or over
    image clusters when BOOT_CLUSTERS matches the query count (E40)."""
    rng = np.random.default_rng(seed)
    hits_a, hits_b = (ranks_a < k).astype(float), (ranks_b < k).astype(float)
    n = len(hits_a)
    cl = BOOT_CLUSTERS
    if cl is not None and len(cl) == n:
        _, inv = np.unique(cl, return_inverse=True)
        n_c = inv.max() + 1
        sa = np.bincount(inv, weights=hits_a, minlength=n_c)
        sb = np.bincount(inv, weights=hits_b, minlength=n_c)
        cnt = np.bincount(inv, minlength=n_c).astype(float)
        idx = rng.integers(0, n_c, (n_boot, n_c))
        delta = (sb[idx].sum(axis=1) - sa[idx].sum(axis=1)) / cnt[idx].sum(axis=1)
    else:
        idx = rng.integers(0, n, (n_boot, n))
        delta = hits_b[idx].mean(axis=1) - hits_a[idx].mean(axis=1)
    return delta.mean(), np.percentile(delta, [2.5, 97.5]), (delta <= 0).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="flickr30k")
    ap.add_argument("--captions-per-image", type=int, default=5)
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    br.set_dataset(str(cfg["img_dir"]), str(cfg["captions"]), cfg["tag"])
    model, preprocess = br.load_model()
    names, img_feats = br.encode_gallery(model, preprocess, limit=args.limit)
    img_feats = img_feats.float()
    queries, gt = load_queries(cfg, names, args.captions_per_image)

    key = f"{args.dataset}_{len(names)}_{args.captions_per_image}"
    cache = ROOT / "results" / f"expansions_{key}.json"
    if not cache.exists():
        sys.exit(f"No expansion cache {cache} — run ablations.py first.")
    expansions = json.loads(cache.read_text())["expansions"]
    print(f"{len(queries)} queries, {len(names)} images, cache: {cache.name}")

    f_base = br.encode_texts(model, queries).float()
    lens = np.array([content_len(q) for q in queries])

    results, rank_store = {}, {}

    def add(tag, feats):
        r = ranks_of(feats, img_feats, gt)
        rank_store[tag] = r
        results[tag] = {f"R@{k}": float((r < k).mean()) for k in (1, 5, 10)}

    add("A baseline", f_base)

    # B: E5-best reproduction (cached order, top-3, T1-ish -> use fused only)
    texts_B = [build_text(q, e[:3]) for q, e in zip(queries, expansions)]
    f_B = br.encode_texts(model, texts_B).float()
    fB = 0.9 * f_base + 0.1 * f_B
    fB /= fB.norm(dim=-1, keepdim=True)
    add("B fused a=.9 (E5 best)", fB)

    # C: relation-reweighted top-2, fused a=0.9
    re_concepts = [rerank(e, REWEIGHTED_PRIORS) for e in expansions]
    n_c = sum(1 for c in re_concepts if c)
    texts_C = [build_text(q, c) for q, c in zip(queries, re_concepts)]
    f_C = br.encode_texts(model, texts_C).float()
    fC = 0.9 * f_base + 0.1 * f_C
    fC /= fC.norm(dim=-1, keepdim=True)
    add(f"C reweighted (n={n_c})", fC)

    # D: C + adaptive alpha by query length
    alpha = torch.tensor(np.clip(0.70 + 0.04 * lens, 0.75, 0.95),
                         dtype=torch.float32, device=f_base.device)[:, None]
    fD = alpha * f_base + (1 - alpha) * f_C
    fD /= fD.norm(dim=-1, keepdim=True)
    add("D adaptive alpha", fD)

    # E: selective — short & confident queries only, a=0.85
    sel_mask = torch.tensor(
        [(l < 6 and bool(c) and c[0]["sim"] >= 0.6)
         for l, c in zip(lens, re_concepts)], device=f_base.device)
    fE = f_base.clone()
    mixed = 0.85 * f_base + 0.15 * f_C
    mixed /= mixed.norm(dim=-1, keepdim=True)
    fE[sel_mask] = mixed[sel_mask]
    add(f"E selective (n={int(sel_mask.sum())})", fE)

    base_r1 = results["A baseline"]["R@1"]
    print(f"\n{'config':<26} {'R@1':>8} {'R@5':>8} {'R@10':>8}   dR@1")
    for tag, m in results.items():
        print(f"{tag:<26} {m['R@1']:8.2%} {m['R@5']:8.2%} {m['R@10']:8.2%} "
              f"  {(m['R@1']-base_r1)*100:+.2f}")

    # bootstrap the best non-baseline config
    best = max((t for t in results if not t.startswith("A")),
               key=lambda t: results[t]["R@1"])
    mean_d, (lo, hi), p = paired_bootstrap(rank_store["A baseline"],
                                           rank_store[best])
    print(f"\nPaired bootstrap ({best} vs baseline, R@1):")
    print(f"  mean delta {mean_d*100:+.2f} pts, 95% CI [{lo*100:+.2f}, {hi*100:+.2f}], "
          f"p(delta<=0) = {p:.4f}")


if __name__ == "__main__":
    main()

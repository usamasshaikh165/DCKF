"""
E25 — Bidirectional retrieval evaluation (text→image AND image→text) + mR.

Motivation (KAIS/KBS venue requirements, supervisor direction 2026-07-25):
journal reviewers in this area expect BOTH retrieval directions plus the mR
summary metric (mean of the six recalls), as reported by KTIR and the
standard Karpathy-protocol literature.

Protocol
  t2i: each caption is a query; rank of its ground-truth image among all
       gallery images (strictly-greater rule; identical to ranks_of, so
       t2i numbers reproduce every previously logged experiment).
  i2t: each image is a query; rank = number of captions scoring strictly
       higher than the image's BEST-scoring ground-truth caption
       ("at least one of its captions in top-K" — standard convention).
  mR:  mean of {t2i, i2t} × {R@1, R@5, R@10}.

Both directions share one similarity computation per chunk and the GT score
is gathered from the SAME matmul that produces the ranking scores — this is
the E10 lesson (1-ulp BLAS mismatch otherwise shifts every rank by +1).

External validation anchors (clip_benchmark, OpenAI ViT-B/32, Karpathy):
  Flickr30k 1k:  t2i 58.78/83.56/90.02   i2t 78.90/94.90/98.20
  COCO 5k:       t2i 30.44/55.94/66.87   i2t 50.12/75.00/83.52
Our own t2i runs sit ~0.1–0.7 above these (JPEG-decode/preprocess noise
band, cf. E1/E2 reproduction notes); i2t must land in the same band.

Configs (same construction as additivity_eval.py / E17):
  A baseline
  B + query-side KG      (reweighted top-2, gentle fusion a_q; text side)
  C + gallery captions   (BLIP caption fusion a_g; image side)
  D + both
B needs results/expansions_{key}.json; C/D need results/blip_captions_{key}.json.
Missing caches degrade gracefully (config skipped, noted in output).

Usage:
  python src/bidirectional_eval.py --dataset flickr30k --limit 1000
  python src/bidirectional_eval.py --dataset coco5k --limit 0
  python src/bidirectional_eval.py --dataset rsicd --limit 0
Output: printed table + results/bidirectional_{key}.json (full metrics,
deltas, paired bootstraps for both directions).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_retrieval as br                          # noqa: E402
from ablations import load_queries                       # noqa: E402
from kg_expanded_eval import DATASETS                    # noqa: E402
from advanced_expansion import (REWEIGHTED_PRIORS, rerank, build_text,  # noqa: E402
                                ranks_of, paired_bootstrap)

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def i2t_ranks(txt_feats, img_feats, gt, n_images, chunk=512):
    """Rank of each image's best ground-truth caption among all captions.

    gt[q] = image index of caption q. Returns array of length n_images.
    Strictly-greater counting, GT scores gathered from the same matmul.
    """
    device = img_feats.device
    # image index -> its caption (query) indices
    caps_per_img: list[list[int]] = [[] for _ in range(n_images)]
    for q_idx, img_idx in enumerate(gt):
        caps_per_img[img_idx].append(q_idx)
    n_caps = [len(c) for c in caps_per_img]
    if min(n_caps) == 0:
        raise ValueError("some gallery image has no ground-truth caption; "
                         "i2t recall would be undefined for it")
    max_caps = max(n_caps)
    # padded gather index [N, max_caps] + validity mask
    pad_idx = torch.zeros(n_images, max_caps, dtype=torch.long, device=device)
    mask = torch.zeros(n_images, max_caps, dtype=torch.bool, device=device)
    for i, caps in enumerate(caps_per_img):
        pad_idx[i, :len(caps)] = torch.tensor(caps, device=device)
        mask[i, :len(caps)] = True

    ranks = torch.empty(n_images, device=device)
    for i in range(0, n_images, chunk):
        sims = img_feats[i:i + chunk] @ txt_feats.T                 # [c, Q]
        gt_scores = sims.gather(1, pad_idx[i:i + chunk])            # [c, max_caps]
        gt_scores = gt_scores.masked_fill(~mask[i:i + chunk], float("-inf"))
        best_gt = gt_scores.max(dim=1, keepdim=True).values         # [c, 1]
        ranks[i:i + chunk] = (sims > best_gt).sum(dim=1).float()
    return ranks.cpu().numpy()


def recalls(ranks):
    out = {f"R@{k}": float((ranks < k).mean()) for k in (1, 5, 10)}
    out["MedR"] = float(np.median(ranks) + 1)
    return out


def eval_config(txt_feats, img_feats, gt, n_images):
    r_t2i = ranks_of(txt_feats, img_feats, gt)
    r_i2t = i2t_ranks(txt_feats, img_feats, gt, n_images)
    m_t2i, m_i2t = recalls(r_t2i), recalls(r_i2t)
    mr = float(np.mean([m_t2i[f"R@{k}"] for k in (1, 5, 10)]
                       + [m_i2t[f"R@{k}"] for k in (1, 5, 10)]))
    return {"t2i": m_t2i, "i2t": m_i2t, "mR": mr,
            "_ranks_t2i": r_t2i, "_ranks_i2t": r_i2t}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="flickr30k")
    ap.add_argument("--captions-per-image", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--alpha-q", type=float, default=0.9)
    ap.add_argument("--alpha-g", type=float, default=0.9)
    ap.add_argument("--caption-suffix", default="",
                    help="e.g. _large_aerial -> blip_captions_{key}_large_aerial.json (E41)")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    br.set_dataset(str(cfg["img_dir"]), str(cfg["captions"]), cfg["tag"])
    model, preprocess = br.load_model()
    names, img_feats = br.encode_gallery(model, preprocess, limit=args.limit)
    img_feats = img_feats.float()
    queries, gt = load_queries(cfg, names, args.captions_per_image)
    __import__('advanced_expansion').set_boot_clusters(gt)  # E40 cluster bootstrap
    n_images = len(names)
    caps_hist = np.bincount(np.bincount(gt))
    print(f"{len(queries)} captions over {n_images} images "
          f"({args.dataset}); captions/image histogram tail: "
          f"{dict((i, int(c)) for i, c in enumerate(caps_hist) if c)}")
    f_q = br.encode_texts(model, queries).float()

    key = f"{args.dataset}_{n_images}_{args.captions_per_image}"
    exp_file = ROOT / "results" / f"expansions_{key}.json"
    cap_file = ROOT / "results" / f"blip_captions_{key}{args.caption_suffix}.json"

    configs = {}
    configs["A baseline"] = (f_q, img_feats)

    if exp_file.exists():
        expansions = json.loads(exp_file.read_text())["expansions"]
        if len(expansions) == len(queries):
            rew = [rerank(e, REWEIGHTED_PRIORS) for e in expansions]
            f_exp = br.encode_texts(
                model, [build_text(q, c) for q, c in zip(queries, rew)]).float()
            fq_kg = args.alpha_q * f_q + (1 - args.alpha_q) * f_exp
            fq_kg /= fq_kg.norm(dim=-1, keepdim=True)
            configs[f"B query-KG aq={args.alpha_q}"] = (fq_kg, img_feats)
        else:
            print(f"NOTE: expansion cache holds {len(expansions)} entries "
                  f"vs {len(queries)} queries — config B skipped")
    else:
        print(f"NOTE: {exp_file.name} missing — config B skipped")

    fused_g = None
    if cap_file.exists():
        cap_map = json.loads(cap_file.read_text())
        f_cap = br.encode_texts(model, [cap_map[n] for n in names]).float()
        fused_g = args.alpha_g * img_feats + (1 - args.alpha_g) * f_cap
        fused_g /= fused_g.norm(dim=-1, keepdim=True)
        configs[f"C gallery-cap ag={args.alpha_g}"] = (f_q, fused_g)
        if any(t.startswith("B") for t in configs):
            configs[f"D both"] = (fq_kg, fused_g)
    else:
        print(f"NOTE: {cap_file.name} missing — configs C/D skipped")

    results = {}
    for tag, (tf, gf) in configs.items():
        results[tag] = eval_config(tf, gf, gt, n_images)

    hdr = (f"\n{'config':<24} | {'t2i R@1':>8} {'R@5':>7} {'R@10':>7} {'MedR':>5}"
           f" | {'i2t R@1':>8} {'R@5':>7} {'R@10':>7} {'MedR':>5} | {'mR':>6}")
    print(hdr); print("-" * len(hdr))
    for tag, m in results.items():
        t, i = m["t2i"], m["i2t"]
        print(f"{tag:<24} | {t['R@1']:8.2%} {t['R@5']:7.2%} {t['R@10']:7.2%} "
              f"{t['MedR']:5.0f} | {i['R@1']:8.2%} {i['R@5']:7.2%} "
              f"{i['R@10']:7.2%} {i['MedR']:5.0f} | {m['mR']:6.2%}")

    boots = {}
    base = results["A baseline"]
    for tag, m in results.items():
        if tag == "A baseline":
            continue
        for direction, unit in (("t2i", "captions"), ("i2t", "images")):
            d, (lo, hi), p = paired_bootstrap(base[f"_ranks_{direction}"],
                                              m[f"_ranks_{direction}"])
            boots[f"{tag} vs A [{direction}]"] = {
                "delta_R@1": d, "ci95": [lo, hi], "p": p, "unit": unit}
            print(f"bootstrap {tag} vs A, {direction} R@1 (paired over {unit}): "
                  f"{d*100:+.2f} pts, 95% CI [{lo*100:+.2f}, {hi*100:+.2f}], "
                  f"p(delta<=0)={p:.4f}")

    out = {
        "dataset": args.dataset, "n_images": n_images,
        "n_captions": len(queries),
        "captions_per_image": args.captions_per_image,
        "alpha_q": args.alpha_q, "alpha_g": args.alpha_g,
        "device": br.DEVICE,
        "configs": {t: {k: v for k, v in m.items() if not k.startswith("_")}
                    for t, m in results.items()},
        "bootstraps": boots,
    }
    out_file = ROOT / "results" / f"bidirectional_{key}{args.caption_suffix}.json"
    out_file.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""
Sensitivity sweep for the drift-gate threshold tau (min_sim), reusing the
cached expansion candidates so no new ConceptNet lookup is needed. Config C
(relation-reweighted, top-2, fused alpha=0.9) held fixed; only tau varies.

Usage:
  python src/tau_sensitivity.py --dataset flickr30k --limit 1000 --captions-per-image 5
  python src/tau_sensitivity.py --dataset rsicd --limit 1093 --captions-per-image 5
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
from kg_expanded_eval import DATASETS                 # noqa: E402
from advanced_expansion import (REWEIGHTED_PRIORS, rerank, build_text,
                                 ranks_of, paired_bootstrap)  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

TAU_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


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
    print(f"{args.dataset}: {len(queries)} queries, {len(names)} images, "
          f"cache: {cache.name}\n")

    f_base = br.encode_texts(model, queries).float()
    r_base = ranks_of(f_base, img_feats, gt)
    base_r1 = float((r_base < 1).mean())

    print(f"{'tau':>6} {'n_kept':>8} {'R@1':>8} {'dR@1':>8}   95% CI"
          f"                p(<=0)")
    for tau in TAU_GRID:
        re_concepts = [rerank(e, REWEIGHTED_PRIORS, min_sim=tau, k=2)
                       for e in expansions]
        n_kept = sum(1 for c in re_concepts if c)
        texts = [build_text(q, c) for q, c in zip(queries, re_concepts)]
        f_txt = br.encode_texts(model, texts).float()
        f_fused = 0.9 * f_base + 0.1 * f_txt
        f_fused /= f_fused.norm(dim=-1, keepdim=True)
        r_fused = ranks_of(f_fused, img_feats, gt)
        r1 = float((r_fused < 1).mean())
        mean_d, (lo, hi), p = paired_bootstrap(r_base, r_fused, n_boot=5000)
        print(f"{tau:6.2f} {n_kept:8d} {r1:8.2%} {(r1-base_r1)*100:+8.2f}   "
              f"[{lo*100:+.2f}, {hi*100:+.2f}]        {p:.4f}")


if __name__ == "__main__":
    main()

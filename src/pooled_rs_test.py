"""
E22 — Pooled significance test: query-side reweighted KG expansion across ALL
remote-sensing datasets (RSICD + RSITMD + UCM + NWPU, 24.5k queries).

Each RS dataset individually shows a small positive query-side effect
(+0.05..+0.19) that is underpowered alone. This pools them with a STRATIFIED
paired bootstrap (resample queries within each dataset, combine weighted by
query count) — the defense-grade single number for "relation-aware KG
expansion helps in the remote-sensing domain".

Config tested: reweighted top-2 (min_sim .55), gentle fusion a=0.9 — the
same config C reported per-dataset in E9/E20/E21.

Usage: python src/pooled_rs_test.py
"""
import json
import sys
from pathlib import Path

import os
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_retrieval as br                        # noqa: E402
from ablations import load_queries                     # noqa: E402
from kg_expanded_eval import DATASETS                  # noqa: E402
from advanced_expansion import (REWEIGHTED_PRIORS, rerank, build_text,  # noqa: E402
                                ranks_of)

ROOT = Path(__file__).resolve().parent.parent
RS = [("rsicd", 0, 5), ("rsitmd", 0, 5), ("ucm", 0, 5), ("nwpu", 0, 5)]


@torch.no_grad()
def main():
    model, preprocess = br.load_model()
    per_ds = {}
    for ds, limit, cpi in RS:
        cfg = DATASETS[ds]
        br.set_dataset(str(cfg["img_dir"]), str(cfg["captions"]), cfg["tag"])
        names, img_feats = br.encode_gallery(model, preprocess, limit=limit)
        img_feats = img_feats.float()
        queries, gt = load_queries(cfg, names, cpi)
        __import__('advanced_expansion').set_boot_clusters(gt)  # E40 cluster bootstrap
        key = f"{ds}_{len(names)}_{cpi}"
        expansions = json.loads(
            (ROOT / "results" / f"expansions_{key}.json").read_text())["expansions"]
        f_q = br.encode_texts(model, queries).float()
        rew = [rerank(e, REWEIGHTED_PRIORS) for e in expansions]
        f_e = br.encode_texts(
            model, [build_text(q, c) for q, c in zip(queries, rew)]).float()
        fused = 0.9 * f_q + 0.1 * f_e
        fused /= fused.norm(dim=-1, keepdim=True)
        ra = ranks_of(f_q, img_feats, gt)
        rb = ranks_of(fused, img_feats, gt)
        per_ds[ds] = (ra, rb, np.asarray(gt))
        print(f"{ds}: {len(queries)} q, "
              f"dR@1={((rb < 1).mean() - (ra < 1).mean()) * 100:+.3f}, "
              f"dR@5={((rb < 5).mean() - (ra < 5).mean()) * 100:+.3f}")

    rng = np.random.default_rng(0)
    n_boot = 10_000
    total_q = sum(len(ra) for ra, *_ in per_ds.values())
    for k in (1, 5, 10):
        deltas = np.zeros(n_boot)
        obs = 0.0
        for ra, rb, g in per_ds.values():
            ha, hb = (ra < k).astype(float), (rb < k).astype(float)
            n = len(ha)
            w = n / total_q
            obs += (hb.mean() - ha.mean()) * w
            if os.environ.get("CLUSTER_BOOT"):          # E40: resample images within each dataset
                _, inv = np.unique(g, return_inverse=True); n_c = inv.max() + 1
                sa = np.bincount(inv, weights=ha, minlength=n_c); sb = np.bincount(inv, weights=hb, minlength=n_c)
                cnt = np.bincount(inv, minlength=n_c).astype(float)
                idx = rng.integers(0, n_c, (n_boot, n_c))
                deltas += (sb[idx].sum(axis=1) - sa[idx].sum(axis=1)) / cnt[idx].sum(axis=1) * w
            else:
                idx = rng.integers(0, n, (n_boot, n))
                deltas += (hb[idx].mean(axis=1) - ha[idx].mean(axis=1)) * w
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        p = (deltas <= 0).mean()
        print(f"POOLED R@{k}: {obs*100:+.3f} pts, "
              f"95% CI [{lo*100:+.3f}, {hi*100:+.3f}], p(delta<=0) = {p:.4f} "
              f"({total_q:,} queries, stratified)")


if __name__ == "__main__":
    main()

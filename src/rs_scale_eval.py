"""
E23 — Remote-sensing scale test: does the pooled RS query-KG effect (F22)
survive realistic in-domain gallery growth?

RSICD queries (5,465) rank the true image among 1,093 RSICD images + N
distractors, with N from two pools at matched scales:
  in-domain : Million-AID aerial embeddings (distractors_millionaid_*.npy)
  web       : DataComp embeddings          (distractors_datacomp_*.npy)
F10 (Flickr) says in-domain distractors dominate difficulty; this is the RS
replication, plus the KG-delta at each scale (F11 analog).

Usage: python src/rs_scale_eval.py --scales 0,10000,50000,100000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_retrieval as br                        # noqa: E402
from ablations import load_queries                     # noqa: E402
from kg_expanded_eval import DATASETS                  # noqa: E402
from advanced_expansion import (REWEIGHTED_PRIORS, rerank, build_text,  # noqa: E402
                                paired_bootstrap)

ROOT = Path(__file__).resolve().parent.parent


def ranks_with_distractors(txt, gallery, gt, chunk=1024):
    gt_t = torch.tensor(gt, device=txt.device)
    out = torch.empty(len(gt), device=txt.device)
    for i in range(0, len(gt), chunk):
        sims = txt[i:i + chunk] @ gallery.T
        gt_scores = sims.gather(1, gt_t[i:i + chunk, None])
        out[i:i + chunk] = (sims > gt_scores).sum(dim=1).float()
    return out.cpu().numpy()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="0,10000,50000,100000")
    args = ap.parse_args()
    scales = [int(s) for s in args.scales.split(",")]

    cfg = DATASETS["rsicd"]
    br.set_dataset(str(cfg["img_dir"]), str(cfg["captions"]), cfg["tag"])
    model, preprocess = br.load_model()
    names, img_feats = br.encode_gallery(model, preprocess, limit=0)
    img_feats = img_feats.float()
    queries, gt = load_queries(cfg, names, 5)
    __import__('advanced_expansion').set_boot_clusters(gt)  # E40 cluster bootstrap
    key = f"rsicd_{len(names)}_5"
    expansions = json.loads(
        (ROOT / "results" / f"expansions_{key}.json").read_text())["expansions"]
    f_q = br.encode_texts(model, queries).float()
    rew = [rerank(e, REWEIGHTED_PRIORS) for e in expansions]
    f_e = br.encode_texts(
        model, [build_text(q, c) for q, c in zip(queries, rew)]).float()
    fused = 0.9 * f_q + 0.1 * f_e
    fused /= fused.norm(dim=-1, keepdim=True)
    print(f"{len(queries)} RSICD queries, {len(names)} true gallery images")

    pools = {}
    for tag, pat in (("in-domain", "distractors_millionaid_*.fp16.npy"),
                     ("web", "distractors_datacomp_*.fp16.npy")):
        f = sorted((ROOT / "results").glob(pat))
        if f:
            pools[tag] = np.load(f[-1], mmap_mode="r")
            print(f"{tag} pool: {f[-1].name} ({pools[tag].shape[0]:,})")

    print(f"\n{'pool':<10} {'N dist.':>9} {'base R@1':>9} {'KG R@1':>8} "
          f"{'dR@1':>7} {'p':>7}   base/KG R@5")
    for tag, pool in pools.items():
        for n in scales:
            if n > pool.shape[0]:
                continue
            if n == 0 and tag == "web":
                continue  # identical to in-domain n=0
            d = torch.from_numpy(np.ascontiguousarray(pool[:n])).float().to(
                img_feats.device) if n else None
            gallery = torch.cat([img_feats, d]) if n else img_feats
            ra = ranks_with_distractors(f_q, gallery, gt)
            rb = ranks_with_distractors(fused, gallery, gt)
            _, _, p = paired_bootstrap(ra, rb)
            r1a, r1b = (ra < 1).mean(), (rb < 1).mean()
            r5a, r5b = (ra < 5).mean(), (rb < 5).mean()
            print(f"{tag:<10} {n:>9,} {r1a:>9.2%} {r1b:>8.2%} "
                  f"{(r1b - r1a) * 100:>+7.2f} {p:>7.4f}   "
                  f"{r5a:.2%}/{r5b:.2%}")
            del gallery, d
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

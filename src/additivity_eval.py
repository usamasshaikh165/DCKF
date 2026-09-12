"""
E17 — Additivity of query-side and gallery-side enrichment.

Query-side E7 (reweighted KG expansion, gentle fusion a_q=0.9) gave +0.2..0.4;
gallery-side frozen-BLIP caption fusion (E13/E15) gave +2.4..2.9 on natural
images. Are the gains independent (additive), overlapping, or synergistic?

Configs:
  A  baseline
  B  query-side only  (E7: rerank cached expansions, fuse a_q=0.9)
  C  gallery-side only (caption fusion, a_g sweep 0.8/0.9)
  D  both
Bootstraps: D vs best C (does query-side still add on an enriched gallery?),
D vs A (headline).

Requires caches: results/expansions_{key}.json (ablations.py),
results/blip_captions_{key}.json (gallery_caption_expansion.py).

Usage:
  python src/additivity_eval.py --dataset flickr30k --limit 1000 --captions-per-image 5
  python src/additivity_eval.py --dataset coco5k --captions-per-image 1 --limit 0
  python src/additivity_eval.py --dataset rsicd --captions-per-image 5 --limit 0
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_retrieval as br                        # noqa: E402
from ablations import load_queries                     # noqa: E402
from kg_expanded_eval import DATASETS                  # noqa: E402
from advanced_expansion import (REWEIGHTED_PRIORS, rerank, build_text,  # noqa: E402
                                ranks_of, paired_bootstrap)

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="flickr30k")
    ap.add_argument("--captions-per-image", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--alpha-q", type=float, default=0.9)
    ap.add_argument("--alphas-g", default="0.8,0.9")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    br.set_dataset(str(cfg["img_dir"]), str(cfg["captions"]), cfg["tag"])
    model, preprocess = br.load_model()
    names, img_feats = br.encode_gallery(model, preprocess, limit=args.limit)
    img_feats = img_feats.float()
    queries, gt = load_queries(cfg, names, args.captions_per_image)
    f_q = br.encode_texts(model, queries).float()

    key = f"{args.dataset}_{len(names)}_{args.captions_per_image}"
    exp_file = ROOT / "results" / f"expansions_{key}.json"
    cap_file = ROOT / "results" / f"blip_captions_{key}.json"
    for f in (exp_file, cap_file):
        if not f.exists():
            sys.exit(f"missing cache: {f}")
    expansions = json.loads(exp_file.read_text())["expansions"]
    cap_map = json.loads(cap_file.read_text())
    print(f"{len(queries)} queries over {len(names)} images ({args.dataset})")

    # query-side E7: reweighted top-2 of cached candidates, gentle fusion
    rew = [rerank(e, REWEIGHTED_PRIORS) for e in expansions]
    f_exp = br.encode_texts(
        model, [build_text(q, c) for q, c in zip(queries, rew)]).float()
    fq_kg = args.alpha_q * f_q + (1 - args.alpha_q) * f_exp
    fq_kg /= fq_kg.norm(dim=-1, keepdim=True)

    # gallery-side: BLIP caption fusion
    f_cap = br.encode_texts(model, [cap_map[n] for n in names]).float()

    results, rank_store = {}, {}

    def add(tag_, tf, gf):
        r = ranks_of(tf, gf, gt)
        rank_store[tag_] = r
        results[tag_] = {f"R@{k}": float((r < k).mean()) for k in (1, 5, 10)}

    add("A baseline", f_q, img_feats)
    add(f"B query-KG aq={args.alpha_q}", fq_kg, img_feats)
    for ag in [float(x) for x in args.alphas_g.split(",")]:
        fused_g = ag * img_feats + (1 - ag) * f_cap
        fused_g /= fused_g.norm(dim=-1, keepdim=True)
        add(f"C gallery-cap ag={ag}", f_q, fused_g)
        add(f"D both ag={ag}", fq_kg, fused_g)

    base_r1 = results["A baseline"]["R@1"]
    d_b = results[f"B query-KG aq={args.alpha_q}"]["R@1"] - base_r1
    print(f"\n{'config':<24} {'R@1':>8} {'R@5':>8} {'R@10':>8}   dR@1")
    for tag_, m in results.items():
        print(f"{tag_:<24} {m['R@1']:8.2%} {m['R@5']:8.2%} {m['R@10']:8.2%} "
              f"  {(m['R@1'] - base_r1) * 100:+.2f}")

    best_c = max((t for t in results if t.startswith("C")),
                 key=lambda t: results[t]["R@1"])
    best_d = max((t for t in results if t.startswith("D")),
                 key=lambda t: results[t]["R@1"])
    d_c = results[best_c]["R@1"] - base_r1
    d_d = results[best_d]["R@1"] - base_r1
    print(f"\nadditivity: dB={d_b*100:+.2f}, dC={d_c*100:+.2f}, "
          f"dB+dC={(d_b+d_c)*100:+.2f}, observed dD={d_d*100:+.2f}")

    for label, base in ((f"vs {best_c} (query-side increment on enriched "
                         f"gallery)", best_c), ("vs A baseline", "A baseline")):
        mean_d, (lo, hi), p = paired_bootstrap(rank_store[base], rank_store[best_d])
        print(f"\nPaired bootstrap {best_d} {label}, R@1:")
        print(f"  mean delta {mean_d*100:+.2f} pts, "
              f"95% CI [{lo*100:+.2f}, {hi*100:+.2f}], p(delta<=0) = {p:.4f}")


if __name__ == "__main__":
    main()

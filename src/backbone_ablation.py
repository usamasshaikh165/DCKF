"""
E19 — Backbone-strength ablation: does the enrichment gain survive ViT-L/14?

F14/F15b predict gains shrink as the text channel strengthens; a committee
will ask whether the whole effect is a ViT-B/32 artifact. Protocol: identical
pipeline, expansion TEXTS held fixed (concepts selected by the B/32 pipeline,
cached), only the encoder changes — isolates backbone strength.

Per dataset:
  A  L/14 baseline
  B  + query-side KG (reweighted top-2, fused a_q=0.9)     [E17 protocol]
  C  + gallery-side BLIP captions (a_g in 0.8/0.9)          [E13 protocol]
  D  both (best a_g)
Bootstraps: B vs A (KG survival), best-of-C/D vs A (headline survival).
B/32 reference deltas (E17): Flickr +0.22/+2.36/+2.32, COCO +0.08/+2.94/+3.14,
RSICD +0.11/(aerial +0.09)/—.

Usage:  python src/backbone_ablation.py
"""
import json
import sys
from pathlib import Path

import clip
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_retrieval as br                        # noqa: E402
from ablations import load_queries                     # noqa: E402
from kg_expanded_eval import DATASETS                  # noqa: E402
from advanced_expansion import (REWEIGHTED_PRIORS, rerank, build_text,  # noqa: E402
                                ranks_of, paired_bootstrap)

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "ViT-L/14"

RUNS = [
    # dataset, limit, caps/img, feature-cache tag, gallery-caption cache
    ("flickr30k", 1000, 5, "flickr30k_l14", "blip_captions_flickr30k_1000_5.json"),
    ("coco5k", 0, 1, "coco5k_l14", "blip_captions_coco5k_5000_1.json"),
    ("rsicd", 0, 5, "rsicd_l14", "blip_captions_rsicd_1093_5_aerial.json"),
]


@torch.no_grad()
def main():
    model, preprocess = clip.load(MODEL_NAME, device=br.DEVICE)
    model.eval()
    print(f"{MODEL_NAME} loaded on {br.DEVICE}")

    for ds, limit, cpi, ftag, cap_name in RUNS:
        cfg = DATASETS[ds]
        br.set_dataset(str(cfg["img_dir"]), str(cfg["captions"]), ftag)
        names, img_feats = br.encode_gallery(model, preprocess, limit=limit)
        img_feats = img_feats.float()
        queries, gt = load_queries(cfg, names, cpi)
        __import__('advanced_expansion').set_boot_clusters(gt)  # E40 cluster bootstrap
        f_q = br.encode_texts(model, queries).float()
        key = f"{ds}_{len(names)}_{cpi}"
        print(f"\n===== {ds}: {len(queries)} queries / {len(names)} images "
              f"({MODEL_NAME}) =====")

        exp_file = ROOT / "results" / f"expansions_{key}.json"
        cap_file = ROOT / "results" / cap_name
        results, rank_store = {}, {}

        def add(tag_, tf, gf):
            r = ranks_of(tf, gf, gt)
            rank_store[tag_] = r
            results[tag_] = {f"R@{k}": float((r < k).mean()) for k in (1, 5, 10)}

        add("A baseline", f_q, img_feats)

        # B: query-side KG, concepts fixed from the B/32 pipeline cache
        if exp_file.exists():
            expansions = json.loads(exp_file.read_text())["expansions"]
            rew = [rerank(e, REWEIGHTED_PRIORS) for e in expansions]
            f_exp = br.encode_texts(
                model, [build_text(q, c) for q, c in zip(queries, rew)]).float()
            fq_kg = 0.9 * f_q + 0.1 * f_exp
            fq_kg /= fq_kg.norm(dim=-1, keepdim=True)
            add("B query-KG aq=0.9", fq_kg, img_feats)
        else:
            print(f"  (no expansion cache {exp_file.name} — skipping B)")
            fq_kg = None

        # C/D: gallery captions re-encoded with L/14
        if cap_file.exists():
            cap_map = json.loads(cap_file.read_text())
            f_cap = br.encode_texts(model, [cap_map[n] for n in names]).float()
            for ag in (0.8, 0.9):
                fg = ag * img_feats + (1 - ag) * f_cap
                fg /= fg.norm(dim=-1, keepdim=True)
                add(f"C gallery-cap ag={ag}", f_q, fg)
                if fq_kg is not None:
                    add(f"D both ag={ag}", fq_kg, fg)
        else:
            print(f"  (no caption cache {cap_name} — skipping C/D)")

        base_r1 = results["A baseline"]["R@1"]
        print(f"{'config':<22} {'R@1':>8} {'R@5':>8} {'R@10':>8}   dR@1")
        for tag_, m in results.items():
            print(f"{tag_:<22} {m['R@1']:8.2%} {m['R@5']:8.2%} "
                  f"{m['R@10']:8.2%}   {(m['R@1'] - base_r1) * 100:+.2f}")

        for probe in ("B query-KG aq=0.9",
                      max((t for t in results if t[0] in "CD"),
                          key=lambda t: results[t]["R@1"], default=None)):
            if probe and probe in results:
                mean_d, (lo, hi), p = paired_bootstrap(
                    rank_store["A baseline"], rank_store[probe])
                print(f"bootstrap {probe} vs A: {mean_d*100:+.2f} "
                      f"[{lo*100:+.2f},{hi*100:+.2f}] p={p:.4f}")


if __name__ == "__main__":
    main()

"""
E28 — Stronger captioner for the RSICD gallery channel (bounded risk-3 shot).

E15/E18: BLIP-base captions are a net liability on RSICD (−0.84 R@1);
prompt-conditioning repairs them only to neutral (+0.09 ns). One upgrade
shot that fits the disk budget: BLIP-large (~1.9 GB) with the E18-best
aerial prompt. BLIP-2 (15 GB) does not fit dell3's free disk.

Configs (both directions, α_g in {0.9, 0.95}):
  A baseline | C captions(large, aerial) | D = C + query-KG (E7 treatment)

Usage (dell3):
  HF_ENDPOINT=https://hf-mirror.com python src/rsicd_blip_large.py
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_retrieval as br                          # noqa: E402
from ablations import load_queries                       # noqa: E402
from kg_expanded_eval import DATASETS                    # noqa: E402
from advanced_expansion import (REWEIGHTED_PRIORS, rerank, build_text,  # noqa: E402
                                paired_bootstrap)
from bidirectional_eval import eval_config               # noqa: E402
import rsicd_domain_caption as rdc                       # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
rdc.BLIP_ID = "Salesforce/blip-image-captioning-large"
PROMPT = "an aerial photograph of"


@torch.no_grad()
def main():
    cfg = DATASETS["rsicd"]
    br.set_dataset(str(cfg["img_dir"]), str(cfg["captions"]), cfg["tag"])
    model, preprocess = br.load_model()
    names, img_feats = br.encode_gallery(model, preprocess, limit=0)
    img_feats = img_feats.float()
    queries, gt = load_queries(cfg, names, 5)
    n_images = len(names)
    key = f"rsicd_{n_images}_5"
    print(f"{len(queries)} captions over {n_images} images (rsicd)")

    paths = sorted(p for p in Path(cfg["img_dir"]).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    caps = rdc.blip_conditional(
        paths, PROMPT,
        ROOT / "results" / f"blip_captions_{key}_large_aerial.json")
    print("example caption:", caps[0])
    f_cap = br.encode_texts(model, caps).float()

    f_q = br.encode_texts(model, queries).float()
    expansions = json.loads(
        (ROOT / "results" / f"expansions_{key}.json").read_text())["expansions"]
    rew = [rerank(e, REWEIGHTED_PRIORS) for e in expansions]
    f_exp = br.encode_texts(
        model, [build_text(q, c) for q, c in zip(queries, rew)]).float()
    fq_kg = 0.9 * f_q + 0.1 * f_exp
    fq_kg /= fq_kg.norm(dim=-1, keepdim=True)

    results = {"A baseline": eval_config(f_q, img_feats, gt, n_images)}
    for ag in (0.9, 0.95):
        fused_g = ag * img_feats + (1 - ag) * f_cap
        fused_g /= fused_g.norm(dim=-1, keepdim=True)
        results[f"C large-cap ag={ag}"] = eval_config(f_q, fused_g, gt, n_images)
        results[f"D both ag={ag}"] = eval_config(fq_kg, fused_g, gt, n_images)

    hdr = (f"\n{'config':<22} | {'t2i R@1':>8} {'R@5':>7} {'R@10':>7}"
           f" | {'i2t R@1':>8} {'R@5':>7} {'R@10':>7} | {'mR':>6}")
    print(hdr); print("-" * len(hdr))
    for tag, m in results.items():
        t, i = m["t2i"], m["i2t"]
        print(f"{tag:<22} | {t['R@1']:8.2%} {t['R@5']:7.2%} {t['R@10']:7.2%} "
              f"| {i['R@1']:8.2%} {i['R@5']:7.2%} {i['R@10']:7.2%} "
              f"| {m['mR']:6.2%}")

    boots = {}
    for tag in list(results)[1:]:
        d, (lo, hi), p = paired_bootstrap(results["A baseline"]["_ranks_t2i"],
                                          results[tag]["_ranks_t2i"])
        boots[f"{tag} vs A"] = {"delta_R@1": d, "ci95": [lo, hi], "p": p}
        print(f"bootstrap {tag} vs A, t2i R@1: {d*100:+.2f} "
              f"[{lo*100:+.2f}, {hi*100:+.2f}], p={p:.4f}")
    best_c = max((t for t in results if t.startswith("C")),
                 key=lambda t: results[t]["t2i"]["R@1"])
    best_d = best_c.replace("C large-cap", "D both")
    d, (lo, hi), p = paired_bootstrap(results[best_c]["_ranks_t2i"],
                                      results[best_d]["_ranks_t2i"])
    boots["KG increment on large-cap"] = {"delta_R@1": d, "ci95": [lo, hi], "p": p}
    print(f"bootstrap KG increment ({best_d} vs {best_c}), t2i R@1: "
          f"{d*100:+.2f} [{lo*100:+.2f}, {hi*100:+.2f}], p={p:.4f}")

    out = {"dataset": "rsicd", "captioner": rdc.BLIP_ID, "prompt": PROMPT,
           "configs": {t: {k: v for k, v in m.items() if not k.startswith("_")}
                       for t, m in results.items()},
           "bootstraps": boots}
    (ROOT / "results" / "rsicd_blip_large.json").write_text(
        json.dumps(out, indent=2))
    print("\nwrote results/rsicd_blip_large.json")


if __name__ == "__main__":
    main()

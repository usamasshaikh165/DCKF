"""
E18 — Domain-conditioned gallery captions for RSICD (next-steps #1).

E15/F15 found BLIP's natural-image captions are a net LIABILITY on RSICD
(-0.84 R@1 at a=0.9), while RSICD is the only dataset where the KG increment
is positive (+0.24, p=.084). Hypothesis: fix the caption channel with
DOMAIN-CONDITIONED captioning (BLIP conditional generation with an aerial
prompt prefix — zero training, frozen model), then the KG has a working
channel to amplify.

Steps:
  1. Caption RSICD gallery with BLIP-base, unconditional (E15 cache) vs
     prompt-conditioned ("an aerial photograph of", "a satellite image of").
  2. Emb-fuse each caption set into image features, a in {0.9, 0.95}.
  3. Best caption set -> KG expansion (E7/E14-best treatment: reweighted
     top-2, channel fusion g=0.8) -> KG increment bootstrap.

Usage:
  HF_ENDPOINT=https://hf-mirror.com python src/rsicd_domain_caption.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_retrieval as br                        # noqa: E402
from ablations import load_queries                     # noqa: E402
from kg_expanded_eval import DATASETS                  # noqa: E402
from advanced_expansion import (REWEIGHTED_PRIORS, rerank, build_text,  # noqa: E402
                                ranks_of, paired_bootstrap)
from expand_query import QueryExpander                 # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BLIP_ID = "Salesforce/blip-image-captioning-base"
PROMPTS = {
    "aerial": "an aerial photograph of",
    "satellite": "a satellite image of",
}


@torch.no_grad()
def blip_conditional(paths, prompt, cache_file, batch_size=32):
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        if set(cached) >= {p.name for p in paths}:
            print(f"Loaded cached captions: {cache_file.name}")
            return [cached[p.name] for p in paths]
    from transformers import BlipProcessor, BlipForConditionalGeneration
    proc = BlipProcessor.from_pretrained(BLIP_ID)
    blip = BlipForConditionalGeneration.from_pretrained(
        BLIP_ID, torch_dtype=torch.float16).to("cuda").eval()
    caps, t0 = [], time.time()
    for i in range(0, len(paths), batch_size):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
        inputs = proc(images=imgs, text=[prompt] * len(imgs),
                      return_tensors="pt", padding=True).to("cuda", torch.float16)
        out = blip.generate(**inputs, max_new_tokens=30)
        caps += [c.strip() for c in proc.batch_decode(out, skip_special_tokens=True)]
        print(f"  captioned {min(i + batch_size, len(paths))}/{len(paths)}", flush=True)
    print(f"conditional captioning [{prompt}]: "
          f"{(time.time() - t0) / len(paths) * 1000:.0f} ms/image")
    cache_file.write_text(json.dumps(dict(zip([p.name for p in paths], caps)), indent=1))
    del blip
    torch.cuda.empty_cache()
    return caps


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions-per-image", type=int, default=5)
    ap.add_argument("--alphas", default="0.9,0.95")
    args = ap.parse_args()

    cfg = DATASETS["rsicd"]
    br.set_dataset(str(cfg["img_dir"]), str(cfg["captions"]), cfg["tag"])
    model, preprocess = br.load_model()
    names, img_feats = br.encode_gallery(model, preprocess, limit=0)
    img_feats = img_feats.float()
    queries, gt = load_queries(cfg, names, args.captions_per_image)
    f_q = br.encode_texts(model, queries).float()
    print(f"{len(queries)} queries over {len(names)} images (rsicd)")

    key = f"rsicd_{len(names)}_{args.captions_per_image}"
    paths = [Path(cfg["img_dir"]) / n for n in names]

    cap_sets = {}
    uncond_file = ROOT / "results" / f"blip_captions_{key}.json"
    if uncond_file.exists():
        cap_sets["uncond"] = [json.loads(uncond_file.read_text())[n] for n in names]
    for tag, prompt in PROMPTS.items():
        cap_sets[tag] = blip_conditional(
            paths, prompt, ROOT / "results" / f"blip_captions_{key}_{tag}.json")
    print("sample captions (first 3 images):")
    for tag, caps in cap_sets.items():
        print(f"  [{tag}] " + " | ".join(caps[:3]))

    results, rank_store = {}, {}

    def add(tag_, gallery):
        r = ranks_of(f_q, gallery, gt)
        rank_store[tag_] = r
        results[tag_] = {f"R@{k}": float((r < k).mean()) for k in (1, 5, 10)}

    add("A baseline", img_feats)
    alphas = [float(x) for x in args.alphas.split(",")]
    feats = {}
    for tag, caps in cap_sets.items():
        feats[tag] = br.encode_texts(model, caps).float()
        for a in alphas:
            fused = a * img_feats + (1 - a) * feats[tag]
            fused /= fused.norm(dim=-1, keepdim=True)
            add(f"B {tag} a={a}", fused)

    # best caption set -> KG on top (E14-best: reweighted top-2, channel g=0.8)
    best_b = max((t for t in results if t.startswith("B")),
                 key=lambda t: results[t]["R@1"])
    best_tag = best_b.split()[1]
    best_a = float(best_b.split("=")[1])
    caps = cap_sets[best_tag]
    print(f"\nKG expansion on best caption set: {best_b}")

    ex = QueryExpander(model=model, device=br.DEVICE, top_k=8, sim_threshold=0.30)
    pool_file = ROOT / "results" / f"gallery_cap_expansions_{key}_{best_tag}.json"
    if pool_file.exists():
        pools = json.loads(pool_file.read_text())["pools"]
    else:
        pools = [ex.expand(c)["concepts"] for c in caps]
        pool_file.write_text(json.dumps({"pools": pools}))

    rew = [rerank(p, REWEIGHTED_PRIORS, min_sim=0.55, k=2) for p in pools]
    n_exp = sum(1 for c in rew if c)
    f_kgtext = br.encode_texts(
        model, [build_text(c, n) for c, n in zip(caps, rew)]).float()
    for g in (0.8, 0.9):
        f_ch = g * feats[best_tag] + (1 - g) * f_kgtext
        f_ch /= f_ch.norm(dim=-1, keepdim=True)
        fused = best_a * img_feats + (1 - best_a) * f_ch
        fused /= fused.norm(dim=-1, keepdim=True)
        add(f"C KG channel g={g} (n={n_exp})", fused)
    # naive replace variant for completeness
    fused = best_a * img_feats + (1 - best_a) * f_kgtext
    fused /= fused.norm(dim=-1, keepdim=True)
    add("C KG replace", fused)

    base_r1 = results["A baseline"]["R@1"]
    print(f"\n{'config':<28} {'R@1':>8} {'R@5':>8} {'R@10':>8}   dR@1")
    for tag_, m in results.items():
        print(f"{tag_:<28} {m['R@1']:8.2%} {m['R@5']:8.2%} {m['R@10']:8.2%} "
              f"  {(m['R@1'] - base_r1) * 100:+.2f}")

    best_c = max((t for t in results if t.startswith("C")),
                 key=lambda t: results[t]["R@1"])
    for label, base in ((f"vs {best_b} (KG increment)", best_b),
                        ("vs A baseline", "A baseline")):
        mean_d, (lo, hi), p = paired_bootstrap(rank_store[base], rank_store[best_c])
        print(f"\nPaired bootstrap {best_c} {label}, R@1:")
        print(f"  mean delta {mean_d*100:+.2f} pts, "
              f"95% CI [{lo*100:+.2f}, {hi*100:+.2f}], p(delta<=0) = {p:.4f}")


if __name__ == "__main__":
    main()

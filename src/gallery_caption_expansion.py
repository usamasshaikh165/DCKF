"""
Gallery-side expansion v2 (E13) — EXTERNAL semantic source: frozen BLIP-base
captions at index time, optionally KG-expanded, fused into the image side.

F12 showed self-derived tags add no information (data-processing chain).
A frozen off-the-shelf captioner IS an external source: its language head was
trained on image-text pairs outside CLIP's embedding space, so its captions
can carry signal the CLIP image embedding lost. Still training-free (no
fine-tuning; contrast KTIR's fine-tuned BLIP). Zero query-side latency.

No leakage: BLIP never sees the human eval captions; it only sees pixels.

Configs (x weight sweep):
  A  baseline image features
  B  emb-fusion   img + BLIP caption           (external text, no KG)
  C  emb-fusion   img + KG-enriched caption    (the thesis mechanism)
  D  score-fusion img + BLIP caption
  E  score-fusion img + KG-enriched caption
Bootstrap: best KG config vs baseline; best KG vs best caption-only (increment).

Usage:
  HF_ENDPOINT=https://hf-mirror.com python src/gallery_caption_expansion.py \
      --dataset flickr30k --limit 1000 --captions-per-image 5
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
from advanced_expansion import ranks_of, paired_bootstrap  # noqa: E402
from expand_query import QueryExpander                 # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BLIP_ID = "Salesforce/blip-image-captioning-base"


@torch.no_grad()
def blip_captions(paths, cache_file, batch_size=32):
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        if set(cached) >= {p.name for p in paths}:
            print(f"Loaded {len(paths)} cached BLIP captions.")
            return [cached[p.name] for p in paths]
    from transformers import BlipProcessor, BlipForConditionalGeneration
    proc = BlipProcessor.from_pretrained(BLIP_ID)
    blip = BlipForConditionalGeneration.from_pretrained(
        BLIP_ID, torch_dtype=torch.float16).to("cuda").eval()
    caps, t0 = [], time.time()
    for i in range(0, len(paths), batch_size):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
        inputs = proc(images=imgs, return_tensors="pt").to("cuda", torch.float16)
        out = blip.generate(**inputs, max_new_tokens=30)
        caps += [c.strip() for c in proc.batch_decode(out, skip_special_tokens=True)]
        print(f"  captioned {min(i + batch_size, len(paths))}/{len(paths)}", flush=True)
    ms = (time.time() - t0) / len(paths) * 1000
    print(f"BLIP captioning: {ms:.0f} ms/image (offline, index-time only)")
    cache_file.write_text(json.dumps(
        dict(zip([p.name for p in paths], caps)), indent=1))
    del blip
    torch.cuda.empty_cache()
    return caps


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="flickr30k")
    ap.add_argument("--captions-per-image", type=int, default=5)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--sim-threshold", type=float, default=0.5)
    ap.add_argument("--alphas", default="0.8,0.9,0.95")
    ap.add_argument("--betas", default="0.05,0.1,0.15,0.2")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    br.set_dataset(str(cfg["img_dir"]), str(cfg["captions"]), cfg["tag"])
    model, preprocess = br.load_model()
    names, img_feats = br.encode_gallery(model, preprocess, limit=args.limit)
    img_feats = img_feats.float()
    queries, gt = load_queries(cfg, names, args.captions_per_image)
    f_q = br.encode_texts(model, queries).float()
    print(f"{len(queries)} queries over {len(names)} images ({args.dataset})")

    key = f"{args.dataset}_{len(names)}_{args.captions_per_image}"
    paths = [Path(cfg["img_dir"]) / n for n in names]
    caps = blip_captions(paths, ROOT / "results" / f"blip_captions_{key}.json")

    # KG-expand the BLIP captions with the existing query-side machinery
    ex = QueryExpander(model=model, device=br.DEVICE,
                       top_k=args.top_k, sim_threshold=args.sim_threshold)
    t0, enriched, n_changed = time.time(), [], 0
    for i, c in enumerate(caps):
        r = ex.expand(c)
        enriched.append(r["enriched"])
        n_changed += r["enriched"] != c
        if (i + 1) % 250 == 0:
            print(f"  expanded {i + 1}/{len(caps)}", flush=True)
    print(f"KG expansion: {n_changed}/{len(caps)} captions enriched, "
          f"{(time.time() - t0) / len(caps) * 1000:.0f} ms/image (offline)")

    f_cap = br.encode_texts(model, caps).float()
    f_kg = br.encode_texts(model, enriched).float()

    results, rank_store = {}, {}

    def add(tag_, ranks):
        rank_store[tag_] = ranks
        results[tag_] = {f"R@{k}": float((ranks < k).mean()) for k in (1, 5, 10)}

    add("A baseline", ranks_of(f_q, img_feats, gt))
    for a in [float(x) for x in args.alphas.split(",")]:
        for label, f_txt in (("B emb cap", f_cap), ("C emb cap+KG", f_kg)):
            fused = a * img_feats + (1 - a) * f_txt
            fused /= fused.norm(dim=-1, keepdim=True)
            add(f"{label} a={a}", ranks_of(f_q, fused, gt))

    gt_t = torch.tensor(gt, device=img_feats.device)
    s_img = f_q @ img_feats.T
    for b in [float(x) for x in args.betas.split(",")]:
        for label, f_txt in (("D score cap", f_cap), ("E score cap+KG", f_kg)):
            s = (1 - b) * s_img + b * (f_q @ f_txt.T)
            gt_scores = s.gather(1, gt_t[:, None])
            add(f"{label} b={b}",
                (s > gt_scores).sum(dim=1).float().cpu().numpy())

    base_r1 = results["A baseline"]["R@1"]
    print(f"\n{'config':<22} {'R@1':>8} {'R@5':>8} {'R@10':>8}   dR@1")
    for tag_, m in results.items():
        print(f"{tag_:<22} {m['R@1']:8.2%} {m['R@5']:8.2%} {m['R@10']:8.2%} "
              f"  {(m['R@1'] - base_r1) * 100:+.2f}")

    best_kg = max((t for t in results if t[0] in "CE"),
                  key=lambda t: results[t]["R@1"])
    best_cap = max((t for t in results if t[0] in "BD"),
                   key=lambda t: results[t]["R@1"])
    for label, a_tag, b_tag in (("best KG vs baseline", "A baseline", best_kg),
                                ("best KG vs best caption-only (KG increment)",
                                 best_cap, best_kg)):
        mean_d, (lo, hi), p = paired_bootstrap(rank_store[a_tag], rank_store[b_tag])
        print(f"\nPaired bootstrap {label} [{b_tag} vs {a_tag}], R@1:")
        print(f"  mean delta {mean_d*100:+.2f} pts, "
              f"95% CI [{lo*100:+.2f}, {hi*100:+.2f}], p(delta<=0) = {p:.4f}")


if __name__ == "__main__":
    main()

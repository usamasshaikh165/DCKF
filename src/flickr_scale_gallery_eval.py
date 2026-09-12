"""
E36 — Natural-image gallery channel under in-domain distractors (review point 5, Flickr30k).

The +2.3 gallery-caption gain was measured on the 1k Karpathy test gallery. Here the other
30,783 Flickr30k images (train/val, never queried) are added as in-domain distractors, and
EVERY gallery item, test or distractor, is fused identically with its own BLIP-base caption
(the paper's natural-image channel). Configs on the 5,000 k1k test queries, t2i R@1/R@5:
  A baseline | B DCKF-Q | C captions a | D both | T caption-only
at gallery sizes 1,000 (test only), 6,000, 11,000, 31,783 (all Flickr30k).
Usage (dell3, GPU 1):  HF_ENDPOINT=https://hf-mirror.com python src/flickr_scale_gallery_eval.py
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_retrieval as br
from ablations import load_queries
from kg_expanded_eval import DATASETS
from advanced_expansion import REWEIGHTED_PRIORS, rerank, build_text, paired_bootstrap
from rs_scale_gallery_eval import ranks, norm
ROOT = Path(__file__).resolve().parent.parent
BLIP_ID = "Salesforce/blip-image-captioning-base"

@torch.no_grad()
def blip_uncond(paths, cache_file, batch_size=48):
    cached = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    todo = [p for p in paths if p.name not in cached]
    if todo:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from PIL import Image
        proc = BlipProcessor.from_pretrained(BLIP_ID)
        blip = BlipForConditionalGeneration.from_pretrained(BLIP_ID, torch_dtype=torch.float16).to("cuda").eval()
        t0 = time.time()
        for i in range(0, len(todo), batch_size):
            imgs = [Image.open(p).convert("RGB") for p in todo[i:i+batch_size]]
            inputs = proc(images=imgs, return_tensors="pt").to("cuda", torch.float16)
            out = blip.generate(**inputs, max_new_tokens=30)
            for p, c in zip(todo[i:i+batch_size], proc.batch_decode(out, skip_special_tokens=True)):
                cached[p.name] = c.strip()
            if (i // batch_size) % 50 == 0:
                print(f"  captioned {min(i+batch_size, len(todo))}/{len(todo)}  {(time.time()-t0)/60:.1f} min", flush=True)
                cache_file.write_text(json.dumps(cached))
        cache_file.write_text(json.dumps(cached)); del blip; torch.cuda.empty_cache()
    return [cached[p.name] for p in paths]

@torch.no_grad()
def main():
    # test gallery + queries (official k1k)
    cfg = DATASETS["flickr30k_k1k"]; br.set_dataset(str(cfg["img_dir"]), str(cfg["captions"]), cfg["tag"])
    model, pre = br.load_model(); names, img = br.encode_gallery(model, pre)
    queries, gt = load_queries(cfg, names, 5); key = f"flickr30k_k1k_{len(names)}_5"; dev = img.device
    __import__('advanced_expansion').set_boot_clusters(gt)  # E40 cluster bootstrap
    f_q = br.encode_texts(model, queries)
    pools = json.loads((ROOT/"results"/f"expansions_{key}.json").read_text())["expansions"]
    rew = [rerank(e, REWEIGHTED_PRIORS) for e in pools]
    f_kg = norm(0.9 * f_q + 0.1 * br.encode_texts(model, [build_text(q, c) for q, c in zip(queries, rew)]))
    cm = json.loads((ROOT/"results"/f"blip_captions_{key}.json").read_text())
    f_cap = br.encode_texts(model, [cm[n] for n in names])
    # distractors: all other Flickr30k images
    all_dir = ROOT / "data/images"; test_set = set(names)
    dpaths = sorted(p for p in all_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.name not in test_set)
    print(f"{len(queries)} queries | {len(names)} test images | {len(dpaths)} in-domain distractors")
    caps = blip_uncond(dpaths, ROOT/"results"/"blip_captions_flickr30k_distractors.json")
    # encode distractor images (own cache) and their captions
    fcache = ROOT/"results"/"image_features_flickr30k_distractors.pt"
    if fcache.exists():
        blob = torch.load(fcache); assert blob["names"] == [p.name for p in dpaths]; d_img = norm(blob["features"].float()).to(dev)
    else:
        from PIL import Image
        feats = []
        for i in range(0, len(dpaths), 128):
            x = torch.stack([pre(Image.open(p).convert("RGB")) for p in dpaths[i:i+128]]).to(dev)
            feats.append(norm(model.encode_image(x).float()).cpu())
            if (i // 128) % 40 == 0: print(f"  encoded {min(i+128, len(dpaths))}/{len(dpaths)}", flush=True)
        d_img = torch.cat(feats); torch.save({"names": [p.name for p in dpaths], "features": d_img}, fcache); d_img = d_img.to(dev)
    d_cap = br.encode_texts(model, caps)
    out = {"scales": []}
    print(f"\n{'gallery':>8} {'config':<20} {'R@1':>7} {'R@5':>7}   dR@1 vs A   CI               p")
    for n in (0, 5000, 10000, len(dpaths)):
        dr, dc = (d_img[:n], d_cap[:n]) if n else (None, None)
        gal_A = torch.cat([img, dr]) if n else img
        rA = ranks(f_q, gal_A, gt); rows = {"A baseline": rA, "B DCKF-Q": ranks(f_kg, gal_A, gt)}
        for al in (0.9, 0.0):
            fr = norm(al * img + (1 - al) * f_cap); gal = torch.cat([fr, norm(al * dr + (1 - al) * dc)]) if n else fr
            tag = "T caption-only" if al == 0.0 else f"C captions a={al}"
            rows[tag] = ranks(f_q, gal, gt)
            if al: rows[f"D both a={al}"] = ranks(f_kg, gal, gt)
        rec = {"gallery": len(names) + n}
        for tag, r in rows.items():
            d, ci, p = paired_bootstrap(rA, r); r1, r5 = (r < 1).mean(), (r < 5).mean()
            rec[tag] = {"R@1": float(r1), "R@5": float(r5), "dR1": float(d), "ci": [float(ci[0]), float(ci[1])], "p": float(p)}
            print(f"{len(names)+n:>8,} {tag:<20} {r1:7.2%} {r5:7.2%}   {d*100:+6.2f}   [{ci[0]*100:+.2f},{ci[1]*100:+.2f}]  {p:.4f}")
        out["scales"].append(rec); print()
    (ROOT/"results"/"flickr_scale_gallery.json").write_text(json.dumps(out, indent=1)); print("wrote results/flickr_scale_gallery.json")
if __name__ == "__main__": main()

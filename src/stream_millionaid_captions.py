"""
E30 — Gallery channel under in-domain distractors (reviewer point 5).

The paper's scale study (E23/E24, Table VIII) only tested the QUERY channel
against distractors. The headline GALLERY channel (caption fusion, Eq. 2)
was never tested at scale. To test it honestly every gallery item, real or
distractor, must receive the same treatment, so this script re-streams the
Million-AID test tarballs (same order as stream_millionaid.py, so image i is
the same image) and stores, per image:
   raw CLIP ViT-B/32 image embedding           -> distractors_millionaid_{N}_raw_v2.fp16.npy
   BLIP-large caption ("an aerial photograph of") text embedding
                                                -> distractors_millionaid_{N}_cap_bliplarge.fp16.npy
   the captions themselves                      -> distractors_millionaid_{N}_captions_bliplarge.jsonl
Fusion (alpha) is applied at evaluation time from the two arrays, so alpha
sweeps need no re-captioning. Memory: BLIP-large fp16 (~1.9 GB) + CLIP; fits
the ~4 GB free on the shared V100s with batch 16.

Usage (dell3, GPU 1):
  CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com \
     python src/stream_millionaid_captions.py --target 100000
Resumable: rerun continues from the meta file's "filled" count by skipping
already-processed members (re-download is the price; nothing is recomputed).
"""
import argparse
import gzip
import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
import clip
from PIL import Image

import stream_millionaid as sm

ROOT = Path(__file__).resolve().parent.parent
BLIP_ID = "Salesforce/blip-image-captioning-large"
PROMPT = "an aerial photograph of"
DIM = 512


@torch.no_grad()
def main(target: int, batch_size: int = 16):
    from transformers import BlipProcessor, BlipForConditionalGeneration
    device = "cuda"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    proc = BlipProcessor.from_pretrained(BLIP_ID)
    blip = BlipForConditionalGeneration.from_pretrained(
        BLIP_ID, torch_dtype=torch.float16).to(device).eval()

    res = ROOT / "results"
    raw_p = res / f"distractors_millionaid_{target}_raw_v2.fp16.npy"
    cap_p = res / f"distractors_millionaid_{target}_cap_bliplarge.fp16.npy"
    txt_p = res / f"distractors_millionaid_{target}_captions_bliplarge.jsonl"
    meta_p = res / f"distractors_millionaid_{target}_bliplarge.meta.json"
    filled = json.loads(meta_p.read_text())["filled"] if meta_p.exists() else 0
    mode = "r+" if raw_p.exists() else "w+"
    raw = np.lib.format.open_memmap(raw_p, mode=mode, dtype=np.float16, shape=(target, DIM))
    cap = np.lib.format.open_memmap(cap_p, mode=mode, dtype=np.float16, shape=(target, DIM))
    txt_f = open(txt_p, "a")
    print(f"resuming at {filled:,}/{target:,}" if filled else f"starting; target {target:,}")

    stream = io.BufferedReader(sm.ChainedStream([sm.BASE + p for p in sm.PARTS]), buffer_size=1 << 22)
    tar = tarfile.open(fileobj=gzip.GzipFile(fileobj=stream), mode="r|")

    seen, batch, t0 = 0, [], time.time()

    def flush(batch):
        nonlocal filled
        pil = [b for b in batch]
        x = torch.stack([preprocess(im) for im in pil]).to(device)
        f_img = model.encode_image(x).float()
        f_img /= f_img.norm(dim=-1, keepdim=True)
        inputs = proc(images=pil, text=[PROMPT] * len(pil), return_tensors="pt", padding=True).to(device, torch.float16)
        out = blip.generate(**inputs, max_new_tokens=30)
        caps = [c.strip() for c in proc.batch_decode(out, skip_special_tokens=True)]
        f_cap = model.encode_text(clip.tokenize(caps, truncate=True).to(device)).float()
        f_cap /= f_cap.norm(dim=-1, keepdim=True)
        take = min(len(batch), target - filled)
        raw[filled:filled + take] = f_img[:take].cpu().numpy().astype(np.float16)
        cap[filled:filled + take] = f_cap[:take].cpu().numpy().astype(np.float16)
        for c in caps[:take]:
            txt_f.write(json.dumps(c) + "\n")
        filled += take
        if filled % 1600 < len(batch):
            el = time.time() - t0
            print(f"  {filled:,}/{target:,}  {el/60:.1f} min  ({filled/max(el,1):.1f} img/s)  e.g. \"{caps[0]}\"", flush=True)
            raw.flush(); cap.flush(); txt_f.flush()
            meta_p.write_text(json.dumps({"filled": filled, "source": "million-aid/test", "captioner": BLIP_ID, "prompt": PROMPT}))

    for member in tar:
        if filled >= target:
            break
        if not member.isfile() or not member.name.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
            continue
        seen += 1
        if seen <= filled:          # already processed in a previous run
            continue
        try:
            batch.append(Image.open(io.BytesIO(tar.extractfile(member).read())).convert("RGB"))
        except Exception:
            continue
        if len(batch) >= batch_size:
            flush(batch); batch = []
    if batch and filled < target:
        flush(batch)
    raw.flush(); cap.flush(); txt_f.close()
    meta_p.write_text(json.dumps({"filled": filled, "source": "million-aid/test", "captioner": BLIP_ID, "prompt": PROMPT}))
    print(f"DONE: {filled:,} images -> {raw_p.name}, {cap_p.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=100_000)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    main(a.target, a.batch_size)

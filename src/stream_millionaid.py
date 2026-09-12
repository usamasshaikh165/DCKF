"""
Stream Million-AID (isaaccorley/million-aid, hf-mirror) test tar parts and
encode a target number of aerial images to CLIP B/32 embeddings WITHOUT
storing any images — the in-domain analog of fetch_distractors.py, for the
remote-sensing scale experiment (F10: in-domain distractors are the ones
that create real difficulty).

Output: results/distractors_millionaid_<target>.fp16.npy  (L2-normed fp16)

The archive is one gzip stream split into parts (test.tar.gzaa..): we chain
the parts into a single file-like object, gunzip on the fly, and read the tar
in streaming mode, stopping as soon as <target> images are encoded.

Usage: python src/stream_millionaid.py --target 100000
"""
import argparse
import gzip
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import requests
import torch
import clip
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BASE = ("https://hf-mirror.com/datasets/isaaccorley/million-aid/"
        "resolve/main/test/")
PARTS = ["test.tar.gzaa", "test.tar.gzab", "test.tar.gzac", "test.tar.gzad"]
DIM = 512


class ChainedStream(io.RawIOBase):
    """Sequential read across the split .gz parts, streamed over HTTP."""

    def __init__(self, urls):
        self.urls = list(urls)
        self.resp = None
        self.it = None

    def _next_part(self):
        if not self.urls:
            return False
        url = self.urls.pop(0)
        self.resp = requests.get(url, stream=True, timeout=120)
        self.resp.raise_for_status()
        self.it = self.resp.iter_content(chunk_size=1 << 20)
        self.buf = b""
        print(f"  [stream] {url.rsplit('/', 1)[-1]}", flush=True)
        return True

    def readable(self):
        return True

    def readinto(self, b):
        while True:
            if self.resp is None and not self._next_part():
                return 0
            try:
                chunk = self.buf or next(self.it)
                self.buf = b""
            except StopIteration:
                self.resp = None
                continue
            n = min(len(chunk), len(b))
            b[:n] = chunk[:n]
            self.buf = chunk[n:]
            return n


@torch.no_grad()
def main(target: int, batch_size: int = 64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    out = ROOT / "results" / f"distractors_millionaid_{target}.fp16.npy"
    feats = np.lib.format.open_memmap(
        out, mode="w+", dtype=np.float16, shape=(target, DIM))
    filled = 0

    stream = io.BufferedReader(ChainedStream([BASE + p for p in PARTS]),
                               buffer_size=1 << 22)
    gz = gzip.GzipFile(fileobj=stream)
    tar = tarfile.open(fileobj=gz, mode="r|")

    batch = []
    for member in tar:
        if filled >= target:
            break
        if not member.isfile():
            continue
        name = member.name.lower()
        if not name.endswith((".jpg", ".jpeg", ".png", ".tif")):
            continue
        try:
            img = Image.open(io.BytesIO(tar.extractfile(member).read()))
            batch.append(preprocess(img.convert("RGB")))
        except Exception:
            continue
        if len(batch) >= batch_size:
            x = torch.stack(batch).to(device)
            f = model.encode_image(x)
            f /= f.norm(dim=-1, keepdim=True)
            take = min(len(batch), target - filled)
            feats[filled:filled + take] = f[:take].cpu().numpy().astype(np.float16)
            filled += take
            batch = []
            if filled % 6400 < batch_size:
                print(f"  encoded {filled:,}/{target:,}", flush=True)
    feats.flush()
    (out.with_suffix(".meta.json")).write_text(
        json.dumps({"filled": filled, "source": "million-aid/test"}))
    print(f"DONE: {filled:,} embeddings -> {out.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=100_000)
    args = ap.parse_args()
    main(args.target)

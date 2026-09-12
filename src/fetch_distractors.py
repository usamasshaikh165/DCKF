"""
Fetch real CLIP ViT-B/32 image embeddings to use as large-gallery distractors.

Source: HF `mlfoundations/datacomp_small` (via hf-mirror.com — HF proper is
blocked/slow on this network). DataComp ships precomputed OpenAI CLIP features
for its 12.8M-sample pool as .npz shards (~3.1 GB each) holding b32_img /
b32_txt / l14_img / l14_txt / dedup arrays.

Space trick: np.savez uses ZIP_STORED (no compression), so each .npy member
sits at a known byte range inside the .npz. We parse the zip central directory
via an HTTP Range request and download ONLY the `b32_img.npy` member
(~518 MB, 505,671 x 512 fp16 per shard) — 6x less than the full shard, and
the images themselves are never touched at all.

Output: results/distractors_datacomp_<target>.fp16.npy — L2-normalized fp16,
row-compatible with the fp32 features from baseline_retrieval.encode_gallery.
Resumable at shard granularity via the .meta.json sidecar.

Torch-free on purpose (README pitfall: faiss+torch can't share a process on
macOS/arm64; this keeps the distractor pipeline usable from FAISS scripts).

Usage:
  python src/fetch_distractors.py --verify          # 10k-vector compatibility check
  python src/fetch_distractors.py --target 1000000  # full download (~1 GB)
"""
import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://hf-mirror.com/datasets/mlfoundations/datacomp_small/resolve/main"
TREE = "https://hf-mirror.com/api/datasets/mlfoundations/datacomp_small/tree/main"
MEMBER = "b32_img.npy"
DIM = 512
LOCAL_FEATS = ROOT / "results" / "image_features_31783.npy"  # fp32, unit-norm


def http_range(url: str, start: int | None, end: int | None, stream=False):
    """Range request; start=None means suffix range (last -end bytes)."""
    rng = f"bytes=-{end}" if start is None else f"bytes={start}-{end}"
    r = requests.get(url, headers={"Range": rng}, timeout=120, stream=stream)
    r.raise_for_status()
    return r


def list_shards() -> list[str]:
    r = requests.get(TREE, timeout=60, allow_redirects=True)
    r.raise_for_status()
    return sorted(e["path"] for e in r.json()
                  if e["type"] == "file" and e["path"].endswith(".npz"))


def locate_member(url: str) -> tuple[int, int]:
    """(data_start, n_rows) of the stored b32_img.npy member inside the npz."""
    tail = http_range(url, None, 65536)
    total = int(tail.headers["Content-Range"].split("/")[-1])
    buf = tail.content
    eocd = buf.rfind(b"PK\x05\x06")
    n_entries = struct.unpack("<H", buf[eocd + 10:eocd + 12])[0]
    cd_off = struct.unpack("<I", buf[eocd + 16:eocd + 20])[0]
    p = len(buf) - (total - cd_off)
    entry = None
    for _ in range(n_entries):
        assert buf[p:p + 4] == b"PK\x01\x02", "central directory parse drifted"
        comp, usize = struct.unpack("<II", buf[p + 20:p + 28])
        nlen, elen, clen = struct.unpack("<HHH", buf[p + 28:p + 34])
        lho = struct.unpack("<I", buf[p + 42:p + 46])[0]
        name = buf[p + 46:p + 46 + nlen].decode()
        if name == MEMBER:
            assert comp == usize, f"{MEMBER} is compressed — range plan invalid"
            entry = (lho, usize)
        p += 46 + nlen + elen + clen
    if entry is None:
        raise RuntimeError(f"{MEMBER} not in {url}")
    lho, usize = entry
    # local file header: 30 bytes + name + extra (extra may differ from CD's)
    lh = http_range(url, lho, lho + 29 + 512).content
    assert lh[:4] == b"PK\x03\x04"
    nlen, elen = struct.unpack("<HH", lh[26:30])
    data_start = lho + 30 + nlen + elen
    # npy header (v1, 128-byte aligned in practice)
    head = http_range(url, data_start, data_start + 127).content
    assert head[:6] == b"\x93NUMPY", "member is not an npy"
    hlen = struct.unpack("<H", head[8:10])[0]
    header = eval(head[10:10 + hlen].decode())  # dict literal from numpy itself
    assert header["descr"] == "<f2" and not header["fortran_order"]
    n, d = header["shape"]
    assert d == DIM
    return data_start + 10 + hlen, n


def fetch_rows(url: str, data_start: int, row0: int, row1: int) -> np.ndarray:
    nbytes = (row1 - row0) * DIM * 2
    start = data_start + row0 * DIM * 2
    r = http_range(url, start, start + nbytes - 1, stream=True)
    buf = bytearray()
    t0, last = time.time(), 0
    for chunk in r.iter_content(chunk_size=1 << 20):
        buf.extend(chunk)
        if len(buf) - last > 50 << 20:
            last = len(buf)
            mbs = len(buf) / (1 << 20) / max(time.time() - t0, 1e-9)
            print(f"    {len(buf) >> 20}/{nbytes >> 20} MB ({mbs:.1f} MB/s)",
                  flush=True)
    assert len(buf) == nbytes, f"short read {len(buf)} != {nbytes}"
    return np.frombuffer(bytes(buf), dtype=np.float16).reshape(-1, DIM)


def normalize_fp16(x: np.ndarray) -> np.ndarray:
    f = x.astype(np.float32)
    f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
    return f.astype(np.float16)


def verify(sample_n: int = 10_000):
    """Fetch a small sample and check it lives in our CLIP B/32 image space."""
    shard = list_shards()[0]
    url = f"{BASE}/{shard}"
    print(f"shard: {shard}")
    data_start, n = locate_member(url)
    print(f"member located: {n:,} rows, data at byte {data_start:,}")
    raw = fetch_rows(url, data_start, 0, sample_n)
    norms = np.linalg.norm(raw.astype(np.float32), axis=1)
    print(f"raw norms: mean {norms.mean():.3f} min {norms.min():.3f} "
          f"max {norms.max():.3f}  (unit => pre-normalized)")
    dc = normalize_fp16(raw).astype(np.float32)

    ours = np.asarray(np.load(LOCAL_FEATS, mmap_mode="r")[:sample_n],
                      dtype=np.float32)
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(dc), 2000), rng.integers(0, len(ours), 2000)
    within_dc = (dc[i] * dc[rng.permutation(i)]).sum(1)
    within_ours = (ours[j] * ours[rng.permutation(j)]).sum(1)
    cross = (dc[i] * ours[j]).sum(1)
    print(f"pairwise cosine  within-datacomp {within_dc.mean():.3f}  "
          f"within-flickr {within_ours.mean():.3f}  cross {cross.mean():.3f}")
    print("CLIP image features form a narrow cone: same-space vectors give "
          "cross ~= within; a different model would give cross ~= 0.")
    ok = abs(cross.mean() - within_ours.mean()) < 0.15 and cross.mean() > 0.2
    print("VERDICT:", "compatible ✓" if ok else "MISMATCH — do not download")
    return ok


def download(target: int):
    out = ROOT / "results" / f"distractors_datacomp_{target}.fp16.npy"
    meta_p = out.with_suffix(".meta.json")
    meta = (json.loads(meta_p.read_text()) if meta_p.exists()
            else {"filled": 0, "shards": []})
    if meta["filled"] >= target:
        print(f"already complete: {out}")
        return
    arr = (np.lib.format.open_memmap(out, mode="r+")
           if out.exists() else
           np.lib.format.open_memmap(out, mode="w+", dtype=np.float16,
                                     shape=(target, DIM)))
    shards = [s for s in list_shards() if s not in meta["shards"]]
    for shard in shards:
        if meta["filled"] >= target:
            break
        url = f"{BASE}/{shard}"
        data_start, n = locate_member(url)
        take = min(n, target - meta["filled"])
        print(f"{shard}: taking {take:,}/{n:,} rows "
              f"({meta['filled']:,}/{target:,} filled)", flush=True)
        step = 100_000                       # ~100 MB per request
        for r0 in range(0, take, step):
            r1 = min(r0 + step, take)
            arr[meta["filled"] + r0: meta["filled"] + r1] = \
                normalize_fp16(fetch_rows(url, data_start, r0, r1))
        arr.flush()
        meta["filled"] += take
        meta["shards"].append(shard)
        meta_p.write_text(json.dumps(meta))
    print(f"done: {meta['filled']:,} distractor vectors in {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--target", type=int, default=1_000_000)
    args = ap.parse_args()
    if args.verify:
        sys.exit(0 if verify() else 1)
    download(args.target)

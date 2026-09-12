"""
Baseline CLIP text-to-image retrieval pipeline (RA-KG-T2I baseline).

Steps (matches proposal Section 5.2 technical route):
  1. Encode all gallery images once with the frozen CLIP image encoder
     (features cached to results/image_features.pt)
  2. Encode text queries with the frozen CLIP text encoder
  3. Cosine similarity search -> ranked images
  4. Report Recall@1/5/10 and median rank

Expected data layout:
  data/images/                 gallery images (*.jpg / *.png)
  data/captions.tsv            TSV: image_filename <TAB> caption
                               (one or more captions per image)

Usage:
  python src/baseline_retrieval.py                  # evaluate Recall@K
  python src/baseline_retrieval.py --query "a dog"  # ad-hoc top-5 search
"""
import argparse
import sys
from pathlib import Path

import torch
import clip
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "images"
CAPTIONS = ROOT / "data" / "captions.tsv"
FEAT_CACHE = ROOT / "results" / "image_features.pt"


def set_dataset(img_dir: str | None, captions: str | None, tag: str = ""):
    """Point the module at a different dataset (e.g. COCO 5k)."""
    global IMG_DIR, CAPTIONS, FEAT_CACHE
    if img_dir:
        IMG_DIR = Path(img_dir).expanduser()
    if captions:
        CAPTIONS = Path(captions).expanduser()
    if tag:
        FEAT_CACHE = ROOT / "results" / f"image_features_{tag}.pt"

DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def _unit32(x):
    """E31 (2026-09-05): normalize in float32, always.

    CLIP on CUDA returns fp16 features; normalizing and caching in fp16
    leaves norms off by up to 6e-4, which alone moved Flickr30k R@1 by
    +0.16 (p<0.001) when fused configs were re-normalized in fp32 while the
    baseline was not. Every feature now leaves this module unit-norm in fp32,
    so baseline and fused configs share one numeric footing."""
    x = x.float()
    return x / x.norm(dim=-1, keepdim=True)


def load_model():
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()
    return model, preprocess


@torch.no_grad()
def encode_gallery(model, preprocess, batch_size: int = 64, limit: int = 0):
    """Encode every image in data/images once; cache features to disk."""
    paths = sorted(p for p in IMG_DIR.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png",
                                           ".tif", ".tiff"})
    if limit:
        paths = paths[:limit]
    if not paths:
        sys.exit(f"No images found in {IMG_DIR} — download a dataset first.")

    cache_file = FEAT_CACHE.with_name(
        f"{FEAT_CACHE.stem}_{len(paths)}{FEAT_CACHE.suffix}")
    if cache_file.exists():
        blob = torch.load(cache_file)
        if blob["names"] == [p.name for p in paths]:
            print(f"Loaded cached features for {len(paths)} images.")
            return blob["names"], _unit32(blob["features"]).to(DEVICE)

    print(f"Encoding {len(paths)} images on {DEVICE} ...")
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = torch.stack([preprocess(Image.open(p).convert("RGB"))
                             for p in paths[i:i + batch_size]]).to(DEVICE)
        f = _unit32(model.encode_image(batch))
        feats.append(f.cpu())
        print(f"  {min(i + batch_size, len(paths))}/{len(paths)}", flush=True)
    features = torch.cat(feats)
    cache_file.parent.mkdir(exist_ok=True)
    torch.save({"names": [p.name for p in paths], "features": features}, cache_file)
    return [p.name for p in paths], features.to(DEVICE)


@torch.no_grad()
def encode_texts(model, texts, batch_size: int = 256):
    feats = []
    for i in range(0, len(texts), batch_size):
        tok = clip.tokenize(texts[i:i + batch_size], truncate=True).to(DEVICE)
        feats.append(_unit32(model.encode_text(tok)))
    return torch.cat(feats)


def evaluate(model, preprocess, limit: int = 0, captions_per_image: int = 5):
    names, img_feats = encode_gallery(model, preprocess, limit=limit)
    name_to_idx = {n: i for i, n in enumerate(names)}

    queries, gt = [], []
    per_image_count = {}
    for line in CAPTIONS.read_text().splitlines():
        if not line.strip():
            continue
        fname, caption = line.split("\t", 1)
        if fname not in name_to_idx:
            continue
        if per_image_count.get(fname, 0) >= captions_per_image:
            continue
        per_image_count[fname] = per_image_count.get(fname, 0) + 1
        queries.append(caption)
        gt.append(name_to_idx[fname])
    print(f"{len(queries)} caption queries over {len(names)} images.")

    txt_feats = encode_texts(model, queries)
    gt_t = torch.tensor(gt, device=DEVICE)

    # chunked rank computation: rank of gt image = #images scoring higher
    ranks = torch.empty(len(queries), device=DEVICE)
    chunk = 2048
    for i in range(0, len(queries), chunk):
        sims = txt_feats[i:i + chunk] @ img_feats.T          # [c, N]
        gt_scores = sims.gather(1, gt_t[i:i + chunk, None])  # [c, 1]
        ranks[i:i + chunk] = (sims > gt_scores).sum(dim=1).float()

    for k in (1, 5, 10):
        r = (ranks < k).float().mean().item()
        print(f"  Recall@{k:<2}: {r:6.2%}")
    print(f"  MedianRank: {ranks.median().item() + 1:.0f}")


def adhoc_query(model, preprocess, text: str, top_k: int = 5):
    names, img_feats = encode_gallery(model, preprocess)
    tf = encode_texts(model, [text])
    sims = (tf @ img_feats.T)[0]
    top = sims.topk(top_k)
    print(f"Top-{top_k} for: {text!r}")
    for score, idx in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {score:.3f}  {names[idx]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", help="run a single ad-hoc text query")
    ap.add_argument("--limit", type=int, default=0,
                    help="use only the first N gallery images (0 = all)")
    ap.add_argument("--captions-per-image", type=int, default=5)
    ap.add_argument("--img-dir", help="override gallery image directory")
    ap.add_argument("--captions", help="override captions.tsv path")
    ap.add_argument("--tag", default="", help="feature-cache tag for this dataset")
    args = ap.parse_args()
    set_dataset(args.img_dir, args.captions, args.tag)
    model, preprocess = load_model()
    if args.query:
        adhoc_query(model, preprocess, args.query)
    else:
        evaluate(model, preprocess, limit=args.limit,
                 captions_per_image=args.captions_per_image)

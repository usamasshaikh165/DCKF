"""
RA-KG-T2I first end-to-end experiment: baseline vs KG-expanded retrieval.

Three query encodings, same gallery (cached CLIP image features):
  A. baseline      CLIP(caption)
  B. expanded      CLIP(enriched caption)            (text-level expansion)
  C. fused         0.6*CLIP(caption) + 0.4*CLIP(enriched), renormalized
                   (embedding-level fusion, as in the pilot study)

Usage:
  python src/kg_expanded_eval.py --dataset coco5k [--captions-per-image 1]
  python src/kg_expanded_eval.py --dataset flickr30k --limit 1000
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_retrieval as br                      # noqa: E402
from expand_query import QueryExpander              # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "coco5k": {
        "img_dir": ROOT / "data/coco5k/images_5k/images_mscoco_2014_5k_test",
        "captions": ROOT / "data/coco5k/captions.tsv",
        "tag": "coco5k",
    },
    "flickr30k": {
        "img_dir": ROOT / "data/images",
        "captions": ROOT / "data/captions.tsv",
        "tag": "",
    },
    "rsicd": {
        "img_dir": ROOT / "data/rsicd/images_test",
        "captions": ROOT / "data/rsicd/captions_test.tsv",
        "tag": "rsicd",
    },
    # E25: official Karpathy 1k test split (mehdidc/retrieval_annotations,
    # the same file clip_benchmark uses; images symlinked from data/images)
    "flickr30k_k1k": {
        "img_dir": ROOT / "data/flickr30k_k1k/images_test",
        "captions": ROOT / "data/flickr30k_k1k/captions_test.tsv",
        "tag": "flickr30k_k1k",
    },
    # E20: KTIR's other two benchmarks (converted by src/convert_rs_datasets.py)
    "rsitmd": {
        "img_dir": ROOT / "data/rsitmd/images_test",
        "captions": ROOT / "data/rsitmd/captions_test.tsv",
        "tag": "rsitmd",
    },
    "ucm": {
        "img_dir": ROOT / "data/ucm/images_test",
        "captions": ROOT / "data/ucm/captions_test.tsv",
        "tag": "ucm",
    },
    "nwpu": {
        "img_dir": ROOT / "data/nwpu/images_test",
        "captions": ROOT / "data/nwpu/captions_test.tsv",
        "tag": "nwpu",
    },
}


def recall_table(txt_feats, img_feats, gt, chunk=2048):
    gt_t = torch.tensor(gt, device=txt_feats.device)
    ranks = torch.empty(len(gt), device=txt_feats.device)
    for i in range(0, len(gt), chunk):
        sims = txt_feats[i:i + chunk] @ img_feats.T
        gt_scores = sims.gather(1, gt_t[i:i + chunk, None])
        ranks[i:i + chunk] = (sims > gt_scores).sum(dim=1).float()
    return {f"R@{k}": (ranks < k).float().mean().item() for k in (1, 5, 10)} | {
        "MedR": ranks.median().item() + 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="coco5k")
    ap.add_argument("--captions-per-image", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="limit gallery images")
    ap.add_argument("--top-k", type=int, default=3, help="concepts per query")
    ap.add_argument("--sim-threshold", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.6,
                    help="fusion weight for original caption embedding")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    br.set_dataset(str(cfg["img_dir"]), str(cfg["captions"]), cfg["tag"])
    model, preprocess = br.load_model()
    names, img_feats = br.encode_gallery(model, preprocess, limit=args.limit)
    name_to_idx = {n: i for i, n in enumerate(names)}

    queries, gt = [], []
    per_img = {}
    for line in Path(cfg["captions"]).read_text().splitlines():
        if not line.strip():
            continue
        fname, caption = line.split("\t", 1)
        if fname not in name_to_idx:
            continue
        if per_img.get(fname, 0) >= args.captions_per_image:
            continue
        per_img[fname] = per_img.get(fname, 0) + 1
        queries.append(caption)
        gt.append(name_to_idx[fname])
    print(f"{len(queries)} queries over {len(names)} images "
          f"({args.dataset}, {args.captions_per_image} cap/img)")

    # --- expansion pass -----------------------------------------------------
    ex = QueryExpander(model=model, device=br.DEVICE,
                       top_k=args.top_k, sim_threshold=args.sim_threshold)
    t0 = time.time()
    enriched, n_changed = [], 0
    for i, q in enumerate(queries):
        r = ex.expand(q)
        enriched.append(r["enriched"])
        n_changed += r["enriched"] != q
        if (i + 1) % 500 == 0:
            print(f"  expanded {i + 1}/{len(queries)} "
                  f"({(time.time() - t0) / (i + 1) * 1000:.0f} ms/query)",
                  flush=True)
    print(f"expansion done: {n_changed}/{len(queries)} queries enriched, "
          f"{(time.time() - t0) / len(queries) * 1000:.0f} ms/query avg")

    # --- encode all three variants -------------------------------------------
    f_base = br.encode_texts(model, queries).float()
    f_exp = br.encode_texts(model, enriched).float()
    f_fuse = args.alpha * f_base + (1 - args.alpha) * f_exp
    f_fuse /= f_fuse.norm(dim=-1, keepdim=True)
    img_feats = img_feats.float()

    rows = {
        "A baseline CLIP": recall_table(f_base, img_feats, gt),
        "B KG text-expanded": recall_table(f_exp, img_feats, gt),
        f"C fused a={args.alpha}": recall_table(f_fuse, img_feats, gt),
    }
    print(f"\n{'method':<22} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'MedR':>6}")
    for name, m in rows.items():
        print(f"{name:<22} {m['R@1']:8.2%} {m['R@5']:8.2%} "
              f"{m['R@10']:8.2%} {m['MedR']:6.0f}")


if __name__ == "__main__":
    main()

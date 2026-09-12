"""
E38 — Does the RSICD caption repair generalize? (review item 2)

The dose-response (BLIP-base -> BLIP-base + aerial prompt -> BLIP-large +
aerial prompt) was run on RSICD only. This script produces the two prompted
caption sets for the three sibling RS galleries so that reviewer_baselines.py
can evaluate the caption channel, its KG increment, and the caption gate on
them at exact parity with RSICD:

  results/blip_captions_{key}_aerial.json         BLIP-base,  "an aerial photograph of"
  results/blip_captions_{key}_large_aerial.json   BLIP-large, "an aerial photograph of"

Usage (dell3):
  python src/rs_large_aerial_captions.py --dataset rsitmd --captioner large
  python src/rs_large_aerial_captions.py --dataset ucm    --captioner base
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_retrieval as br                          # noqa: E402
from kg_expanded_eval import DATASETS                    # noqa: E402
import rsicd_domain_caption as rdc                       # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROMPT = "an aerial photograph of"
CAPTIONERS = {"base": ("Salesforce/blip-image-captioning-base", "aerial"),
              "large": ("Salesforce/blip-image-captioning-large", "large_aerial")}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("rsitmd", "ucm", "nwpu", "rsicd"))
    ap.add_argument("--captioner", required=True, choices=tuple(CAPTIONERS))
    ap.add_argument("--captions-per-image", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    br.set_dataset(str(cfg["img_dir"]), str(cfg["captions"]), cfg["tag"])
    # gallery order and size must match reviewer_baselines.py, which keys the
    # caption file by image name, so only the count matters here
    model, preprocess = br.load_model()
    names, _ = br.encode_gallery(model, preprocess, limit=0)
    key = f"{args.dataset}_{len(names)}_{args.captions_per_image}"
    del model
    torch.cuda.empty_cache()

    blip_id, suffix = CAPTIONERS[args.captioner]
    rdc.BLIP_ID = blip_id
    paths = sorted(p for p in Path(cfg["img_dir"]).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})
    assert len(paths) == len(names), f"{len(paths)} images on disk vs {len(names)} in gallery"
    out = ROOT / "results" / f"blip_captions_{key}_{suffix}.json"
    caps = rdc.blip_conditional(paths, PROMPT, out, batch_size=args.batch_size)
    print(f"{args.dataset}: {len(caps)} captions with {blip_id} -> {out.name}")
    print("examples:", caps[0], "|", caps[len(caps) // 2], "|", caps[-1])


if __name__ == "__main__":
    main()

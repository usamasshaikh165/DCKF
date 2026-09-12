"""
Convert NWPU-Captions (ModelScope YepingZhao mirror: dataset_nwpu.json,
class-keyed with raw..raw_4 caption fields + NWPU_images.tar.gz) into the
project layout: data/nwpu/images_test/ + data/nwpu/captions_test.tsv.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "nwpu_raw"
OUT = ROOT / "data" / "nwpu"


def main():
    (OUT / "images_test").mkdir(parents=True, exist_ok=True)
    tmp = RAW / "unzipped"
    if not tmp.exists():
        tmp.mkdir()
        subprocess.run(["tar", "xzf", str(RAW / "NWPU_images.tar.gz"),
                        "-C", str(tmp)], check=True)
    img_index = {p.name: p for p in tmp.rglob("*")
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif"}}
    print(f"{len(img_index)} images in archive")

    data = json.loads((RAW / "dataset_nwpu.json").read_text())
    lines, n_img, missing = [], 0, 0
    for cls, items in data.items():
        for im in items:
            if im.get("split") != "test":
                continue
            fname = im["filename"]
            src = img_index.get(fname)
            if src is None:
                missing += 1
                continue
            dst = OUT / "images_test" / fname
            if not dst.exists():
                shutil.copy(src, dst)
            n_img += 1
            for k in ("raw", "raw_1", "raw_2", "raw_3", "raw_4"):
                cap = im.get(k, "").strip().replace("\t", " ")
                if cap:
                    lines.append(f"{fname}\t{cap}")
    (OUT / "captions_test.tsv").write_text("\n".join(lines) + "\n")
    print(f"test: {n_img} images, {len(lines)} captions, {missing} missing")


if __name__ == "__main__":
    main()

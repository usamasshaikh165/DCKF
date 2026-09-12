"""
Convert RSITMD / UCM-Captions (canonical dataset_*.json + images.zip, from
ModelScope YepingZhao mirrors of the original releases) into the project
layout used by every eval script:

  data/{name}/images_test/          test-split images only
  data/{name}/captions_test.tsv     filename<TAB>caption (5 per image)

Split comes from the JSON's own per-image "split" field (the standard
AMFMN/Karpathy-style split shipped by the dataset authors).

Usage (on dell3):
  python src/convert_rs_datasets.py data/rsitmd_raw rsitmd
  python src/convert_rs_datasets.py data/ucm_raw ucm
"""
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(raw_dir: str, name: str):
    raw = ROOT / raw_dir if not Path(raw_dir).is_absolute() else Path(raw_dir)
    out = ROOT / "data" / name
    (out / "images_test").mkdir(parents=True, exist_ok=True)

    js = max((p for p in raw.glob("dataset*.json")
              if "infos" not in p.name), key=lambda p: p.stat().st_size)
    data = json.loads(js.read_text())["images"]
    splits = {}
    for im in data:
        splits.setdefault(im.get("split", "train"), 0)
        splits[im.get("split", "train")] += 1
    print(f"{name}: {len(data)} images, splits={splits}")

    tmp = raw / "unzipped"
    if not tmp.exists():
        zf = next(raw.glob("*.zip"))
        with zipfile.ZipFile(zf) as z:
            z.extractall(tmp)
    # index every jpg/tif under the extraction dir by basename
    img_index = {p.name: p for p in tmp.rglob("*")
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif"}}
    print(f"  {len(img_index)} images in archive")

    lines, n_img, missing = [], 0, 0
    for im in data:
        if im.get("split") != "test":
            continue
        fname = im["filename"]
        src = img_index.get(fname)
        if src is None:
            missing += 1
            continue
        dst = out / "images_test" / fname
        if not dst.exists():
            shutil.copy(src, dst)
        n_img += 1
        for s in im["sentences"]:
            cap = s["raw"].strip().replace("\t", " ").replace("\n", " ")
            if cap:
                lines.append(f"{fname}\t{cap}")
    (out / "captions_test.tsv").write_text("\n".join(lines) + "\n")
    print(f"  test: {n_img} images, {len(lines)} captions, {missing} missing")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

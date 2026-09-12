"""
RA-KG-T2I ablation harness (proposal §5.4: ablations a-e).

Design: expansion is the expensive step (~220 ms/query), so expansions are
computed ONCE and cached to results/expansions_<dataset>.json with full
metadata (concept, relation, seed, prior, sim). Every ablation below is then
pure re-encoding / vector math over the cache:

  1. baseline            no expansion
  2. expand-all          enrich every query (template T1)
  3. selective           enrich only short queries (< N content words)
  4. per-relation        only IsA / only AtLocation / only UsedFor / ...
  5. templates           T1 ', a scene with c1, c2, c3' | T2 ', c1' | T3 ', c1, c2'
  6. alpha sweep         fused = a*base + (1-a)*expanded, a in 0.5..0.9

Usage:
  python src/ablations.py --dataset coco5k --captions-per-image 1
  python src/ablations.py --dataset flickr30k --limit 1000
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_retrieval as br                       # noqa: E402
from expand_query import QueryExpander, STOPWORDS     # noqa: E402
from kg_expanded_eval import DATASETS, recall_table   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
_token_re = re.compile(r"[a-z]+")


def content_len(caption: str) -> int:
    return sum(1 for t in _token_re.findall(caption.lower())
               if t not in STOPWORDS and len(t) > 2)


def load_queries(cfg, names, captions_per_image):
    name_to_idx = {n: i for i, n in enumerate(names)}
    queries, gt = [], []
    per_img = {}
    for line in Path(cfg["captions"]).read_text().splitlines():
        if not line.strip():
            continue
        fname, caption = line.split("\t", 1)
        if fname not in name_to_idx:
            continue
        if per_img.get(fname, 0) >= captions_per_image:
            continue
        per_img[fname] = per_img.get(fname, 0) + 1
        queries.append(caption)
        gt.append(name_to_idx[fname])
    return queries, gt


def get_expansions(queries, model, device, cache_key) -> list[list[dict]]:
    """Concept lists per query, cached on disk keyed by query-set hash."""
    cache = ROOT / "results" / f"expansions_{cache_key}.json"
    qhash = hashlib.md5("\n".join(queries).encode()).hexdigest()[:10]
    if cache.exists():
        blob = json.loads(cache.read_text())
        if blob["qhash"] == qhash:
            print(f"Loaded cached expansions ({cache.name})")
            return blob["expansions"]
    ex = QueryExpander(model=model, device=device, top_k=5, sim_threshold=0.5)
    out, t0 = [], time.time()
    for i, q in enumerate(queries):
        out.append(ex.expand(q)["concepts"])
        if (i + 1) % 1000 == 0:
            print(f"  expanded {i+1}/{len(queries)} "
                  f"({(time.time()-t0)/(i+1)*1000:.0f} ms/q)", flush=True)
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps({"qhash": qhash, "expansions": out}))
    print(f"Expansion pass done ({(time.time()-t0)/len(queries)*1000:.0f} ms/q), cached.")
    return out


def apply_template(caption, concepts, template):
    if not concepts:
        return caption
    names = [c["concept"] for c in concepts]
    if template == "T1":
        return f"{caption}, a scene with " + ", ".join(names[:3])
    if template == "T2":
        return f"{caption}, {names[0]}"
    if template == "T3":
        return f"{caption}, " + ", ".join(names[:2])
    raise ValueError(template)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="coco5k")
    ap.add_argument("--captions-per-image", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--short-len", type=int, default=5,
                    help="selective: expand only queries with < N content words")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    br.set_dataset(str(cfg["img_dir"]), str(cfg["captions"]), cfg["tag"])
    model, preprocess = br.load_model()
    names, img_feats = br.encode_gallery(model, preprocess, limit=args.limit)
    img_feats = img_feats.float()
    queries, gt = load_queries(cfg, names, args.captions_per_image)
    print(f"{len(queries)} queries over {len(names)} images ({args.dataset})")

    key = f"{args.dataset}_{len(names)}_{args.captions_per_image}"
    expansions = get_expansions(queries, model, br.DEVICE, key)

    f_base = br.encode_texts(model, queries).float()
    results = {"baseline": recall_table(f_base, img_feats, gt)}

    def eval_texts(tag, texts):
        f = br.encode_texts(model, texts).float()
        results[tag] = recall_table(f, img_feats, gt)
        return f

    # --- expand-all, template variants --------------------------------------
    f_T1 = eval_texts("expand-all T1",
                      [apply_template(q, e, "T1") for q, e in zip(queries, expansions)])
    eval_texts("expand-all T2",
               [apply_template(q, e, "T2") for q, e in zip(queries, expansions)])
    eval_texts("expand-all T3",
               [apply_template(q, e, "T3") for q, e in zip(queries, expansions)])

    # --- selective: only short queries ---------------------------------------
    sel = [apply_template(q, e, "T2") if content_len(q) < args.short_len else q
           for q, e in zip(queries, expansions)]
    n_sel = sum(1 for a, b in zip(sel, queries) if a != b)
    eval_texts(f"selective<{args.short_len}w (n={n_sel})", sel)

    # --- per-relation contribution -------------------------------------------
    for rel in ("IsA", "AtLocation", "UsedFor", "PartOf", "CapableOf"):
        texts = [apply_template(q, [c for c in e if c["relation"] == rel], "T2")
                 for q, e in zip(queries, expansions)]
        n = sum(1 for t, q in zip(texts, queries) if t != q)
        eval_texts(f"only-{rel} (n={n})", texts)

    # --- alpha sweep on best template (T1 features already computed) ---------
    for a in (0.5, 0.6, 0.7, 0.8, 0.9):
        f = a * f_base + (1 - a) * f_T1
        f /= f.norm(dim=-1, keepdim=True)
        results[f"fused a={a}"] = recall_table(f, img_feats, gt)

    # --- report ---------------------------------------------------------------
    print(f"\n{'config':<28} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'MedR':>6}")
    base = results["baseline"]["R@1"]
    for tag, m in results.items():
        delta = (m["R@1"] - base) * 100
        print(f"{tag:<28} {m['R@1']:8.2%} {m['R@5']:8.2%} {m['R@10']:8.2%} "
              f"{m['MedR']:6.0f}   dR@1 {delta:+.2f}")

    out_csv = ROOT / "results" / f"ablations_{key}.csv"
    with open(out_csv, "w") as f:
        f.write("config,R@1,R@5,R@10,MedR\n")
        for tag, m in results.items():
            f.write(f"\"{tag}\",{m['R@1']:.4f},{m['R@5']:.4f},"
                    f"{m['R@10']:.4f},{m['MedR']:.0f}\n")
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()

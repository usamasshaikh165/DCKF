"""
E26 — Knowledge-selection baselines at exact parity (KAIS/KBS comparative-
study requirement; directly answers KTIR's random-triplet-selection ablation).

All configs share: the SAME cached ConceptNet candidate pools
(results/expansions_{key}.json), the SAME sim floor (0.55), the SAME number
of kept concepts (k=2), the SAME text template (build_text) and the SAME
gentle fusion (a_q). Only the knowledge-SELECTION strategy differs — so any
metric difference isolates the selection component:

  ours       relation-aware: reweighted priors x CLIP sim (E7 config)
  original   original ConceptNet-style priors x CLIP sim (pre-F4 naive)
  random     uniform random from the pool (KTIR's selection strategy)
  wordnet    source swap: first WordNet synonym of each cached seed
             (no relation weighting, no drift control — tests whether the
             KG + relation machinery beats a thesaurus)
  none       baseline (no expansion) for reference

Both retrieval directions + mR via bidirectional_eval.eval_config.
Randomness is seeded per query index (reproducible; no wall-clock use).

Usage:
  python src/knowledge_baselines.py --dataset flickr30k --limit 1000
  python src/knowledge_baselines.py --dataset rsicd --limit 0
Output: printed table + results/baselines_{key}.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_retrieval as br                          # noqa: E402
from ablations import load_queries                       # noqa: E402
from kg_expanded_eval import DATASETS                    # noqa: E402
from advanced_expansion import (REWEIGHTED_PRIORS, rerank, build_text,  # noqa: E402
                                paired_bootstrap)
from bidirectional_eval import eval_config               # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MIN_SIM, K = 0.55, 2


def select_original(pool):
    """Original priors (stored per-candidate as 'score') x CLIP sim."""
    scored = [(c["score"] * c["sim"], c) for c in pool if c["sim"] >= MIN_SIM]
    scored.sort(key=lambda t: -t[0])
    return [c for s, c in scored[:K] if s > 0]


def select_random(pool, q_idx, seed=0):
    cands = [c for c in pool if c["sim"] >= MIN_SIM]
    rng = random.Random(seed * 1_000_003 + q_idx)
    return rng.sample(cands, min(K, len(cands)))


def select_wordnet(pool, query):
    """Source swap: first WordNet synonym of each cached seed word."""
    from nltk.corpus import wordnet as wn
    seeds, seen = [], set()
    for c in pool:
        s = c.get("seed", "")
        if s and s not in seen:
            seen.add(s)
            seeds.append(s)
    out = []
    for s in seeds:
        for syn in wn.synsets(s.replace(" ", "_")):
            names = [l.name().replace("_", " ").lower() for l in syn.lemmas()]
            novel = [n for n in names
                     if n != s.lower() and n not in query.lower()]
            if novel:
                out.append({"concept": novel[0]})
                break
        if len(out) >= K:
            break
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="flickr30k")
    ap.add_argument("--captions-per-image", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--alpha-q", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=("fusion", "text"), default="fusion",
                    help="fusion: gentle embedding fusion (default, the "
                         "thesis operating point). text: expanded text "
                         "REPLACES the query (no fusion, no gate) — the "
                         "operating point where selection strategy matters")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    br.set_dataset(str(cfg["img_dir"]), str(cfg["captions"]), cfg["tag"])
    model, preprocess = br.load_model()
    names, img_feats = br.encode_gallery(model, preprocess, limit=args.limit)
    img_feats = img_feats.float()
    queries, gt = load_queries(cfg, names, args.captions_per_image)
    __import__('advanced_expansion').set_boot_clusters(gt)  # E40 cluster bootstrap
    n_images = len(names)

    key = f"{args.dataset}_{n_images}_{args.captions_per_image}"
    exp_file = ROOT / "results" / f"expansions_{key}.json"
    if not exp_file.exists():
        sys.exit(f"missing expansion cache: {exp_file}")
    pools = json.loads(exp_file.read_text())["expansions"]
    assert len(pools) == len(queries), \
        f"cache/query mismatch: {len(pools)} vs {len(queries)}"
    print(f"{len(queries)} captions over {n_images} images ({args.dataset})")

    try:
        from nltk.corpus import wordnet as wn  # noqa: F401
        wn.synsets("test")
        have_wordnet = True
    except Exception as e:
        print(f"NOTE: WordNet unavailable ({e}) — wordnet baseline skipped")
        have_wordnet = False

    f_q = br.encode_texts(model, queries).float()

    def fused(selections):
        texts = [build_text(q, c) for q, c in zip(queries, selections)]
        f_exp = br.encode_texts(model, texts).float()
        if args.mode == "text":
            return f_exp
        f = args.alpha_q * f_q + (1 - args.alpha_q) * f_exp
        return f / f.norm(dim=-1, keepdim=True)

    strategies = {"none (baseline)": None}
    strategies["random (KTIR-style)"] = [
        select_random(p, i, args.seed) for i, p in enumerate(pools)]
    strategies["original priors"] = [select_original(p) for p in pools]
    if have_wordnet:
        strategies["wordnet synonyms"] = [
            select_wordnet(p, q) for p, q in zip(pools, queries)]
    strategies["relation-aware (ours)"] = [
        rerank(p, REWEIGHTED_PRIORS) for p in pools]

    results = {}
    for tag, sel in strategies.items():
        tf = f_q if sel is None else fused(sel)
        results[tag] = eval_config(tf, img_feats, gt, n_images)
        if sel is not None:
            n_used = sum(1 for s in sel if s)
            results[tag]["queries_expanded"] = n_used

    hdr = (f"\n{'strategy':<24} | {'t2i R@1':>8} {'R@5':>7} {'R@10':>7}"
           f" | {'i2t R@1':>8} {'R@5':>7} {'R@10':>7} | {'mR':>6} | {'#exp':>5}")
    print(hdr); print("-" * len(hdr))
    for tag, m in results.items():
        t, i = m["t2i"], m["i2t"]
        print(f"{tag:<24} | {t['R@1']:8.2%} {t['R@5']:7.2%} {t['R@10']:7.2%} "
              f"| {i['R@1']:8.2%} {i['R@5']:7.2%} {i['R@10']:7.2%} "
              f"| {m['mR']:6.2%} | {m.get('queries_expanded', '-'):>5}")

    boots = {}
    base = results["none (baseline)"]
    ours = results["relation-aware (ours)"]
    for tag, m in results.items():
        if tag == "none (baseline)":
            continue
        d, (lo, hi), p = paired_bootstrap(base["_ranks_t2i"], m["_ranks_t2i"])
        boots[f"{tag} vs baseline"] = {"delta_R@1": d, "ci95": [lo, hi], "p": p}
        print(f"bootstrap {tag} vs baseline, t2i R@1: {d*100:+.2f} "
              f"[{lo*100:+.2f}, {hi*100:+.2f}], p={p:.4f}")
    for tag in list(strategies)[1:-1]:
        d, (lo, hi), p = paired_bootstrap(results[tag]["_ranks_t2i"],
                                          ours["_ranks_t2i"])
        boots[f"ours vs {tag}"] = {"delta_R@1": d, "ci95": [lo, hi], "p": p}
        print(f"bootstrap ours vs {tag}, t2i R@1: {d*100:+.2f} "
              f"[{lo*100:+.2f}, {hi*100:+.2f}], p={p:.4f}")

    out = {
        "dataset": args.dataset, "n_images": n_images,
        "n_captions": len(queries),
        "captions_per_image": args.captions_per_image,
        "alpha_q": args.alpha_q, "min_sim": MIN_SIM, "k": K,
        "seed": args.seed, "device": br.DEVICE, "mode": args.mode,
        "strategies": {t: {k: v for k, v in m.items() if not k.startswith("_")}
                       for t, m in results.items()},
        "bootstraps": boots,
    }
    suffix = "" if args.mode == "fusion" else f"_{args.mode}"
    out_file = ROOT / "results" / f"baselines_{key}{suffix}.json"
    out_file.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

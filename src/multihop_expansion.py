"""
E16 — Multi-hop KG reasoning pilot (proposal Problem 3).

Does 2-hop expansion (seed -> neighbor -> neighbor-of-neighbor, e.g.
ship -> vessel -> transport) add context that 1-hop misses, especially for
short/ambiguous queries? Query-side, per the proposal.

Scoring: path score = prod(reweighted relation priors along path) x direction
discounts x per-hop decay, then x CLIP-sim(query, concept) drift control —
the same relation-aware machinery, extended one hop.

Configs (all text-fused at E7-best a=0.9):
  A  baseline
  B  1-hop reweighted top-2, min_sim .55       (E7 reference)
  C  1+2-hop pooled, same rerank               (does 2-hop enrich the pool?)
  D  2-hop backoff: only where 1-hop kept none (multi-hop as repair)
  E  2-hop only for short queries (content_len < 6)
Bootstrap: best of C/D/E vs B (the multi-hop increment) and vs A.

Usage:
  python src/multihop_expansion.py --dataset rsicd --captions-per-image 5
  python src/multihop_expansion.py --dataset flickr30k --limit 1000 --captions-per-image 5
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_retrieval as br                        # noqa: E402
from ablations import load_queries, content_len        # noqa: E402
from kg_expanded_eval import DATASETS                  # noqa: E402
from advanced_expansion import (REWEIGHTED_PRIORS, build_text,  # noqa: E402
                                ranks_of, paired_bootstrap)
from expand_query import QueryExpander                 # noqa: E402
from kg_lookup import neighbors, _connect              # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HOP2_DECAY = 0.7
K1_POOL, K1_EXPAND, K2 = 8, 3, 5   # hop1 pool, hop1 seeds for hop2, hop2 per node


def clean(c: str) -> bool:
    return len(c.split("_")) <= 3


def hop_candidates(seed: str, con) -> dict:
    """{concept: {score, relation, hop, via}} for 1-hop + 2-hop from `seed`."""
    def edges(node):
        out = {}
        for rel, other, direction in neighbors(node, con):
            prior = REWEIGHTED_PRIORS.get(rel, 0.0)
            if prior <= 0 or other == node or not clean(other):
                continue
            s = prior if direction == "out" else prior * 0.8
            if s > out.get(other, (0, None))[0]:
                out[other] = (s, rel)
        return out

    cands = {}
    h1 = edges(seed)
    for c, (s, rel) in sorted(h1.items(), key=lambda t: -t[1][0])[:K1_POOL]:
        cands[c] = {"score": s, "relation": rel, "hop": 1, "via": seed}
    for c1, (s1, _) in sorted(h1.items(), key=lambda t: -t[1][0])[:K1_EXPAND]:
        for c2, (s2, rel2) in sorted(edges(c1).items(),
                                     key=lambda t: -t[1][0])[:K2]:
            if c2 == seed or c2 in h1:
                continue
            s = s1 * s2 * HOP2_DECAY
            if c2 not in cands or s > cands[c2]["score"]:
                cands[c2] = {"score": s, "relation": rel2, "hop": 2, "via": c1}
    return cands


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="rsicd")
    ap.add_argument("--captions-per-image", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--min-sim", type=float, default=0.55)
    ap.add_argument("--keep", type=int, default=2)
    ap.add_argument("--hop2-parity", action="store_true",
                    help="score 2-hop by last edge prior only (no path "
                         "product/decay) — 'best possible shot' ablation")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    br.set_dataset(str(cfg["img_dir"]), str(cfg["captions"]), cfg["tag"])
    model, preprocess = br.load_model()
    names, img_feats = br.encode_gallery(model, preprocess, limit=args.limit)
    img_feats = img_feats.float()
    queries, gt = load_queries(cfg, names, args.captions_per_image)
    f_q = br.encode_texts(model, queries).float()
    print(f"{len(queries)} queries over {len(names)} images ({args.dataset})")

    # --- candidate pools (cached) --------------------------------------------
    key = f"{args.dataset}_{len(names)}_{args.captions_per_image}"
    pool_file = ROOT / "results" / f"multihop_pools_{key}.json"
    if pool_file.exists():
        pools = json.loads(pool_file.read_text())["pools"]
        print("Loaded cached multihop pools.")
    else:
        ex = QueryExpander()             # seeds only; no model needed
        con = _connect()
        t0, pools = time.time(), []
        for i, q in enumerate(queries):
            merged = {}
            for seed in ex.extract_seeds(q):
                for c, meta in hop_candidates(seed, con).items():
                    if c not in merged or meta["score"] > merged[c]["score"]:
                        merged[c] = meta
            pools.append(merged)
            if (i + 1) % 500 == 0:
                print(f"  pooled {i + 1}/{len(queries)} "
                      f"({(time.time()-t0)/(i+1)*1000:.0f} ms/q)", flush=True)
        con.close()
        pool_file.write_text(json.dumps({"pools": pools}))
        print(f"pools built: {(time.time()-t0)/len(queries)*1000:.0f} ms/query")

    # --- CLIP drift control: sim(query, concept), computed in bulk -----------
    uniq = {}
    for p in pools:
        for c in p:
            uniq.setdefault(c, len(uniq))
    f_c = br.encode_texts(model,
                          [c.replace("_", " ") for c in uniq]).float()  # [U, d]
    print(f"{len(uniq)} unique candidate concepts "
          f"(pool avg {sum(len(p) for p in pools)/len(pools):.1f}/query)")

    def kept_for(i, allow_hop2, backoff=False):
        p = pools[i]
        idx = [uniq[c] for c in p]
        if not idx:
            return []
        sims = (f_q[i] @ f_c[idx].T).tolist()
        scored = []
        for (c, meta), s in zip(p.items(), sims):
            if s < args.min_sim:
                continue
            score = meta["score"]
            if args.hop2_parity and meta["hop"] == 2:
                score = REWEIGHTED_PRIORS.get(meta["relation"], 0.0)
            scored.append((score * s, c, meta["hop"]))
        scored.sort(key=lambda t: -t[0])
        h1 = [(f, c) for f, c, h in scored if h == 1]
        h2 = [(f, c) for f, c, h in scored if h == 2]
        if backoff:
            pool = h1 if h1 else h2
        elif allow_hop2:
            pool = sorted(h1 + h2, key=lambda t: -t[0])
        else:
            pool = h1
        return [{"concept": c.replace("_", " ")} for _, c in pool[:args.keep]]

    lens = [content_len(q) for q in queries]
    variants = {
        "B 1-hop (E7 ref)": [kept_for(i, False) for i in range(len(queries))],
        "C 1+2-hop pooled": [kept_for(i, True) for i in range(len(queries))],
        "D 2-hop backoff": [kept_for(i, False, backoff=True)
                            for i in range(len(queries))],
        "E 2-hop if short": [kept_for(i, lens[i] < 6)
                             for i in range(len(queries))],
    }

    results, rank_store = {}, {}

    def add(tag_, txt_feats):
        r = ranks_of(txt_feats, img_feats, gt)
        rank_store[tag_] = r
        results[tag_] = {f"R@{k}": float((r < k).mean()) for k in (1, 5, 10)}

    add("A baseline", f_q)
    for tag_, kept in variants.items():
        n_exp = sum(1 for k in kept if k)
        n_h2 = sum(1 for i, k in enumerate(kept)
                   if any(c["concept"].replace(" ", "_") in pools[i]
                          and pools[i][c["concept"].replace(" ", "_")]["hop"] == 2
                          for c in k))
        f_e = br.encode_texts(
            model, [build_text(q, k) for q, k in zip(queries, kept)]).float()
        fused = args.alpha * f_q + (1 - args.alpha) * f_e
        fused /= fused.norm(dim=-1, keepdim=True)
        add(f"{tag_} (n={n_exp},h2={n_h2})", fused)

    base_r1 = results["A baseline"]["R@1"]
    ref = next(t for t in results if t.startswith("B"))
    ref_r1 = results[ref]["R@1"]
    print(f"\nfusion a={args.alpha}, min_sim={args.min_sim}, keep={args.keep}")
    print(f"{'config':<34} {'R@1':>8} {'R@5':>8} {'R@10':>8}  dR@1(A)  dR@1(B)")
    for tag_, m in results.items():
        print(f"{tag_:<34} {m['R@1']:8.2%} {m['R@5']:8.2%} {m['R@10']:8.2%} "
              f" {(m['R@1']-base_r1)*100:+.2f}   {(m['R@1']-ref_r1)*100:+.2f}")

    best_mh = max((t for t in results if t[0] in "CDE"),
                  key=lambda t: results[t]["R@1"])
    for label, base in ((f"vs {ref} (multi-hop increment)", ref),
                       ("vs A baseline", "A baseline")):
        mean_d, (lo, hi), p = paired_bootstrap(rank_store[base], rank_store[best_mh])
        print(f"\nPaired bootstrap {best_mh} {label}, R@1:")
        print(f"  mean delta {mean_d*100:+.2f} pts, "
              f"95% CI [{lo*100:+.2f}, {hi*100:+.2f}], p(delta<=0) = {p:.4f}")


if __name__ == "__main__":
    main()

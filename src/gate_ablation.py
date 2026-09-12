"""
E37 — Isolating the drift gate (review item 1).

The parity study (knowledge_baselines.py) applies the 0.55 similarity gate in
BOTH operating points, and the cached candidate pools already carry a 0.5
floor, so no configuration in the paper has ever run FUSION WITHOUT A GATE.
This script rebuilds the candidate pool with no floor and a deep top-k, then
runs the full factorial at exact parity (same pool, k=2, same template,
same beta):

  scoring   x  gate tau                     x  operating point
  typed w(r)*cos   |  0 (off), .3, .4, .45, .5, .55 (used), .6, .65  |  fusion b=0.9
  prior-only w(r)  |  0 (off), .55                                   |  raw text

Also verifies that "typed, tau=0.55" on the new pool reproduces the paper's
cached selection (same concepts for >= 99% of queries) so the two runs are
comparable.

Usage (dell3):
  python src/gate_ablation.py --dataset flickr30k_k1k
  python src/gate_ablation.py --dataset rsicd
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_retrieval as br                                    # noqa: E402
from ablations import load_queries                                 # noqa: E402
from kg_expanded_eval import DATASETS                              # noqa: E402
from advanced_expansion import (REWEIGHTED_PRIORS, build_text,      # noqa: E402
                                ranks_of, paired_bootstrap)
from expand_query import QueryExpander                             # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BETA, K = 0.9, 2
TAUS = (0.0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65)


def norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def rec(r):
    return {f"R@{k}": float((r < k).mean()) for k in (1, 5, 10)}


POOL5 = 5   # the paper's cache keeps the top-5 candidates by ORIGINAL prior x cos


def select(pool, tau, scoring, prefilter=False):
    """k=2 selection at gate tau. scoring='typed' -> w(r)*cos; 'prior' -> w(r) only
    (ties broken by ConceptNet edge weight), i.e. typed scoring switched off.
    prefilter=True reproduces the paper's pipeline exactly: stage 1 keeps the
    top-5 candidates with sim >= tau ranked by original_prior*cos ('final'),
    stage 2 reranks those by the relation-aware weights. prefilter=False uses
    the whole no-floor pool (pool depth becomes a separate factor)."""
    if prefilter:
        # paper pipeline: cache floor 0.5 -> top-5 by original prior x cos -> gate tau.
        # Lowering tau below 0.5 lowers the cache floor with it, so tau=0 removes both.
        stage1 = [c for c in pool if c["sim"] >= min(tau, 0.5)]
        pool = sorted(stage1, key=lambda c: -c["final"])[:POOL5]
    cands = [c for c in pool if c["sim"] >= tau]
    if scoring == "typed":
        scored = [(REWEIGHTED_PRIORS.get(c["relation"], 0.0) * c["sim"], c) for c in cands]
    else:
        scored = [(REWEIGHTED_PRIORS.get(c["relation"], 0.0) + 1e-3 * c["score"], c) for c in cands]
    scored.sort(key=lambda t: -t[0])
    return [c for s, c in scored[:K] if s > 0]


def nofloor_pool(queries, model, key, top_k):
    cache = ROOT / "results" / f"expansions_{key}_nofloor.json"
    qhash = hashlib.md5("\n".join(queries).encode()).hexdigest()[:10]
    if cache.exists():
        blob = json.loads(cache.read_text())
        if blob["qhash"] == qhash and blob.get("top_k") == top_k:
            print(f"Loaded no-floor pool ({cache.name})")
            return blob["expansions"]
    ex = QueryExpander(model=model, device=br.DEVICE, top_k=top_k, sim_threshold=-1.0)
    out, t0 = [], time.time()
    for i, q in enumerate(queries):
        out.append(ex.expand(q)["concepts"])
        if (i + 1) % 1000 == 0:
            print(f"  expanded {i+1}/{len(queries)} ({(time.time()-t0)/(i+1)*1000:.0f} ms/q)", flush=True)
    cache.write_text(json.dumps({"qhash": qhash, "top_k": top_k, "expansions": out}))
    print(f"No-floor pool built ({(time.time()-t0)/len(queries)*1000:.0f} ms/q), cached.")
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--captions-per-image", type=int, default=5)
    ap.add_argument("--pool-topk", type=int, default=50)
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    br.set_dataset(str(cfg["img_dir"]), str(cfg["captions"]), cfg["tag"])
    model, preprocess = br.load_model()
    names, img_feats = br.encode_gallery(model, preprocess, limit=args.limit)
    img_feats = img_feats.float()
    queries, gt = load_queries(cfg, names, args.captions_per_image)
    __import__('advanced_expansion').set_boot_clusters(gt)  # E40 cluster bootstrap
    key = f"{args.dataset}_{len(names)}_{args.captions_per_image}"
    print(f"{len(queries)} queries over {len(names)} images ({args.dataset})")

    pools = nofloor_pool(queries, model, key, args.pool_topk)
    sims_all = [c["sim"] for p in pools for c in p]
    print(f"pool: mean {sum(len(p) for p in pools)/len(pools):.1f} cands/query; "
          f"{sum(s < 0.5 for s in sims_all)} of {len(sims_all)} candidates below the old 0.5 floor")

    # --- parity check against the paper's cached pool -------------------------------------
    paper_file = ROOT / "results" / f"expansions_{key}.json"
    parity = None
    if paper_file.exists():
        paper = json.loads(paper_file.read_text())["expansions"]
        same = 0
        for p_new, p_old in zip(pools, paper):
            emu = sorted([c for c in p_new if c["sim"] >= 0.5], key=lambda c: -c["final"])[:5]
            a = [c["concept"] for c in select(emu, 0.55, "typed")]
            b = [c["concept"] for c in select(p_old, 0.55, "typed")]
            same += (a == b)
        parity = same / len(pools)
        print(f"parity with paper cache: identical tau=0.55 selection for {parity*100:.1f}% of queries")

    f_q = norm(br.encode_texts(model, queries).float())
    r_base = ranks_of(f_q, img_feats, gt)
    out = {"dataset": args.dataset, "key": key, "n_queries": len(queries), "n_images": len(names),
           "pool_topk": args.pool_topk, "parity_with_paper_cache": parity,
           "baseline": rec(r_base), "rows": []}

    def run(scoring, tau, mode, prefilter, ref=None, ref_label=None):
        sel = [select(p, tau, scoring, prefilter) for p in pools]
        n_exp = sum(1 for s in sel if s)
        f_e = norm(br.encode_texts(model, [build_text(q, c) for q, c in zip(queries, sel)]).float())
        f = f_e if mode == "text" else norm(BETA * f_q + (1 - BETA) * f_e)
        r = ranks_of(f, img_feats, gt)
        d, ci, p = paired_bootstrap(r_base, r)
        pool_tag = "top5" if prefilter else "deep"
        label = f"{pool_tag} {scoring:<5} tau={tau:<4} {mode:<6}"
        entry = {"pool": pool_tag, "scoring": scoring, "tau": tau, "mode": mode, "n_expanded": n_exp,
                 **rec(r), "dR1_vs_baseline": float(d), "ci": [float(ci[0]), float(ci[1])], "p_le0": float(p)}
        line = (f"  {label} n_exp={n_exp:<5} R@1 {rec(r)['R@1']*100:6.2f} | vs base {d*100:+.2f} "
                f"CI[{ci[0]*100:+.2f},{ci[1]*100:+.2f}] p={p:.4f}")
        if ref is not None:
            d2, ci2, p2 = paired_bootstrap(ref, r)
            entry.update({"dR1_vs_ref": float(d2), "ci_vs_ref": [float(ci2[0]), float(ci2[1])],
                          "p_le0_vs_ref": float(p2), "ref": ref_label})
            line += f" | vs {ref_label} {d2*100:+.2f} CI[{ci2[0]*100:+.2f},{ci2[1]*100:+.2f}] p={p2:.4f}"
        print(line, flush=True)
        out["rows"].append(entry)
        return r

    print(f"\nbaseline R@1 {rec(r_base)['R@1']*100:.2f}")
    refs = {}
    for prefilter, tag in ((True, "top5 (paper pipeline: top-5 by original prior x cos, then rerank)"),
                           (False, "deep (no top-5 prefilter: rerank over the whole pool)")):
        print(f"\n[A] {tag}; typed scoring, fusion b=0.9, gate sweep (tau=0 is gate OFF)")
        for tau in TAUS:
            r = run("typed", tau, "fusion", prefilter)
            if tau == 0.55:
                refs[prefilter] = r
        print("[A'] the missing cell: gate-off fusion vs gate-on fusion")
        run("typed", 0.0, "fusion", prefilter, ref=refs[prefilter], ref_label="tau=.55")
        print("[B] typed scoring, raw text, gate on/off")
        for tau in (0.55, 0.0):
            run("typed", tau, "text", prefilter)
        print("[C] prior-only scoring (cos removed from the score), gate on/off")
        for tau in (0.55, 0.0):
            run("prior", tau, "fusion", prefilter, ref=refs[prefilter], ref_label="typed,tau=.55")
        for tau in (0.55, 0.0):
            run("prior", tau, "text", prefilter)
    print("\n[D] pool depth alone: deep vs top5 at tau=0.55, typed, fusion")
    run("typed", 0.55, "fusion", False, ref=refs[True], ref_label="top5,tau=.55")

    for prefilter in (True, False):
        changed = sum(1 for p in pools
                      if [c["concept"] for c in select(p, 0.0, "typed", prefilter)]
                      != [c["concept"] for c in select(p, 0.55, "typed", prefilter)])
        out[f"selection_changed_by_gate_{'top5' if prefilter else 'deep'}"] = changed
        print(f"gate changes the k=2 selection ({'top5' if prefilter else 'deep'} pool) for "
              f"{changed} of {len(pools)} queries ({changed/len(pools)*100:.1f}%)")

    dst = ROOT / "results" / f"gate_ablation_{key}.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"wrote {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

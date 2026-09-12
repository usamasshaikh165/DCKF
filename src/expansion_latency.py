
"""
E39 — Query-expansion latency, as deployed vs. with a concept-embedding cache
(review item 7: the paper quotes 160-230 ms/query, dominated by drift scoring,
and must reconcile that with its own critique of LLM latency).

Measures, on the same machine the paper's figure came from (Apple M3, MPS):
  (a) baseline query encode          CLIP text encode of the query alone
  (b) expansion as deployed          seeds -> ConceptNet lookup -> encode query+all
                                     candidates -> gate/rerank -> encode expanded text
  (c) expansion with concept cache   same, but candidate embeddings come from a
                                     dict filled once offline (one encode per unique
                                     concept), so per query only the expanded text
                                     is encoded

Reports medians over N queries and the size of the offline cache.
Usage:  venv/bin/python src/expansion_latency.py --dataset rsicd --n 500
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_retrieval as br                                   # noqa: E402
from kg_expanded_eval import DATASETS                             # noqa: E402
from ablations import load_queries                                # noqa: E402
from advanced_expansion import REWEIGHTED_PRIORS, rerank, build_text  # noqa: E402
from expand_query import QueryExpander                            # noqa: E402
import clip                                                       # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def enc(model, texts, device):
    tok = clip.tokenize(texts, truncate=True).to(device)
    f = model.encode_text(tok).float()
    return f / f.norm(dim=-1, keepdim=True)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rsicd")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--captions-per-image", type=int, default=5)
    args = ap.parse_args()
    cfg = DATASETS[args.dataset]
    br.set_dataset(str(cfg["img_dir"]), str(cfg["captions"]), cfg["tag"])
    model, _ = br.load_model()
    device = br.DEVICE
    names = sorted(p.name for p in Path(cfg["img_dir"]).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})
    queries, _ = load_queries(cfg, names, args.captions_per_image)
    queries = queries[:args.n]
    ex = QueryExpander(model=model, device=device, top_k=5, sim_threshold=0.5)

    # warm-up
    for q in queries[:10]:
        enc(model, [q], device); ex.expand(q)
    sync()

    t_base, t_dep, t_cache, t_lookup = [], [], [], []
    n_cands = []

    # (a) baseline: encode query only
    for q in queries:
        sync(); t0 = time.perf_counter(); enc(model, [q], device); sync()
        t_base.append(time.perf_counter() - t0)

    # (b) as deployed: QueryExpander.expand (lookup + encode query+candidates + gate),
    #     then rerank + encode expanded text
    for q in queries:
        sync(); t0 = time.perf_counter()
        pool = ex.expand(q)["concepts"]
        sel = rerank(pool, REWEIGHTED_PRIORS, min_sim=0.55, k=2)
        enc(model, [build_text(q, sel)], device)
        sync(); t_dep.append(time.perf_counter() - t0)

    # (c) with a concept-embedding cache filled offline
    #     offline part: unique concepts over the whole candidate universe
    cand_lists = []
    for q in queries:
        seeds = ex.extract_seeds(q)
        from kg_lookup import extract_expansions
        cands = []
        for s in seeds:
            for e in extract_expansions(s, top_k=8, con=ex.con):
                cands.append({**e, "seed": s})
        cand_lists.append(cands)
    uniq = sorted({c["concept"] for cl in cand_lists for c in cl})
    sync(); t0 = time.perf_counter()
    cache = {}
    for i in range(0, len(uniq), 256):
        f = enc(model, uniq[i:i + 256], device)
        for c, v in zip(uniq[i:i + 256], f):
            cache[c] = v
    sync(); t_offline = time.perf_counter() - t0

    for q in queries:
        sync(); t0 = time.perf_counter()
        # lookup (SQLite) timed separately as well
        t1 = time.perf_counter()
        seeds = ex.extract_seeds(q)
        cands = []
        for s in seeds:
            for e in extract_expansions(s, top_k=8, con=ex.con):
                cands.append({**e, "seed": s})
        t_lookup.append(time.perf_counter() - t1)
        fq = enc(model, [q], device)[0]                  # needed for retrieval anyway
        if cands:
            M = torch.stack([cache[c["concept"]] for c in cands])
            sims = (M @ fq).tolist()
            for c, s in zip(cands, sims):
                c["sim"] = s
            sel = rerank(cands, REWEIGHTED_PRIORS, min_sim=0.55, k=2)
        else:
            sel = []
        if sel:
            enc(model, [build_text(q, sel)], device)
        sync(); t_cache.append(time.perf_counter() - t0)
        n_cands.append(len(cands))

    ms = lambda xs: statistics.median(xs) * 1000  # noqa: E731
    p90 = lambda xs: sorted(xs)[int(0.9 * len(xs))] * 1000  # noqa: E731
    out = {
        "dataset": args.dataset, "n_queries": len(queries), "device": device,
        "median_ms": {"baseline_query_encode": ms(t_base), "expansion_as_deployed": ms(t_dep),
                      "expansion_with_concept_cache": ms(t_cache), "conceptnet_lookup_only": ms(t_lookup)},
        "p90_ms": {"baseline_query_encode": p90(t_base), "expansion_as_deployed": p90(t_dep),
                   "expansion_with_concept_cache": p90(t_cache)},
        "mean_candidates_per_query": statistics.mean(n_cands),
        "concept_cache": {"unique_concepts": len(uniq), "offline_seconds": t_offline},
    }
    print(json.dumps(out, indent=1))
    dst = ROOT / "results" / f"expansion_latency_{args.dataset}.json"
    dst.write_text(json.dumps(out, indent=1))
    print("wrote", dst.relative_to(ROOT))


if __name__ == "__main__":
    main()

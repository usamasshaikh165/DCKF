"""
E14 — close the F13b gap: apply the E7 treatment (relation reweighting,
gentle fusion, selectivity) to KG expansion of gallery BLIP captions.

E13 found captions give +2.36 R@1 but NAIVE KG on top loses −0.45. Query-side
history says naive expansion always loses (F1) until reweighting + gentle
fusion + selectivity fix it (E7). This applies the same three fixes on the
gallery side. Image-side fusion fixed at E13-best a=0.9 throughout.

Configs (all vs B, the caption-only reference):
  A  baseline
  B  caption-only, a=0.9                      (E13 best: +2.36)
  C  naive KG text-replace (E13 C)            (reference: the failure)
  D  reweighted top-2 text-replace (min_sim .55)
  E  channel fusion: f_txt = g*f_cap + (1-g)*f_D, g in {0.8, 0.9}
  F  selective: D-text only where best reranked prior*sim >= thresh, else cap
Bootstrap: best KG config vs B (the KG increment — the number that matters).

Usage:
  python src/gallery_kg_rerank.py --dataset flickr30k --limit 1000 --captions-per-image 5
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_retrieval as br                        # noqa: E402
from ablations import load_queries                     # noqa: E402
from kg_expanded_eval import DATASETS                  # noqa: E402
from advanced_expansion import (REWEIGHTED_PRIORS, rerank, build_text,  # noqa: E402
                                ranks_of, paired_bootstrap)
from expand_query import QueryExpander                 # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="flickr30k")
    ap.add_argument("--captions-per-image", type=int, default=5)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.9, help="image-side fusion")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    br.set_dataset(str(cfg["img_dir"]), str(cfg["captions"]), cfg["tag"])
    model, preprocess = br.load_model()
    names, img_feats = br.encode_gallery(model, preprocess, limit=args.limit)
    img_feats = img_feats.float()
    queries, gt = load_queries(cfg, names, args.captions_per_image)
    f_q = br.encode_texts(model, queries).float()

    key = f"{args.dataset}_{len(names)}_{args.captions_per_image}"
    cap_file = ROOT / "results" / f"blip_captions_{key}.json"
    if not cap_file.exists():
        sys.exit(f"{cap_file} missing — run gallery_caption_expansion.py first")
    cap_map = json.loads(cap_file.read_text())
    caps = [cap_map[n] for n in names]
    print(f"{len(queries)} queries, {len(names)} images, BLIP captions loaded")

    # --- broad candidate pool per caption (cached, with relation+sim meta) ---
    pool_file = ROOT / "results" / f"gallery_cap_expansions_{key}.json"
    if pool_file.exists():
        pools = json.loads(pool_file.read_text())["pools"]
        print("Loaded cached expansion pools.")
    else:
        ex = QueryExpander(model=model, device=br.DEVICE,
                           top_k=8, sim_threshold=0.30)
        t0, pools = time.time(), []
        for i, c in enumerate(caps):
            pools.append(ex.expand(c)["concepts"])
            if (i + 1) % 250 == 0:
                print(f"  expanded {i + 1}/{len(caps)}", flush=True)
        pool_file.write_text(json.dumps({"pools": pools}))
        print(f"expansion pools built: "
              f"{(time.time() - t0) / len(caps) * 1000:.0f} ms/image")

    a = args.alpha
    f_cap = br.encode_texts(model, caps).float()

    def img_fuse(f_txt):
        fused = a * img_feats + (1 - a) * f_txt
        return fused / fused.norm(dim=-1, keepdim=True)

    results, rank_store = {}, {}

    def add(tag_, gallery):
        r = ranks_of(f_q, gallery, gt)
        rank_store[tag_] = r
        results[tag_] = {f"R@{k}": float((r < k).mean()) for k in (1, 5, 10)}

    add("A baseline", img_feats)
    add("B cap-only", img_fuse(f_cap))

    # C: naive (E13 replication): top-3 by final score, sim>=0.5, replace text
    naive = [[c for c in p if c["sim"] >= 0.5][:3] for p in pools]
    f_C = br.encode_texts(model, [build_text(c, n) for c, n in zip(caps, naive)]).float()
    add("C naive KG", img_fuse(f_C))

    # D: reweighted top-2, min_sim 0.55, replace text
    rew = [rerank(p, REWEIGHTED_PRIORS, min_sim=0.55, k=2) for p in pools]
    n_d = sum(1 for c in rew if c)
    f_D = br.encode_texts(model, [build_text(c, n) for c, n in zip(caps, rew)]).float()
    add(f"D reweighted (n={n_d})", img_fuse(f_D))

    # E: gentle channel fusion of cap-emb with D-emb before image fusion
    for g in (0.8, 0.9):
        f_ch = g * f_cap + (1 - g) * f_D
        f_ch /= f_ch.norm(dim=-1, keepdim=True)
        add(f"E channel g={g}", img_fuse(f_ch))

    # F: selective — D-text only where best reweighted concept is confident
    for thresh in (0.55, 0.65):
        mask, texts = [], []
        for c, con in zip(caps, rew):
            conf = con and (REWEIGHTED_PRIORS.get(con[0]["relation"], 0)
                            * con[0]["sim"]) >= thresh
            mask.append(bool(conf))
            texts.append(build_text(c, con) if conf else c)
        f_F = br.encode_texts(model, texts).float()
        add(f"F selective t={thresh} (n={sum(mask)})", img_fuse(f_F))

    base_r1 = results["A baseline"]["R@1"]
    cap_r1 = results["B cap-only"]["R@1"]
    print(f"\nimage-side fusion a={a}")
    print(f"{'config':<26} {'R@1':>8} {'R@5':>8} {'R@10':>8}   dR@1(A)  dR@1(B)")
    for tag_, m in results.items():
        print(f"{tag_:<26} {m['R@1']:8.2%} {m['R@5']:8.2%} {m['R@10']:8.2%} "
              f"  {(m['R@1'] - base_r1) * 100:+.2f}   {(m['R@1'] - cap_r1) * 100:+.2f}")

    best_kg = max((t for t in results if t[0] in "CDEF"),
                  key=lambda t: results[t]["R@1"])
    for label, ref in (("vs B cap-only (KG increment)", "B cap-only"),
                       ("vs A baseline", "A baseline")):
        mean_d, (lo, hi), p = paired_bootstrap(rank_store[ref], rank_store[best_kg])
        print(f"\nPaired bootstrap {best_kg} {label}, R@1:")
        print(f"  mean delta {mean_d*100:+.2f} pts, "
              f"95% CI [{lo*100:+.2f}, {hi*100:+.2f}], p(delta<=0) = {p:.4f}")


if __name__ == "__main__":
    main()

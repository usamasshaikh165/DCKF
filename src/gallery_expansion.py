"""
Gallery-side offline KG expansion (E12 pilot) — enrich IMAGE embeddings at
index time, zero query-side latency. Motivated by F9 (query-side ceiling) and
F11 (query-side gain washes out at scale).

Leakage note: gallery captions are the eval queries, so image-side text must
come from the image itself. Pipeline per gallery image:
  1. Zero-shot tag: top-T concepts from a ConceptNet-derived vocabulary
     (high-degree English concepts, CLIP-scored against the image)
  2. Expand tags via ConceptNet neighbors, scored
     reweighted_prior (F4: IsA demoted) x CLIP-sim(image, neighbor)
  3. Fuse: img' = a*img + (1-a)*txt_emb("a photo of tags[, neighbors]")

Configs (x alpha sweep):
  A  baseline image features
  B  fused with tag text only        (control: tagging effect, no KG)
  C  fused with tag + KG neighbors   (the thesis mechanism)
Significance: paired bootstrap, best C vs A, and best C vs best B (KG increment).

Usage:
  python src/gallery_expansion.py --dataset flickr30k --limit 1000 --captions-per-image 5
"""
import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_retrieval as br                        # noqa: E402
from ablations import load_queries                     # noqa: E402
from kg_expanded_eval import DATASETS                  # noqa: E402
from advanced_expansion import (REWEIGHTED_PRIORS, ranks_of,  # noqa: E402
                                paired_bootstrap)
from kg_lookup import neighbors, _connect              # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WORD_RE = re.compile(r"^[a-z]+(_[a-z]+)?$")


def build_vocab(size: int) -> list[str]:
    """Top-`size` ConceptNet concepts by degree (1-2 word, alphabetic)."""
    con = _connect()
    rows = con.execute(
        """SELECT concept, SUM(cnt) AS deg FROM (
             SELECT start AS concept, COUNT(*) AS cnt FROM edges GROUP BY start
             UNION ALL
             SELECT end, COUNT(*) FROM edges GROUP BY end)
           GROUP BY concept ORDER BY deg DESC""").fetchall()
    con.close()
    vocab = [c for c, _ in rows if WORD_RE.match(c) and len(c) >= 3]
    return vocab[:size]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="flickr30k")
    ap.add_argument("--captions-per-image", type=int, default=5)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--vocab-size", type=int, default=10_000)
    ap.add_argument("--tags-per-image", type=int, default=3)
    ap.add_argument("--neighbors-per-image", type=int, default=2)
    ap.add_argument("--alphas", default="0.7,0.8,0.9,0.95")
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]

    cfg = DATASETS[args.dataset]
    br.set_dataset(str(cfg["img_dir"]), str(cfg["captions"]), cfg["tag"])
    model, preprocess = br.load_model()
    names, img_feats = br.encode_gallery(model, preprocess, limit=args.limit)
    img_feats = img_feats.float()
    queries, gt = load_queries(cfg, names, args.captions_per_image)
    f_q = br.encode_texts(model, queries).float()
    print(f"{len(queries)} queries over {len(names)} images ({args.dataset})")

    # --- 1. zero-shot tagging against ConceptNet vocabulary ------------------
    t0 = time.time()
    vocab = build_vocab(args.vocab_size)
    v_texts = [f"a photo of {c.replace('_', ' ')}" for c in vocab]
    f_vocab = br.encode_texts(model, v_texts).float()          # [V, d]
    tag_sims = img_feats @ f_vocab.T                            # [N, V]
    top = tag_sims.topk(args.tags_per_image, dim=1)
    tags = [[vocab[j] for j in row] for row in top.indices.tolist()]
    print(f"vocab {len(vocab)}, tagged {len(names)} images "
          f"({(time.time() - t0):.1f}s total)")

    # --- 2. KG neighbors of tags, scored prior x CLIP-sim(image, neighbor) ---
    t0 = time.time()
    con = _connect()
    cand_per_img = []           # list of {neighbor: prior} per image
    uniq = {}
    for img_tags in tags:
        cands = {}
        for seed in img_tags:
            for rel, other, direction in neighbors(seed, con):
                prior = REWEIGHTED_PRIORS.get(rel, 0.0)
                if prior <= 0 or other == seed or not WORD_RE.match(other):
                    continue
                score = prior if direction == "out" else prior * 0.8
                if score > cands.get(other, 0.0):
                    cands[other] = score
        for c in cands:
            uniq.setdefault(c, len(uniq))
        cand_per_img.append(cands)
    con.close()

    uniq_list = list(uniq)
    f_nb = br.encode_texts(
        model, [f"a photo of {c.replace('_', ' ')}" for c in uniq_list]).float()
    nb_sims = img_feats @ f_nb.T                                # [N, U]
    chosen = []
    for i, cands in enumerate(cand_per_img):
        scored = sorted(
            ((prior * nb_sims[i, uniq[c]].item(), c) for c, prior in cands.items()),
            key=lambda t: -t[0])
        chosen.append([c for _, c in scored[:args.neighbors_per_image]])
    per_img_ms = (time.time() - t0) / len(names) * 1000
    print(f"KG expansion: {len(uniq_list)} unique neighbors, "
          f"{per_img_ms:.0f} ms/image (offline, index-time only)")

    # cache for inspection / reuse (convention: results/, keyed by dataset+size)
    key = f"{args.dataset}_{len(names)}_{args.captions_per_image}"
    cache = ROOT / "results" / f"gallery_expansions_{key}.json"
    cache.write_text(json.dumps(
        {"tags": dict(zip(names, tags)), "neighbors": dict(zip(names, chosen)),
         "vocab_size": len(vocab), "ms_per_image": per_img_ms}, indent=1))

    # --- 3. build gallery texts + fuse ---------------------------------------
    def gtexts(with_kg: bool) -> list[str]:
        out = []
        for tg, nb in zip(tags, chosen):
            terms = [t.replace("_", " ") for t in tg]
            if with_kg:
                terms += [n.replace("_", " ") for n in nb]
            out.append("a photo of " + ", ".join(dict.fromkeys(terms)))
        return out

    f_tag = br.encode_texts(model, gtexts(with_kg=False)).float()
    f_kg = br.encode_texts(model, gtexts(with_kg=True)).float()

    results, rank_store = {}, {}

    def add(tag_, gallery):
        r = ranks_of(f_q, gallery, gt)
        rank_store[tag_] = r
        results[tag_] = {f"R@{k}": float((r < k).mean()) for k in (1, 5, 10)}

    add("A baseline", img_feats)
    for a in alphas:
        for label, f_txt in (("B tags", f_tag), ("C tags+KG", f_kg)):
            fused = a * img_feats + (1 - a) * f_txt
            fused /= fused.norm(dim=-1, keepdim=True)
            add(f"{label} a={a}", fused)

    # score-level fusion: sidesteps the CLIP modality gap (text and image
    # embeddings occupy different cones; mixing vectors drags all images in a
    # shared text direction). Blend similarities instead of embeddings.
    def add_scorefused(tag_, f_txt, beta):
        s = (1 - beta) * (f_q @ img_feats.T) + beta * (f_q @ f_txt.T)
        gt_t = torch.tensor(gt, device=s.device)
        gt_scores = s.gather(1, gt_t[:, None])
        r = (s > gt_scores).sum(dim=1).float().cpu().numpy()
        rank_store[tag_] = r
        results[tag_] = {f"R@{k}": float((r < k).mean()) for k in (1, 5, 10)}

    for beta in (0.05, 0.1, 0.15, 0.2):
        add_scorefused(f"D score tags b={beta}", f_tag, beta)
        add_scorefused(f"E score tags+KG b={beta}", f_kg, beta)

    base_r1 = results["A baseline"]["R@1"]
    print(f"\n{'config':<20} {'R@1':>8} {'R@5':>8} {'R@10':>8}   dR@1")
    for tag_, m in results.items():
        print(f"{tag_:<20} {m['R@1']:8.2%} {m['R@5']:8.2%} {m['R@10']:8.2%} "
              f"  {(m['R@1'] - base_r1) * 100:+.2f}")

    best_kg = max((t for t in results if t[0] in "CE"),
                  key=lambda t: results[t]["R@1"])
    best_tag = max((t for t in results if t[0] in "BD"),
                   key=lambda t: results[t]["R@1"])
    for label, a_tag, b_tag in (("best KG vs baseline", "A baseline", best_kg),
                                ("best KG vs best tags-only (KG increment)",
                                 best_tag, best_kg)):
        mean_d, (lo, hi), p = paired_bootstrap(rank_store[a_tag], rank_store[b_tag])
        print(f"\nPaired bootstrap {label} [{b_tag} vs {a_tag}], R@1:")
        print(f"  mean delta {mean_d*100:+.2f} pts, "
              f"95% CI [{lo*100:+.2f}, {hi*100:+.2f}], p(delta<=0) = {p:.4f}")


if __name__ == "__main__":
    main()

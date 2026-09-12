"""
E29 — Reviewer-requested baselines and sensitivity, all on cached features.

Answers four review points in one pass (same protocol as E25/E26: cached
gallery features, cached ConceptNet pools, cached BLIP captions, paired
bootstrap over queries, t2i R@1 primary; R@5/R@10 reported):

  PRF   embedding-space pseudo-relevance feedback (Rocchio): the query is
        fused with the centroid of its top-k baseline retrievals using the
        SAME beta as DCKF-Q. Knowledge-free query-side competitor.
  CAP   caption-only gallery (alpha=0, caption embedding replaces the image)
        and the full alpha sweep for the gallery channel.
  BETA/K  beta sweep and k (kept concepts) sweep for the query channel.
  GATE  a drift gate on the caption channel: gallery item k keeps its fused
        embedding only if cos(I_k, C_k) >= threshold (quantiles of the
        observed cosine distribution); otherwise the raw image embedding.

Usage (dell3):
  python src/reviewer_baselines.py --dataset flickr30k_k1k
  python src/reviewer_baselines.py --dataset rsicd --cap-suffix large_aerial
Output: printed tables + results/reviewer_{key}[_{cap-suffix}].json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_retrieval as br                                    # noqa: E402
from ablations import load_queries                                 # noqa: E402
from kg_expanded_eval import DATASETS                              # noqa: E402
from advanced_expansion import (REWEIGHTED_PRIORS, rerank, build_text,   # noqa: E402
                                ranks_of, paired_bootstrap)

ROOT = Path(__file__).resolve().parent.parent


def norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def rec(r):
    return {f"R@{k}": float((r < k).mean()) for k in (1, 5, 10)}


def load_caps(path, names):
    obj = json.loads(path.read_text())
    if isinstance(obj, dict) and "captions" in obj and isinstance(obj["captions"], (dict, list)):
        obj = obj["captions"]
    if isinstance(obj, dict):
        return [obj[n] for n in names]
    assert len(obj) == len(names), "caption list length mismatch"
    return list(obj)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--captions-per-image", type=int, default=5)
    ap.add_argument("--cap-suffix", default="", help="e.g. large_aerial, aerial")
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--alpha", type=float, default=0.9)
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
    print(f"{len(queries)} queries over {n_images} images ({args.dataset})")

    f_q = norm(br.encode_texts(model, queries).float())
    r_base = ranks_of(f_q, img_feats, gt)
    out = {"dataset": args.dataset, "key": key, "n_queries": len(queries),
           "n_images": n_images, "baseline": rec(r_base), "rows": []}

    def row(section, label, r, ref=None, ref_label="baseline"):
        d, ci, p = paired_bootstrap(r_base, r)
        entry = {"section": section, "label": label, **rec(r),
                 "dR1_vs_baseline": float(d), "ci": [float(ci[0]), float(ci[1])], "p_le0": float(p)}
        line = (f"  {label:<38} R@1 {rec(r)['R@1']:.4f}  R@5 {rec(r)['R@5']:.4f}  "
                f"R@10 {rec(r)['R@10']:.4f} | vs base {d*100:+.2f} CI[{ci[0]*100:+.2f},{ci[1]*100:+.2f}] p={p:.4f}")
        if ref is not None:
            d2, ci2, p2 = paired_bootstrap(ref, r)
            entry.update({"dR1_vs_" + ref_label: float(d2), "p_le0_vs_" + ref_label: float(p2)})
            line += f" | vs {ref_label} {d2*100:+.2f} p={p2:.4f}"
        print(line)
        out["rows"].append(entry)
        return r

    # ---------------------------------------------------------------- query channel: DCKF-Q reference
    exp_file = ROOT / "results" / f"expansions_{key}.json"
    r_kg = None
    if exp_file.exists():
        pools = json.loads(exp_file.read_text())["expansions"]
        assert len(pools) == len(queries), "expansion cache does not match query list"

        def kg_feats(beta, k):
            rew = [rerank(e, REWEIGHTED_PRIORS, min_sim=0.55, k=k) for e in pools]
            f_e = norm(br.encode_texts(model, [build_text(q, c) for q, c in zip(queries, rew)]).float())
            return norm(beta * f_q + (1 - beta) * f_e), sum(1 for c in rew if c)

        print("\n[Q] DCKF-Q reference and beta / k sweeps")
        fq_ref, n_exp = kg_feats(args.beta, 2)
        r_kg = row("Q", f"DCKF-Q beta={args.beta} k=2 (ref; {n_exp} expanded)", ranks_of(fq_ref, img_feats, gt))
        for beta in (0.5, 0.7, 0.8, 0.95, 1.0):
            f, n_exp = kg_feats(beta, 2)
            row("BETA", f"DCKF-Q beta={beta} k=2", ranks_of(f, img_feats, gt))
        for k in (1, 3, 4):
            f, n_exp = kg_feats(args.beta, k)
            row("K", f"DCKF-Q beta={args.beta} k={k} ({n_exp} expanded)", ranks_of(f, img_feats, gt))
    else:
        print(f"NOTE: {exp_file.name} missing; query-channel rows skipped")

    # ---------------------------------------------------------------- PRF (Rocchio in embedding space)
    print("\n[PRF] pseudo-relevance feedback: query fused with centroid of top-k baseline images")
    sims = f_q @ img_feats.T
    for k in (1, 3, 5, 10):
        top = sims.topk(k, dim=1).indices                    # [Q, k]
        cent = norm(img_feats[top].mean(dim=1))              # [Q, D]
        for beta in ((args.beta,) if k != 3 else (0.7, 0.8, args.beta, 0.95)):
            f = norm(beta * f_q + (1 - beta) * cent)
            row("PRF", f"PRF k={k} beta={beta}", ranks_of(f, img_feats, gt), ref=r_kg, ref_label="DCKF-Q")
    if r_kg is not None:
        top = sims.topk(3, dim=1).indices
        cent = norm(img_feats[top].mean(dim=1))
        f = norm(args.beta * fq_ref + (1 - args.beta) * cent)
        row("PRF", f"DCKF-Q + PRF k=3 beta={args.beta}", ranks_of(f, img_feats, gt), ref=r_kg, ref_label="DCKF-Q")

    # ---------------------------------------------------------------- gallery channel: caption-only, alpha sweep, gate
    suffix = f"_{args.cap_suffix}" if args.cap_suffix else ""
    cap_file = ROOT / "results" / f"blip_captions_{key}{suffix}.json"
    if cap_file.exists():
        caps = load_caps(cap_file, names)
        f_cap = norm(br.encode_texts(model, caps).float())
        print(f"\n[CAP] gallery channel with {cap_file.name}; alpha sweep (alpha=0 is caption-only, 1.0 is baseline)")
        r_C = None
        for alpha in (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0):
            fused = norm(alpha * img_feats + (1 - alpha) * f_cap)
            r = row("ALPHA", f"gallery alpha={alpha}" + ("  (caption-only)" if alpha == 0 else ""), ranks_of(f_q, fused, gt))
            if alpha == args.alpha:
                r_C = r
        # caption gate: keep fusion only where cos(I_k, C_k) >= quantile threshold
        cos = (img_feats * f_cap).sum(dim=-1)
        qs = torch.quantile(cos, torch.tensor([0.1, 0.25, 0.5, 0.75], device=cos.device))
        print(f"\n[GATE] cos(I_k, C_k): mean {cos.mean():.3f}  quantiles 10/25/50/75% = "
              + "/".join(f"{v:.3f}" for v in qs.tolist()))
        fused_full = norm(args.alpha * img_feats + (1 - args.alpha) * f_cap)
        for frac, thr in zip((0.1, 0.25, 0.5, 0.75), qs.tolist()):
            keep = (cos >= thr).unsqueeze(1)
            gated = torch.where(keep, fused_full, img_feats)
            row("GATE", f"caption gate: drop lowest {int(frac*100)}% (thr {thr:.3f}) alpha={args.alpha}",
                ranks_of(f_q, gated, gt), ref=r_C, ref_label="C-ungated")
            if r_kg is not None:
                row("GATE", f"  + DCKF-Q (D gated {int(frac*100)}%)", ranks_of(fq_ref, gated, gt), ref=r_C, ref_label="C-ungated")
        if r_kg is not None:
            row("D", f"D = DCKF-Q + gallery alpha={args.alpha} (ungated)", ranks_of(fq_ref, fused_full, gt), ref=r_C, ref_label="C-ungated")
    else:
        print(f"NOTE: {cap_file.name} missing; gallery-channel rows skipped")

    dst = ROOT / "results" / f"reviewer_{key}{suffix}.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

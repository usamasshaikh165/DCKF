"""
E24 — Domain-tuned frozen backbone: RemoteCLIP ViT-B-32 on all 4 RS datasets.

Answers the committee question "RS-tuned CLIPs exist — why not use them?"
with data: swap the frozen general CLIP for the frozen, published
RemoteCLIP-ViT-B-32 checkpoint (Liu et al., TGRS 2024) and rerun the E22
protocol unchanged. Still zero training on our side — the pipeline is
backbone-agnostic. Expansion TEXTS held fixed (B/32-selected concepts,
cached), same as E19, so only the encoder changes.

F15b prediction: RemoteCLIP's text tower is domain-adapted -> the text
channel is less deficient -> the query-KG increment should SHRINK vs the
general-CLIP +0.077 pooled effect. Either outcome is a finding:
  - survives  -> KG stacks with domain adaptation
  - shrinks   -> repair mechanism confirmed on a third backbone

Setup (once, on dell3):
  pip install open_clip_torch
  curl -L -o ~/RA-KG-T2I/models/RemoteCLIP-ViT-B-32.pt \
    https://hf-mirror.com/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-B-32.pt

Usage: python src/remoteclip_rs.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_retrieval as br                        # noqa: E402
from ablations import load_queries                     # noqa: E402
from kg_expanded_eval import DATASETS                  # noqa: E402
from advanced_expansion import (REWEIGHTED_PRIORS, rerank, build_text,  # noqa: E402
                                ranks_of, paired_bootstrap)

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "models" / "RemoteCLIP-ViT-B-32.pt"
RS = [("rsicd", 0, 5), ("rsitmd", 0, 5), ("ucm", 0, 5), ("nwpu", 0, 5)]


class TokenizerShim:
    """br.encode_texts calls clip.tokenize; give open_clip the same interface."""

    def __init__(self):
        self._tok = open_clip.get_tokenizer("ViT-B-32")

    def __call__(self, texts, truncate=True):
        return self._tok(texts)


@torch.no_grad()
def main():
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32")
    ckpt = torch.load(CKPT, map_location="cpu")
    msg = model.load_state_dict(ckpt)
    model = model.to(br.DEVICE).eval()
    print(f"RemoteCLIP ViT-B-32 loaded on {br.DEVICE} ({msg})")

    # monkey-patch the tokenizer used inside br.encode_texts
    import baseline_retrieval
    import clip as openai_clip
    baseline_retrieval.clip = type(sys)("clip_shim")
    baseline_retrieval.clip.tokenize = TokenizerShim()
    baseline_retrieval.clip.load = openai_clip.load  # unused, keep sane

    per_ds = {}
    for ds, limit, cpi in RS:
        cfg = DATASETS[ds]
        br.set_dataset(str(cfg["img_dir"]), str(cfg["captions"]),
                       f"{cfg['tag']}_remoteclip")
        names, img_feats = br.encode_gallery(model, preprocess, limit=limit)
        img_feats = img_feats.float()
        queries, gt = load_queries(cfg, names, cpi)
        key = f"{ds}_{len(names)}_{cpi}"
        expansions = json.loads(
            (ROOT / "results" / f"expansions_{key}.json").read_text())["expansions"]

        f_q = br.encode_texts(model, queries).float()
        rew = [rerank(e, REWEIGHTED_PRIORS) for e in expansions]
        f_e = br.encode_texts(
            model, [build_text(q, c) for q, c in zip(queries, rew)]).float()
        fused = 0.9 * f_q + 0.1 * f_e
        fused /= fused.norm(dim=-1, keepdim=True)

        ra = ranks_of(f_q, img_feats, gt)
        rb = ranks_of(fused, img_feats, gt)
        per_ds[ds] = (ra, rb)

        r1a, r1b = (ra < 1).mean(), (rb < 1).mean()
        print(f"\n===== {ds}: {len(queries)} q / {len(names)} imgs =====")
        print(f"A baseline (RemoteCLIP): R@1 {r1a:.2%}  "
              f"R@5 {(ra < 5).mean():.2%}  R@10 {(ra < 10).mean():.2%}")
        print(f"B query-KG a=0.9:        R@1 {r1b:.2%}  dR@1 {(r1b - r1a) * 100:+.3f}")
        mean_d, (lo, hi), p = paired_bootstrap(ra, rb)
        print(f"bootstrap B vs A: {mean_d * 100:+.3f} "
              f"[{lo * 100:+.3f},{hi * 100:+.3f}] p={p:.4f}")

    # stratified pooled test, identical to E22
    rng = np.random.default_rng(0)
    n_boot = 10_000
    total_q = sum(len(ra) for ra, _ in per_ds.values())
    print(f"\n===== POOLED (stratified, {total_q:,} queries) =====")
    for k in (1, 5, 10):
        deltas = np.zeros(n_boot)
        obs = 0.0
        for ra, rb in per_ds.values():
            ha, hb = (ra < k).astype(float), (rb < k).astype(float)
            n = len(ha)
            w = n / total_q
            obs += (hb.mean() - ha.mean()) * w
            idx = rng.integers(0, n, (n_boot, n))
            deltas += (hb[idx].mean(axis=1) - ha[idx].mean(axis=1)) * w
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        p = (deltas <= 0).mean()
        print(f"POOLED R@{k}: {obs * 100:+.3f} pts, "
              f"95% CI [{lo * 100:+.3f}, {hi * 100:+.3f}], p = {p:.4f}")


if __name__ == "__main__":
    main()

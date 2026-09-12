"""
Embedding-space analysis figures (KTIR Fig.5/Fig.6 analogs), RSICD test set.

fig5_heatmap: text-image cosine similarity for one pair per scene category,
  baseline CLIP vs DCKF query expansion (top-2 concepts, tau=0.55, beta=0.9).
fig6_tsne: t-SNE of caption embeddings for the 10 largest scene categories,
  baseline vs DCKF-fused, with silhouette scores.

Run from repo root: ./venv/bin/python paper/make_fig_embed.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import baseline_retrieval as br                      # noqa: E402
from expand_query import QueryExpander              # noqa: E402

TOP_K, TAU, BETA = 2, 0.55, 0.9
INK, ACCENT, GRAY = "#26313b", "#2c5f8a", "#8a939b"

plt.rcParams.update({
    "font.size": 8, "font.family": "serif",
    "axes.linewidth": 0.6, "text.color": INK, "axes.edgecolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})

br.set_dataset(str(ROOT / "data/rsicd/images_test"),
               str(ROOT / "data/rsicd/captions_test.tsv"), "rsicd")
model, preprocess = br.load_model()
names, img_feats = br.encode_gallery(model, preprocess)
img_feats = img_feats.float()
name_to_idx = {n: i for i, n in enumerate(names)}

# captions grouped by image, category from the original filename prefix
caps_by_img = defaultdict(list)
for line in (ROOT / "data/rsicd/captions_test.tsv").read_text().splitlines():
    if not line.strip():
        continue
    fname, caption = line.split("\t", 1)
    caps_by_img[fname].append(caption)

def category(fname):
    stem = fname.rsplit(".", 1)[0]
    head = stem.rsplit("_", 1)[0]
    return head if head and not head[0].isdigit() else None

by_cat = defaultdict(list)
for fname in names:
    c = category(fname)
    if c and fname in caps_by_img:
        by_cat[c].append(fname)

ex = QueryExpander(model=model, device=br.DEVICE,
                   top_k=TOP_K, sim_threshold=TAU)

def dckf_feats(queries):
    enriched = [ex.expand(q)["enriched"] for q in queries]
    f_base = br.encode_texts(model, queries).float()
    f_exp = br.encode_texts(model, enriched).float()
    f = BETA * f_base + (1 - BETA) * f_exp
    return f_base, f / f.norm(dim=-1, keepdim=True)

# ---------------------------------------------------------------- heatmap --
cats = sorted(by_cat)
print(f"{len(cats)} categories: {cats}")
pair_files = [by_cat[c][0] for c in cats]
pair_caps = [caps_by_img[f][0] for f in pair_files]
f_base, f_dckf = dckf_feats(pair_caps)
I = img_feats[[name_to_idx[f] for f in pair_files]]
S0 = (f_base @ I.T).cpu().numpy()
S1 = (f_dckf @ I.T).cpu().numpy()
print("diag mean base %.4f dckf %.4f | off-diag base %.4f dckf %.4f" % (
    np.diag(S0).mean(), np.diag(S1).mean(),
    S0[~np.eye(len(S0), dtype=bool)].mean(),
    S1[~np.eye(len(S1), dtype=bool)].mean()))

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), dpi=200)
vmin = min(S0.min(), S1.min()); vmax = max(S0.max(), S1.max())
for ax, S, title in ((axes[0], S0, "(a) Baseline CLIP"),
                     (axes[1], S1, "(b) DCKF")):
    im = ax.imshow(S, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=8.5)
    ax.set_xticks(range(len(cats)))
    ax.set_yticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=90, fontsize=5.6)
    ax.set_yticklabels(cats if ax is axes[0] else [], fontsize=5.6)
    ax.set_xlabel("image index (one per category)", fontsize=7.5)
axes[0].set_ylabel("text query index", fontsize=7.5)
fig.colorbar(im, ax=axes, shrink=0.82, pad=0.02).set_label(
    "cosine similarity", fontsize=7.5)
fig.savefig(ROOT / "paper/fig5_heatmap.pdf", bbox_inches="tight")
fig.savefig(ROOT / "paper/fig5_heatmap.png", dpi=300, bbox_inches="tight")
print("wrote fig5_heatmap")

# ------------------------------------------------------------------ t-SNE --
top10 = sorted(by_cat, key=lambda c: -len(by_cat[c]))[:10]
texts, labels = [], []
for ci, c in enumerate(top10):
    for fname in by_cat[c]:
        for cap in caps_by_img[fname]:
            texts.append(cap)
            labels.append(ci)
labels = np.array(labels)
print(f"t-SNE set: {len(texts)} captions over {top10}")
f_base, f_dckf = dckf_feats(texts)

from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), dpi=200)
cmap = plt.get_cmap("tab10")
for ax, F, title in ((axes[0], f_base.cpu().numpy(), "(a) Baseline CLIP"),
                     (axes[1], f_dckf.cpu().numpy(), "(b) DCKF")):
    sil = silhouette_score(F, labels, metric="cosine")
    Z = TSNE(n_components=2, init="pca", perplexity=30, random_state=0,
             metric="cosine").fit_transform(F)
    for ci, c in enumerate(top10):
        m = labels == ci
        ax.scatter(Z[m, 0], Z[m, 1], s=2.5, color=cmap(ci), label=c,
                   linewidths=0, alpha=0.75)
    ax.set_title(f"{title}  (silhouette {sil:.3f})", fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])
    print(title, "silhouette", sil)
axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6.4,
               markerscale=3.2, frameon=False, handletextpad=0.3)
fig.tight_layout(pad=0.5)
fig.savefig(ROOT / "paper/fig6_tsne.pdf", bbox_inches="tight")
fig.savefig(ROOT / "paper/fig6_tsne.png", dpi=300, bbox_inches="tight")
print("wrote fig6_tsne")

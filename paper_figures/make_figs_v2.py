"""New/replacement figures for the DCKF paper (preview build).
 fig_motivation.pdf  : intro figure (a) raw substitution drifts vs (b) DCKF gate + gentle fusion
 fig_regimes.pdf     : replacement for the forest plot (grouped bars, two regimes)
 fig_tau.pdf         : tau-sensitivity curves (Flickr30k dev subset, RSICD), from persisted logs
 fig_headline_delta.pdf : replacement for the baseline-vs-best bar chart (deltas, numbers from Tables II-IV)
All numbers are copied from main.tex tables / results logs; nothing new is computed.
"""
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, FancyArrowPatch

OUT = Path(__file__).parent
INK, ACCENT, GRAY, LIGHT = "#26313b", "#2c5f8a", "#8a939b", "#dde2e7"
PURPLE, GREEN, RED = "#7a5aa6", "#2e8b3a", "#c23b3b"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
    "text.color": INK, "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})

# ------------------------------------------------------------------ motivation
def vec(ax, ang, r, color, lw, label=None, ls="-", dx=0, dy=0, fs=7.5, style="normal"):
    x, y = r*np.cos(np.radians(ang)), r*np.sin(np.radians(ang))
    ax.add_patch(FancyArrowPatch((0,0),(x,y), arrowstyle="-|>", mutation_scale=8,
                 color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0, zorder=4))
    if label: ax.text(x+dx, y+dy, label, color=color, fontsize=fs, ha="center", va="center", style=style, zorder=6)
    return x, y
def img(ax, ang, r, color, label, dx, dy):
    x, y = r*np.cos(np.radians(ang)), r*np.sin(np.radians(ang))
    ax.scatter([x],[y], s=34, marker="s", facecolor="white", edgecolor=color, lw=1.4, zorder=5)
    ax.text(x+dx, y+dy, label, color=color, fontsize=6.6, ha="center", va="center", zorder=6)

fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.7), dpi=200)
for ax, panel in zip(axs, "ab"):
    th = np.radians(np.linspace(-25, 105, 200))
    ax.plot(np.cos(th), np.sin(th), color=LIGHT, lw=1.2, zorder=1)
    ax.set_xlim(-0.30, 1.25); ax.set_ylim(-0.78, 1.40); ax.set_aspect("equal"); ax.axis("off")
    img(ax, 78, 1.0, GREEN, "correct\nimage", -0.02, 0.17)
    img(ax, 4, 1.0, RED, "distractor", -0.02, -0.12)
    vec(ax, 64, 0.86, INK, 1.4, "T (query)", dx=-0.30, dy=-0.02)
    if panel == "a":
        ax.set_title("(a) raw substitution", fontsize=7.8, pad=6)
        vec(ax, 12, 0.92, PURPLE, 1.4, "E (expanded text)", dx=-0.64, dy=-0.24)
        ax.annotate("", xy=(0.92*np.cos(np.radians(15)), 0.92*np.sin(np.radians(15))),
                    xytext=(0.92*np.cos(np.radians(58)), 0.92*np.sin(np.radians(58))),
                    arrowprops=dict(arrowstyle="->", color=RED, lw=0.9, connectionstyle="arc3,rad=-0.35"), zorder=3)
        ax.text(0.70, 0.66, "drift", color=RED, fontsize=7, style="italic")
        ax.text(0.48, -0.66, "query replaced by E\n$-2.28$ R@1 (Flickr30k)", fontsize=6.6, ha="center", color=RED)
    else:
        ax.set_title("(b) DCKF", fontsize=7.8, pad=6)
        # drift gate: keep candidates with cos(T,c) >= tau  (cone around T)
        half = np.degrees(np.arccos(0.55))
        ax.add_patch(Wedge((0,0), 1.0, 64-half, 64+half, facecolor="#eaf1f8", edgecolor="none", zorder=0))
        ax.text(-0.12, 0.55, "gate:\ncos $\\geq\\tau$", fontsize=6.4, color=ACCENT, ha="center", va="center", zorder=6)
        vec(ax, 28, 0.74, PURPLE, 1.1, "E (kept)", dx=0.16, dy=-0.04)
        vec(ax, -20, 0.55, GRAY, 1.0, "rejected", ls="--", dx=0.18, dy=0.0, fs=6.4)
        vec(ax, 60, 0.9, ACCENT, 1.6, "$\\tilde{T}=0.9\\,T+0.1\\,E$", dx=0.33, dy=0.05, fs=6.8)
        ax.text(0.48, -0.66, "gentle fusion, T stays dominant\n$+0.38$ R@1$^{*}$ (Flickr30k)", fontsize=6.6, ha="center", color=GREEN)
fig.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.0, wspace=0.0)
fig.savefig(OUT/"fig_motivation.pdf"); fig.savefig(OUT/"fig_motivation.png", dpi=300)

# ------------------------------------------------------------------ regimes (forest-plot replacement)
ROWS = [  # from paper/make_fig2.py (unchanged numbers)
 ("RSICD\ncaptions\nbroken",   0.24, -0.09, 0.57, False),
 ("RSICD\ncaptions\nrepaired", 0.20,  0.02, 0.38, True),
 ("RS\nqueries\npooled", 0.077, -0.000, 0.155, True),
 ("Flickr30k\nqueries",         0.46,  0.16, 0.76, True),
 ("COCO\nqueries",              0.03, -0.09, 0.15, False),
 ("COCO\ncaptions",    0.11,  None, None, False),
 ("Flickr30k\ncaptions", 0.02, None, None, False)]
fig, ax = plt.subplots(figsize=(3.5, 2.7), dpi=200)
xs = [0,1.1,2.2,3.3, 4.9,6.0,7.1]
ax.axvspan(4.3, 7.75, color="#f2f4f6", zorder=0)
ax.axhline(0, color=INK, lw=0.8, zorder=3)
for x,(lab,d,lo,hi,sig) in zip(xs, ROWS):
    ax.bar(x, d, width=0.68, color=(ACCENT if sig else "white"), edgecolor=(ACCENT if sig else GRAY),
           linewidth=(0 if sig else 1.1), zorder=2)
    if lo is not None:
        ax.errorbar(x, d, yerr=[[d-lo],[hi-d]], fmt="none", ecolor=(INK if sig else GRAY), elinewidth=0.9, capsize=2, zorder=4)
    top = (hi if hi is not None else max(d,0)) + 0.03
    ax.text(x, top, f"{d:+.2f}"+("*" if sig else ""), ha="center", va="bottom", fontsize=6.8, color=(INK if sig else GRAY), zorder=5)
ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in ROWS], fontsize=5.9)
ax.set_ylabel("KG increment ($\\Delta$R@1, points)", fontsize=7.5)
ax.set_xlim(-0.6, 7.75); ax.set_ylim(-0.22, 1.12)
ax.spines[["top","right"]].set_visible(False); ax.tick_params(axis="x", length=0)
ax.text(1.1, 1.11, "weak text channel:\nknowledge helps", ha="center", va="top", fontsize=7, color=ACCENT, fontweight="bold")
ax.text(6.0, 1.11, "strong text channel:\nknowledge adds nothing", ha="center", va="top", fontsize=7, color=GRAY, fontweight="bold")
ax.annotate("", xy=(7.6,-0.185), xytext=(-0.4,-0.185), arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.7))
ax.text(3.6, -0.175, "text channel strength", ha="center", va="bottom", fontsize=6.4, color=GRAY, style="italic")
fig.tight_layout(pad=0.4)
fig.savefig(OUT/"fig_regimes.pdf"); fig.savefig(OUT/"fig_regimes.png", dpi=300)

# ------------------------------------------------------------------ tau sensitivity (results/tau_sweep_*.txt)
tau = [0.50,0.55,0.60,0.65,0.70,0.75]
fl = dict(d=[0.34,0.26,0.32,0.26,0.12,0.18], lo=[0.08,0.00,0.08,0.02,-0.08,0.02], hi=[0.64,0.54,0.58,0.52,0.34,0.36])
rs = dict(d=[0.18,0.18,0.20,0.18,0.13,0.02], lo=[-0.02,-0.02,0.00,-0.02,-0.05,-0.13], hi=[0.40,0.40,0.40,0.38,0.33,0.18])
fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=200)
for ax, data, title in zip(axs, (fl, rs), ("(a) Flickr30k (development subset)", "(b) RSICD")):
    ax.axvspan(0.50, 0.65, color="#f2f4f6", zorder=0)
    ax.fill_between(tau, data["lo"], data["hi"], color=ACCENT, alpha=0.15, lw=0, zorder=1)
    ax.plot(tau, data["d"], color=ACCENT, lw=1.3, marker="o", ms=3, zorder=3)
    ax.axhline(0, color=INK, lw=0.6, zorder=2)
    ax.axvline(0.55, color=RED, lw=0.8, ls="--", zorder=2)
    ax.set_title(title, fontsize=7.2, pad=2)
    ax.set_xticks(tau); ax.set_xticklabels([f"{t:.2f}" for t in tau], fontsize=6.4)
    ax.set_xlabel("drift threshold $\\tau$", fontsize=7.2)
    ax.spines[["top","right"]].set_visible(False)
axs[0].set_ylabel("$\\Delta$R@1 (points)", fontsize=7.2)
axs[0].set_ylim(-0.15, 0.65); axs[1].set_ylim(-0.15, 0.65)
axs[0].text(0.552, 0.62, "$\\tau$ used", color=RED, fontsize=6.2, ha="left", va="top")
axs[1].text(0.575, 0.62, "stable region", color=GRAY, fontsize=6.2, ha="center", va="top", style="italic")
fig.tight_layout(pad=0.4, w_pad=0.8)
fig.savefig(OUT/"_superseded"/"fig_tau_devsubset.pdf")  # superseded by make_fig_tau_v3.py (official splits, tau from 0)

# ------------------------------------------------------------------ headline deltas (Tables II, III, IV, V)
datasets = ["Flickr30k","COCO","RSICD","RSITMD","UCM","NWPU"]
q  = [0.38, 0.04, 0.18, 0.13, 0.19, 0.05]            # query expansion alone (B / Table IV)
qs = ["*", "", "", "", "", ""]            # two-sided p<0.05: RSICD Q is p=0.096
full = [2.24, 2.93, 0.66, None, None, 0.28]          # best configuration where captions help (D / Table V combined)
fs = ["‡", "‡", "*", "", "", "*"]         # RSICD D p=.005, NWPU D (BLIP-large aerial) p=.006
x = np.arange(6); w = 0.36
fig, ax = plt.subplots(figsize=(3.5, 2.2), dpi=200)
ax.axhline(0, color=INK, lw=0.6)
b1 = ax.bar(x - w/2, q, w*0.92, color=GRAY, label="query expansion (KG)", zorder=3)
xf = [xi + w/2 for xi, v in zip(x, full) if v is not None]
vf = [v for v in full if v is not None]
b2 = ax.bar(xf, vf, w*0.92, color=ACCENT, label="+ gallery captions (full DCKF)", zorder=3)
for xi, v, s in zip(x - w/2, q, qs):
    ax.text(xi, v + 0.06, f"+{v:.2f}{s}", ha="center", va="bottom", fontsize=5.8, color=INK, rotation=90)
for xi, v, s in zip(xf, vf, [s for s, v in zip(fs, full) if v is not None]):
    ax.text(xi, v + 0.06, f"+{v:.2f}{s}", ha="center", va="bottom", fontsize=5.8, color=INK, rotation=90)
ax.set_ylabel("$\\Delta$ t2i R@1 vs. frozen CLIP (points)", fontsize=7.2)
ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=7); ax.set_ylim(0, 3.7)
ax.yaxis.grid(True, linestyle="--", linewidth=0.4, color=LIGHT, zorder=0)
ax.spines[["top","right"]].set_visible(False); ax.tick_params(length=0)
ax.legend(fontsize=6.4, frameon=False, loc="upper right", handlelength=1.2)
fig.tight_layout(pad=0.4)
fig.savefig(OUT/"fig_headline_delta.pdf"); fig.savefig(OUT/"fig_headline_delta.png", dpi=300)
print("done")

# ------------------------------------------------------------------ regimes as a 2x2 mechanism map
fig, ax = plt.subplots(figsize=(3.5, 2.55), dpi=200); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 8.0)
cols = [(2.3, "weak text channel", ACCENT), (6.15, "strong text channel", GRAY)]
rows = [(4.55, "query side", "(expanding the\nquery text)"), (1.45, "gallery side", "(captions fused\ninto the index)")]
cells = {  # (row, col): list of (label, delta, sig)
 (0,0): [("Flickr30k queries", "+0.46", True), ("RS queries, pooled", "+0.08", True)],
 (0,1): [("COCO queries", "+0.03", False)],
 (1,0): [("RSICD captions, repaired", "+0.20", True), ("RSICD captions, broken", "+0.24", False)],
 (1,1): [("COCO captions", "+0.11", False), ("Flickr30k captions", "+0.02", False)],
}
cw, ch = 3.75, 2.9
for ci,(cx,cl,ccol) in enumerate(cols):
    ax.text(cx, 6.85, cl, ha="center", va="center", fontsize=7.2, fontweight="bold", color=ccol)
    ax.text(cx, 6.42, ["knowledge helps","knowledge adds nothing"][ci], ha="center", va="center", fontsize=6.4, style="italic", color=ccol)
ax.annotate("", xy=(8.15, 7.45), xytext=(0.35, 7.45), arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.7))
ax.text(4.25, 7.55, "text channel strength", ha="center", va="bottom", fontsize=6.0, color=GRAY, style="italic")
for ri,(cy,rl,rsub) in enumerate(rows):
    ax.text(0.2, cy+0.25, rl, ha="left", va="center", fontsize=7.0, fontweight="bold", color=INK, rotation=90)
    for ci,(cx,cl,ccol) in enumerate(cols):
        x0, y0 = cx-cw/2+0.15, cy-ch/2
        face = "#e6eef6" if ci==0 else "#f2f4f6"
        ax.add_patch(plt.Rectangle((x0, y0), cw-0.3, ch, facecolor=face, edgecolor="white", lw=1.5))
        items = cells[(ri,ci)]
        for k,(lab,d,sig) in enumerate(items):
            yy = cy + (0.55 if len(items)==2 else 0) - k*1.25
            ax.text(x0+0.18, yy+0.28, lab, ha="left", va="center", fontsize=6.3, color=INK)
            ax.text(x0+0.18, yy-0.28, d + ("* significant" if sig else "  not significant"), ha="left", va="center",
                    fontsize=6.3, color=(ACCENT if sig else GRAY), fontweight=("bold" if sig else "normal"))
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig.savefig(OUT/"fig_regimes_matrix.pdf"); fig.savefig(OUT/"fig_regimes_matrix.png", dpi=300)
print("matrix done")

# ------------------------------------------------------------------ Fig 6 alternative: similarity geometry without a 30x30 grid
S0=np.load(OUT.parent/"results/embed_geometry/S0.npy"); S1=np.load(OUT.parent/"results/embed_geometry/S1.npy"); n=len(S0); m=np.eye(n,dtype=bool)
fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.85), dpi=200)
ax=axs[0]
ax.scatter(S0[~m], S1[~m], s=4, color=GRAY, alpha=0.45, lw=0, label="unmatched pairs (870)", zorder=2)
ax.scatter(S0[m], S1[m], s=9, color=ACCENT, lw=0, label="matched pairs (30)", zorder=3)
lo,hi=min(S0.min(),S1.min())-0.005, max(S0.max(),S1.max())+0.005
ax.plot([lo,hi],[lo,hi], color=INK, lw=0.6, ls="--", zorder=1)
ax.set_xlim(lo,hi); ax.set_ylim(lo,hi); ax.set_aspect("equal")
ax.set_xlabel("cosine, baseline CLIP", fontsize=7); ax.set_ylabel("cosine, DCKF", fontsize=7)
ax.set_title("(a) every text-image cell", fontsize=7.2, pad=2)
ax.legend(fontsize=5.6, frameon=False, loc="lower right", handletextpad=0.2, borderaxespad=0.2, markerscale=1.2)
ax.tick_params(labelsize=6); ax.spines[["top","right"]].set_visible(False)
ax=axs[1]
D=S1-S0; bins=np.linspace(-0.008,0.008,33)
ax.hist(D[~m], bins=bins, color=GRAY, alpha=0.55, lw=0, density=True, label="unmatched pairs")
ax.hist(D[m], bins=bins, histtype="step", color=ACCENT, lw=1.3, density=True, label="matched pairs")
ax.axvline(0, color=INK, lw=0.6)
ax.set_xlabel("$\\Delta$ cosine (DCKF $-$ baseline)", fontsize=6.6); ax.set_ylabel("density", fontsize=7)
ax.set_title("(b) how far each cell moved", fontsize=7.2, pad=2)
ax.set_xticks([-0.008,0,0.008]); ax.set_xticklabels(["-0.008","0","+0.008"])
ax.legend(fontsize=5.8, frameon=False, loc="upper right", borderaxespad=0.1, handlelength=1.2)
ax.tick_params(labelsize=6); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(pad=0.4, w_pad=1.0)
fig.savefig(OUT/"fig_geometry.pdf"); fig.savefig(OUT/"fig_geometry.png", dpi=300)
print("diag mean", S0[m].mean().round(4), S1[m].mean().round(4), "offdiag", S0[~m].mean().round(4), S1[~m].mean().round(4),
      "| max |delta| over all cells", np.abs(S1-S0).max().round(4), "| matched-pair rank-1 rows base/dckf:", int((S0.argmax(1)==np.arange(n)).sum()), int((S1.argmax(1)==np.arange(n)).sum()),
      "| mean delta matched/unmatched", (S1-S0)[m].mean().round(4), (S1-S0)[~m].mean().round(4),
      "| median within-row rank of matched pair (1=best) base/dckf", np.median([(S0[i]>S0[i,i]).sum()+1 for i in range(n)]), np.median([(S1[i]>S1[i,i]).sum()+1 for i in range(n)]))

# ------------------------------------------------------------------ Fig 6 option C: 10x10 heatmap, readable
import json
cats=json.load(open(OUT.parent/"results/embed_geometry/cats.json"))
ten=["airport","beach","bridge","denseresidential","industrial","parking","pond","river","storagetanks","viaduct"]
idx=[cats.index(c) for c in ten]; A=S0[np.ix_(idx,idx)]; B=S1[np.ix_(idx,idx)]
short=["airport","beach","bridge","dense resid.","industrial","parking","pond","river","storage tanks","viaduct"]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.35), dpi=200)
vmin=min(A.min(),B.min()); vmax=max(A.max(),B.max())
for ax, S, title in ((axes[0], A, "(a) Baseline CLIP"), (axes[1], B, "(b) DCKF query expansion")):
    im = ax.imshow(S, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=8.5, pad=4)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7.2)
    ax.set_yticklabels(short if ax is axes[0] else [], fontsize=7.2)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{S[i,j]:.2f}"[1:], ha="center", va="center", fontsize=5.6,
                    color=("black" if S[i,j] > (vmin+vmax)/2 else "white"),
                    fontweight=("bold" if i==j else "normal"))
    ax.set_xlabel("image (one per category)", fontsize=7.5)
    ax.tick_params(length=0)
axes[0].set_ylabel("caption query (one per category)", fontsize=7.5)
cb=fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02); cb.set_label("cosine similarity", fontsize=7.5); cb.ax.tick_params(labelsize=6.5)
fig.savefig(OUT/"fig_heatmap10.pdf", bbox_inches="tight"); fig.savefig(OUT/"fig_heatmap10.png", dpi=300, bbox_inches="tight")
m=np.eye(10,dtype=bool)
print("10x10 stats: diag", A[m].mean().round(3), B[m].mean().round(3), "offdiag", A[~m].mean().round(3), B[~m].mean().round(3),
      "rows diag max", int((A.argmax(1)==np.arange(10)).sum()), int((B.argmax(1)==np.arange(10)).sum()), "max|delta|", np.abs(B-A).max().round(4))

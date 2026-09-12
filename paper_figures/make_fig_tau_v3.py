"""Fig. 8 (v3): drift-threshold sweep on the OFFICIAL splits from results/gate_ablation_*.json
(paper pipeline: top-5 pool, typed scoring, fusion b=0.9), tau from 0 (no gate) to 0.65. Bands = 95% bootstrap CI."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent; OUT=Path(__file__).parent
INK, ACCENT, GRAY, RED, LIGHT = "#26313b", "#2c5f8a", "#8a939b", "#c23b3b", "#d9dee3"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6, "text.color": INK,
    "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK})
fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.9), dpi=200, sharey=False)
for ax, key, title in zip(axs, ("flickr30k_k1k_1000_5", "rsicd_1093_5"), ("(a) Flickr30k (official split)", "(b) RSICD")):
    d = json.load(open(ROOT/"results"/"imgboot"/f"gate_ablation_{key}.json"))
    rows = [r for r in d["rows"] if r["pool"]=="top5" and r["scoring"]=="typed" and r["mode"]=="fusion" and "ref" not in r]
    rows = sorted({r["tau"]: r for r in rows}.values(), key=lambda r: r["tau"])
    x=[r["tau"] for r in rows]; y=[r["dR1_vs_baseline"]*100 for r in rows]
    lo=[r["ci"][0]*100 for r in rows]; hi=[r["ci"][1]*100 for r in rows]
    ax.fill_between(x, lo, hi, color=ACCENT, alpha=0.18, lw=0)
    ax.plot(x, y, color=ACCENT, lw=1.4, marker="o", ms=3.2, zorder=3)
    ax.axhline(0, color=INK, lw=0.6); ax.axvline(0.55, color=RED, lw=0.9, ls="--")
    ax.set_title(title, fontsize=7.2); ax.set_xlabel("drift threshold $\\tau$", fontsize=7.2)
    ax.set_xticks([0, 0.3, 0.45, 0.55, 0.65]); ax.set_xticklabels(["0\n(no gate)", ".30", ".45", ".55", ".65"], fontsize=6.2)
    ax.tick_params(axis="y", labelsize=6.5); ax.spines[["top","right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4, color=LIGHT, zorder=0)
axs[0].set_ylabel("$\\Delta$R@1 (points)", fontsize=7.2)
axs[0].set_ylim(-0.4, 0.9); axs[1].set_ylim(-0.4, 0.9)
axs[0].text(0.56, 0.86, "$\\tau$ used", color=RED, fontsize=6.2, ha="left", va="top")
fig.tight_layout(pad=0.7, w_pad=0.9)
fig.savefig(OUT/"fig_tau.pdf"); fig.savefig(OUT/"fig_tau.png", dpi=300); print("fig_tau v3 written")

"""Fig. 4 candidate: RSICD gallery channel as a function of captioner quality (KBS line-plot style,
baseline dashed, cf. Listwise Fig. 8). Numbers: Table V / E15 / E18 / E28 in results/EXPERIMENTS.md."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT=Path(__file__).parent
INK, ACCENT, GRAY, RED = "#26313b", "#2c5f8a", "#8a939b", "#c23b3b"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6, "text.color": INK,
    "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK})
steps=["BLIP-base\n(no prompt)", "BLIP-base\n+ aerial prompt", "BLIP-large\n+ aerial prompt"]
base=5.40
caps=[5.40-0.79, 5.40+0.01, 5.40+0.48]          # -0.79 | +0.01 (n.s.) | +0.48 (two-sided image-cluster p=.118)  [reviewer_rsicd_*.json, alpha=0.9]
capkg=[caps[0]+0.07, caps[1]+0.09, caps[2]+0.18] # query-KG increment D-C: +0.07 | +0.09 | +0.18 (two-sided image-cluster p=.102)
x=[0,1,2]
fig, ax = plt.subplots(figsize=(3.5, 2.3), dpi=200)
ax.axhline(base, color=INK, lw=0.9, ls="--", zorder=1)
ax.text(-0.3, base-0.05, "baseline 5.40", fontsize=6.2, color=INK, ha="left", va="top")
ax.plot(x, caps, color=GRAY, lw=1.3, marker="s", ms=4.5, zorder=3, label="gallery captions only")
ax.plot(x, capkg, color=ACCENT, lw=1.5, marker="o", ms=4.5, zorder=4, label="captions + KG (full DCKF)")
# KG increment annotations
for xi, lo, hi, lab, sig in zip(x, caps, capkg, ["+0.07", "+0.09", "+0.18 (p=.10)"], [False, False, False]):
    if abs(hi-lo) > 0.03:
        ax.annotate("", xy=(xi+0.13, hi), xytext=(xi+0.13, lo), arrowprops=dict(arrowstyle="-|>", color=(ACCENT if sig else GRAY), lw=0.8, mutation_scale=6), zorder=5)
    if xi == 1:
        ax.text(0.88, 5.62, "KG " + lab, fontsize=6.2, color=GRAY, va="bottom", ha="right")
    else:
        ax.text(xi+0.18, (lo+hi)/2, "KG " + lab, fontsize=6.2, color=(ACCENT if sig else GRAY), va="center", ha="left")
# channel effect labels under the grey points
for xi, v, lab, ha, dx in zip(x, caps, ["$-0.79$", "$+0.01$ (n.s.)", "$+0.48$ (p=.12)"], ["center","left","center"], [0, 0.12, 0]):
    ax.text(xi+dx, v-(0.16 if xi==1 else 0.10), lab, fontsize=6.2, color=GRAY, ha=ha, va="top")
ax.text(2, capkg[2]+0.09, "6.06\n($+0.66$, $p{=}0.030$)", fontsize=6.2, color=ACCENT, ha="center", va="bottom")
ax.set_xticks(x); ax.set_xticklabels(steps, fontsize=6.8)
ax.set_xlim(-0.35, 2.55); ax.set_ylim(4.2, 6.6)
ax.set_ylabel("RSICD t2i R@1 (%)", fontsize=7.5)
ax.set_xlabel("gallery captioner quality", fontsize=7.5)
ax.spines[["top","right"]].set_visible(False); ax.tick_params(axis="x", length=0); ax.tick_params(axis="y", labelsize=7)
ax.legend(fontsize=6.3, frameon=False, loc="upper left", borderaxespad=0.2, handlelength=1.6)
fig.tight_layout(pad=0.4)
fig.savefig(OUT/"fig_dose.pdf"); fig.savefig(OUT/"fig_dose.png", dpi=300); print("dose ok")

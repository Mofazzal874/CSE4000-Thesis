"""ICCIT 2026, Fig. 1 -- "the answer" chart.

Panel (a): effect of each lever on recall of people below 8 px, expressed as a
within-protocol delta so the two C2A protocols are never compared with one another.
The three network changes sit in one block, the training change in its own block.
Panel (b): the absolute level that survives on real disaster footage.

Every number is copied from AUDIT_2026-08-21.md:
  official split, very-tiny recall   0.7427 -> 0.7461 -> 0.7575, state-space 0.7567
  scene-disjoint matched pair        0.7299 -> 0.7508, bootstrap 95% CI [1.77, 2.38] pp
  real disaster frames, <8 px        0 of 54 instances recovered, both models
Sized for one IEEE two-column A4 column (3.45 in usable width), 8 pt text.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
BLUE, ORANGE, GREY, RED = "#0072B2", "#E69F00", "#9a9a9a", "#B4453C"
INK, MUTED = "#222222", "#8a8a8a"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.7,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "xtick.major.size": 2, "ytick.major.size": 2,
    "axes.axisbelow": True, "figure.facecolor": "white",
})

# rows are drawn bottom-up: y = 0 is the lowest row
LEVERS = [
    (0, "Supervision\n(label assignment)", +2.08, ORANGE),
    (1, "Sequence modeling\n(state-space neck)", -0.08, GREY),
    (2, "Resolution\n(stride-4 head)", +1.14, BLUE),
    (3, "Attention\n(CBAM)", +0.34, BLUE),
]
CI = (2.08 - 1.77, 2.38 - 2.08)          # bootstrap 95% CI on the supervision delta
XMAX = 4.45

fig = plt.figure(figsize=(3.45, 3.0))
gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1.0], hspace=0.60,
                      left=0.315, right=0.995, top=0.905, bottom=0.125)

ax = fig.add_subplot(gs[0])
for y, name, d, colour in LEVERS:
    ax.barh(y, d, height=0.60, color=colour, edgecolor="none")
    if y == 0:
        ax.errorbar(d, y, xerr=[[CI[0]], [CI[1]]], fmt="none",
                    ecolor=INK, elinewidth=0.7, capsize=2)
    tip = d + CI[1] if y == 0 else d
    ax.text(tip + (0.10 if d >= 0 else -0.10), y, f"{d:+.2f}", va="center",
            ha="left" if d >= 0 else "right", fontsize=7.5, color=INK)
ax.axvline(0, color=INK, lw=0.7)
ax.axhline(0.5, color="#cccccc", lw=0.7, ls=(0, (3, 2)))
ax.set_yticks([y for y, *_ in LEVERS])
ax.set_yticklabels([n for _, n, _, _ in LEVERS], fontsize=7.5, linespacing=1.12)
ax.set_xlim(-0.85, XMAX)
ax.set_xticks([0, 1, 2, 3])
ax.set_xlabel("change in recall of people below 8 px (pp)", fontsize=7.5, labelpad=1.5)
ax.grid(axis="x", color="#dddddd", lw=0.5)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_title("(a) what each lever is worth", fontsize=8, color=INK, pad=4, loc="left")
ax.text(XMAX, 2.0, "network,\nofficial\nsplit", fontsize=6.5, color=MUTED,
        va="center", ha="right", linespacing=1.1)
ax.text(XMAX, 0.0, "training,\nscene-disjoint\nsplit", fontsize=6.5, color=MUTED,
        va="center", ha="right", linespacing=1.1)

ax2 = fig.add_subplot(gs[1])
ax2.bar([0, 1], [0.7508, 0.0], width=0.46, color=[ORANGE, RED], edgecolor="none")
ax2.text(0, 0.7508 + 0.04, "0.751", ha="center", fontsize=7.5, color=INK)
ax2.text(1, 0.055, "0.000", ha="center", fontsize=7.5, color=RED)
ax2.text(1, 0.185, "none of the\n54 instances", ha="center", fontsize=6.5,
         color=MUTED, linespacing=1.1)
ax2.set_xlim(-0.6, 1.6)
ax2.set_ylim(0, 0.94)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(["C2A composites\n(scene-disjoint test)",
                     "real disaster\nfootage"], fontsize=7, linespacing=1.12)
ax2.set_yticks([0, 0.25, 0.50, 0.75])
ax2.set_yticklabels(["0", "0.25", "0.50", "0.75"])
ax2.set_ylabel("recall below 8 px", fontsize=7.5, labelpad=2)
ax2.grid(axis="y", color="#dddddd", lw=0.5)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
ax2.tick_params(axis="x", length=0)
ax2.set_title("(b) where the ladder ends", fontsize=8, color=INK, pad=4, loc="left")

out = HERE / "fig_answer.png"
fig.savefig(out, dpi=600)
print("wrote", out, out.stat().st_size // 1024, "KB")

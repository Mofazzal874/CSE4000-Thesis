"""ICCIT 2026 figure: adaptation stages per real domain.
Sized for a single IEEE two-column A4 column (3.45 in usable width), 8 pt labels
per IEEE figure-text guidance. Numbers copied from the verified eval JSONs
(see MANIFEST): AP50_allpoint per stage and domain.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
INK, MUTED = "#222222", "#888888"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.7,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "xtick.major.size": 2, "ytick.major.size": 2,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.facecolor": "white",
})

domains = ["Drone test\n(n=60)", "Disaster,\ntraining events\n(n=25)",
           "Disaster,\nunseen event\n(n=23)"]
zero_shot = [0.335, 0.133, 0.396]
drone_ft = [0.783, 0.124, None]      # not measured on the unseen event
all_real = [0.795, 0.411, 0.413]

x = [0, 1, 2]
w = 0.26
fig, ax = plt.subplots(figsize=(3.45, 2.55))


def bars(offset, vals, color, label, annotate=False):
    xs = [xi + offset for xi, v in zip(x, vals) if v is not None]
    vs = [v for v in vals if v is not None]
    ax.bar(xs, vs, width=w, color=color, label=label,
           edgecolor="white", linewidth=0.8)
    if annotate:      # label only the final stage; exact values live in Table II
        for xi, v in zip(xs, vs):
            ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                        xytext=(0, 2), ha="center", fontsize=8, color=INK)


bars(-w, zero_shot, BLUE, "C2A-trained (zero-shot)")
bars(0.0, drone_ft, ORANGE, "+ drone fine-tune")
bars(w, all_real, GREEN, "+ all-real fine-tune", annotate=True)
ax.annotate("n.m.", (2, 0.015), ha="center", fontsize=7, color=MUTED)

# not-significant bracket over the unseen-event pair (base vs adapted)
yb = 0.47
ax.plot([2 - w, 2 - w, 2 + w, 2 + w], [yb, yb + 0.022, yb + 0.022, yb],
        color=INK, linewidth=0.7)
ax.annotate("n.s. ($p$=0.47)", (2, yb + 0.04), ha="center", fontsize=7,
            color=INK)

ax.set_xticks(x)
ax.set_xticklabels(domains, fontsize=7)
ax.set_ylabel("AP50", fontsize=8)
ax.set_ylim(0, 0.92)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=1,
          handlelength=1.1, handletextpad=0.4, borderaxespad=0.2,
          labelspacing=0.3)
fig.tight_layout(pad=0.3)
out = HERE / "fig_adapt_stages.png"
fig.savefig(out, dpi=600)
plt.close(fig)
print("wrote", out)

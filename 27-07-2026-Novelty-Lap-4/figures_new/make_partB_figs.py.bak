"""Part-B report figures (laptop, matplotlib). Regenerable: numbers below are copied verbatim
from results\\MANIFEST.md (S2 alpha sweep; d3_all multi-domain evals). DPI 300 for print.
Palette Okabe-Ito subset, validated colorblind-safe (dataviz six-checks, light mode)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
INK, MUTED = "#333333", "#888888"

plt.rcParams.update({
    "font.size": 10.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white",
})

# ---------------------------------------------------------------- fig 1: alpha dose-response
# S2 50-ep pilots, scene-split TEST (MANIFEST 2026-07-25)
alpha = [0.0, 0.5, 1.0]
vt = [0.6931, 0.7120, 0.7293]
ap_s = [0.5596, 0.5526, 0.5053]

fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.plot(alpha, vt, "-o", color=BLUE, linewidth=2, markersize=7, label="very-tiny recall ($<$8 px)")
ax.plot(alpha, ap_s, "-s", color=ORANGE, linewidth=2, markersize=7, label="AP$_{small}$ (COCO)")
for x, y in zip(alpha, vt):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=10, color=INK)
for x, y in zip(alpha, ap_s):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, -14),
                ha="center", fontsize=10, color=INK)
ax.axvline(0.5, color=MUTED, linewidth=0.8, linestyle="--")
ax.annotate("chosen $\\gamma$=0.5", (0.5, 0.755), ha="center", fontsize=10, color=MUTED)
ax.set_xlabel("assignment blend weight $\\gamma$   (0 = CIoU only, 1 = NWD only)")
ax.set_ylabel("metric on C2A scene-split test")
ax.set_xticks(alpha)
ax.set_ylim(0.47, 0.78)
ax.legend(frameon=False, loc="center left", fontsize=10)
fig.tight_layout()
fig.savefig(HERE / "fig_gamma_dose.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- fig 2: sim-to-real grouped bars
# AP50_allpoint per domain x model stage (MANIFEST 2026-07-26 / 08-06)
domains = ["own drone test\n(n=60)", "R-set disaster\nwithin-video (n=25)", "cross-event disaster\nunseen video (n=23)"]
zero_shot = [0.335, 0.133, 0.396]
drone_ft = [0.783, 0.124, None]          # None = not measured on cross-event
all_real = [0.795, 0.411, 0.413]

x = [0, 1, 2]
w = 0.26
fig, ax = plt.subplots(figsize=(6.0, 3.7))

def bars(offset, vals, color, label):
    xs = [xi + offset for xi, v in zip(x, vals) if v is not None]
    vs = [v for v in vals if v is not None]
    b = ax.bar(xs, vs, width=w, color=color, label=label, edgecolor="white", linewidth=1.5)
    for xi, v in zip(xs, vs):
        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=10, color=INK)
    return b

bars(-w, zero_shot, BLUE, "C2A-trained (zero-shot)")
bars(0.0, drone_ft, ORANGE, "+ drone fine-tune")
bars(w, all_real, GREEN, "+ all-real fine-tune (drone+disaster+campus)")
ax.annotate("n.m.", (2, 0.02), textcoords="offset points", xytext=(0, 4),
            ha="center", fontsize=10, color=MUTED)          # drone-ft not measured on cross-event
# n.s. bracket over the cross-event pair (base vs all-real): p=0.47
yb = 0.47
ax.plot([2 - w, 2 - w, 2 + w, 2 + w], [yb, yb + 0.02, yb + 0.02, yb], color=INK, linewidth=0.9)
ax.annotate("n.s. ($p$=0.47)", (2, yb + 0.03), ha="center", fontsize=10, color=INK)

ax.set_xticks(x)
ax.set_xticklabels(domains, fontsize=10)
ax.set_ylabel("AP$_{50}$ (all-point)")
ax.set_ylim(0, 0.88)
ax.legend(frameon=False, fontsize=10, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.tight_layout()
fig.savefig(HERE / "fig_sim2real_bars.png", dpi=300)
plt.close(fig)

print("wrote", HERE / "fig_gamma_dose.png")
print("wrote", HERE / "fig_sim2real_bars.png")

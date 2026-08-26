"""ICCIT 2026, stride-geometry figure, redrawn for one IEEE column.

The same 32 by 32 px patch of a 640 px input is shown at four detection strides,
holding one identical target of 12 px on the square root of area. Cells spanned
is that size divided by the stride: 0.4, 0.75, 1.5 and 3.0. This is the same
construction as the hand-drawn version it replaces, redrawn so the panel labels
are legible at 3.45 in and 7 pt rather than shrunk to illegibility.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).parent
RED, INK, GRID, ADD = "#B4453C", "#222222", "#c9c9c9", "#E69F00"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7,
    "figure.facecolor": "white",
})

PATCH = 32.0            # px of input covered by one panel
TW, TH = 7.6, 19.0      # target box, sqrt(area) = 12.0 px
PANELS = [(32, "P5", "0.4"), (16, "P4", "0.75"), (8, "P3", "1.5"), (4, "P2", "3.0")]

fig, axes = plt.subplots(1, 4, figsize=(3.45, 1.42))
fig.subplots_adjust(left=0.012, right=0.988, top=0.985, bottom=0.30, wspace=0.16)

for ax, (stride, name, cells) in zip(axes, PANELS):
    added = stride == 4
    edge = ADD if added else INK
    ax.add_patch(Rectangle((0, 0), PATCH, PATCH, fill=False, ec=edge,
                           lw=1.3 if added else 1.0))
    n = int(PATCH / stride)
    for k in range(1, n):
        ax.plot([k * stride, k * stride], [0, PATCH], color=GRID, lw=0.5, zorder=0)
        ax.plot([0, PATCH], [k * stride, k * stride], color=GRID, lw=0.5, zorder=0)
    ax.add_patch(Rectangle(((PATCH - TW) / 2, (PATCH - TH) / 2), TW, TH,
                           fill=False, ec=RED, lw=1.1, zorder=3))
    ax.set_xlim(-0.6, PATCH + 0.6)
    ax.set_ylim(-0.6, PATCH + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    tag = f"{name}, stride {stride}" + (" (added)" if added else "")
    ax.text(PATCH / 2, -3.2, tag, ha="center", va="top", fontsize=7,
            color=ADD if added else INK)
    ax.text(PATCH / 2, -9.4, f"cell {stride} px", ha="center", va="top",
            fontsize=6.5, color=INK)
    ax.text(PATCH / 2, -15.2, f"target {cells} cells", ha="center", va="top",
            fontsize=6.5, color=ADD if added else INK)

out = HERE / "fig_stride.png"
fig.savefig(out, dpi=600)
print("wrote", out, out.stat().st_size // 1024, "KB")

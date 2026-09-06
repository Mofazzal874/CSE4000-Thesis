"""Part-B report figures (laptop, matplotlib). Regenerable: numbers below are copied verbatim
from results\\MANIFEST.md (S2 alpha sweep; d3_all multi-domain evals). DPI 300 for print.
Palette Okabe-Ito subset, validated colorblind-safe (dataviz six-checks, light mode).

TYPOGRAPHY
----------
Both figures go into the report through

    \\figorplaceholder{figures/<name>.png}{7.0cm}

which scales the image to fit a 7.0 cm box that is \\textwidth (438.514 pt =
6.0677 in) wide. Drawn at 5.0x3.2 and 6.0x3.7 in, they were being shrunk by
0.86 and 0.74, so a 10.5 pt label landed at about 7.8 pt on the page -- well
under the 11 pt of the caption beside it.

Both are therefore drawn at exactly the size of that box, which makes the
scale factor 1.0: a point set here is a point on the printed page, and the
image fills the width instead of leaving a margin either side. The reserved
box height does not change, so page layout is untouched.

Fonts are the report's own Times New Roman, loaded from its bundled fonts/ so
the figures match the surrounding page rather than matplotlib's DejaVu.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).parent
FONT_DIR = HERE.parent.parent / "Defense" / "draft3_01_09_26" / "fonts"
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
INK, MUTED = "#333333", "#888888"

# Report geometry, measured from the built document via \the\textwidth.
TEXTWIDTH_IN = 438.51411 / 72.27
BOX_H_IN = 7.0 / 2.54       # the 7.0cm \figorplaceholder box both figures use
PT_TARGET = 10.0            # printed size for figure text (captions are 11 pt)


def use_document_font():
    """Load the report's bundled Times New Roman so figures match the page."""
    faces = ["times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"]
    missing = [f for f in faces if not (FONT_DIR / f).exists()]
    if missing:
        raise SystemExit(f"missing bundled font(s) in {FONT_DIR}: {missing}")
    for f in faces:
        font_manager.fontManager.addfont(str(FONT_DIR / f))
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "axes.unicode_minus": False,   # Times has no U+2212; use the hyphen
        "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "figure.facecolor": "white",
    })


# Draw at the exact size of the box LaTeX puts the image in, so nothing is
# rescaled and the font sizes below are true printed points.
FIGSIZE = (TEXTWIDTH_IN, BOX_H_IN)


def pt(figsize, target=PT_TARGET):
    """Return (font size to set here, the resulting scale on the page).

    LaTeX fits the image into a TEXTWIDTH_IN x BOX_H_IN box preserving aspect,
    so the scale is whichever of the two constraints binds first. Drawing at
    FIGSIZE makes it 1.0 and the returned size is the printed size.
    """
    w, h = figsize
    scale = min(TEXTWIDTH_IN / w, BOX_H_IN / h)
    return target / scale, scale


use_document_font()

# ---------------------------------------------------------------- fig 1: alpha dose-response
# S2 50-ep pilots, scene-split TEST (MANIFEST 2026-07-25)
alpha = [0.0, 0.5, 1.0]
vt = [0.6931, 0.7120, 0.7293]
ap_s = [0.5596, 0.5526, 0.5053]

FS1 = FIGSIZE
P1, S1 = pt(FS1)
plt.rcParams.update({"font.size": P1, "axes.labelsize": P1,
                     "xtick.labelsize": P1, "ytick.labelsize": P1})

fig, ax = plt.subplots(figsize=FS1)
ax.plot(alpha, vt, "-o", color=BLUE, linewidth=2, markersize=7, label="very-tiny recall ($<$8 px)")
ax.plot(alpha, ap_s, "-s", color=ORANGE, linewidth=2, markersize=7, label="AP$_{small}$ (COCO)")
for x, y in zip(alpha, vt):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=P1, color=INK)
for x, y in zip(alpha, ap_s):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, -14),
                ha="center", fontsize=P1, color=INK)
ax.axvline(0.5, color=MUTED, linewidth=0.8, linestyle="--")
ax.annotate("chosen $\\gamma$=0.5", (0.5, 0.755), ha="center", fontsize=P1, color=MUTED)
ax.set_xlabel("assignment blend weight $\\gamma$   (0 = CIoU only, 1 = NWD only)")
ax.set_ylabel("metric on C2A scene-split test")
ax.set_xticks(alpha)
ax.set_ylim(0.47, 0.78)
ax.legend(frameon=False, loc="center left", fontsize=P1)
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
FS2 = FIGSIZE
P2, S2 = pt(FS2)
plt.rcParams.update({"font.size": P2, "axes.labelsize": P2,
                     "xtick.labelsize": P2, "ytick.labelsize": P2})

fig, ax = plt.subplots(figsize=FS2)


def bars(offset, vals, color, label, hatch=None):
    # Colour AND pattern, so the three stages stay separable in greyscale and
    # for readers with colour-vision deficiency. matplotlib draws the hatch in
    # the edge colour, so the edge is dark rather than white.
    xs = [xi + offset for xi, v in zip(x, vals) if v is not None]
    vs = [v for v in vals if v is not None]
    b = ax.bar(xs, vs, width=w, color=color, label=label, hatch=hatch,
               edgecolor="black", linewidth=0.6)
    for xi, v in zip(xs, vs):
        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=P2, color=INK)
    return b


bars(-w, zero_shot, BLUE, "C2A-trained (zero-shot)", hatch="/")
bars(0.0, drone_ft, ORANGE, "+ drone fine-tune", hatch=".")
bars(w, all_real, GREEN, "+ all-real fine-tune (drone+disaster+campus)", hatch="x")
ax.annotate("n.m.", (2, 0.02), textcoords="offset points", xytext=(0, 4),
            ha="center", fontsize=P2, color=MUTED)          # drone-ft not measured on cross-event
# n.s. bracket over the cross-event pair (base vs all-real): p=0.47
yb = 0.54          # clears the 0.396 / 0.413 value labels beneath it
ax.plot([2 - w, 2 - w, 2 + w, 2 + w], [yb, yb + 0.02, yb + 0.02, yb], color=INK, linewidth=0.9)
ax.annotate("n.s. ($p$=0.47)", (2, yb + 0.03), ha="center", fontsize=P2, color=INK)

ax.set_xticks(x)
ax.set_xticklabels(domains, fontsize=P2)
ax.set_ylabel("AP$_{50}$ (all-point)")
ax.set_ylim(0, 0.88)
ax.legend(frameon=False, fontsize=P2, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.tight_layout()
fig.savefig(HERE / "fig_sim2real_bars.png", dpi=300)
plt.close(fig)

print("wrote", HERE / "fig_gamma_dose.png")
print(f"   drawn {FS1[0]:.4f}x{FS1[1]:.4f} in, scale {S1:.3f}, text {P1:.1f} pt")
print("wrote", HERE / "fig_sim2real_bars.png")
print(f"   drawn {FS2[0]:.4f}x{FS2[1]:.4f} in, scale {S2:.3f}, text {P2:.1f} pt")

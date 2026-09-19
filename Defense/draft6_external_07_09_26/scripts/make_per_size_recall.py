"""Regenerate Figure 4.8 (per-size recall, all inference configurations).

The original of this figure was produced ad hoc and the script was never kept,
so the PNG in figures/ had no traceable source. This rebuilds it from the
evaluation record instead, which makes the numbers auditable:

    source  Defense/demo/results/metrics/grand_summary.json
    output  Defense/draft3_01_09_26/figures/per_size_recall_all_configs.png

Every bar is read straight from that file; nothing is typed in by hand. The
combined SAHI+TTA configuration is deliberately left out -- it duplicates
SAHI 256 to four decimals and the report does not discuss it.

TYPOGRAPHY -- why the figure is built at an odd size
----------------------------------------------------
The report sets \\textwidth to 438.514 pt = 6.0677 in, and the figure is
included with

    \\figorplaceholder{figures/per_size_recall_all_configs.png}{5.8cm}

which scales the image to width = \\linewidth inside a 5.8 cm box. If the
figure is drawn at any other physical size, LaTeX rescales it and every label
shrinks or grows by that ratio -- which is how a plot ends up with text that
does not match the page. Drawing it at exactly 6.0677 x 2.2835 in makes the
scale factor 1.0, so a point set here is a point on the printed page.

Text is therefore set in real points, in the document's own Times New Roman
(loaded from the bundled fonts/ so this works anywhere the report builds), at
the 11 pt used for captions. Body text is 12 pt.

Run from this folder:  python scripts/make_per_size_recall.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
SUMMARY = REPORT.parent / "demo" / "results" / "metrics" / "grand_summary.json"
OUT = REPORT / "figures" / "per_size_recall_all_configs.png"
FONT_DIR = REPORT / "fonts"

# Page geometry, measured from the built document (\the\textwidth) rather than
# assumed, and the height of the box the figure is placed in.
FIG_W_IN = 438.51411 / 72.27      # \textwidth in inches
FIG_H_IN = 5.8 / 2.54             # the 5.8cm \figorplaceholder box
PT_BODY = 12                      # document body size
PT_FIG = 11                       # caption size; in-figure text matches it
PT_LEGEND = 10                    # legends conventionally sit one point below

# Size bands follow the metric contract: edges in pixels of sqrt(area).
BUCKETS = ["very_tiny", "tiny", "small", "medium"]
XLABELS = ["very-tiny\n(<8 px)", "tiny\n(8-16)", "small\n(16-32)", "medium\n(32-96)"]

# key in grand_summary.json -> (legend label, colour, hatch)
# Colour AND pattern, so each series is identifiable in greyscale printing and
# to a reader with colour-vision deficiency. matplotlib draws a hatch in the
# EDGE colour, so the bars need an edge for the pattern to appear at all.
SERIES = [
    ("baseline_640",       "baseline 640", "#1f77b4", ""),
    ("sahi_slice256_ov30", "SAHI 256",     "#ff7f0e", "/"),
    ("sahi_slice320_ov25", "SAHI 320",     "#2ca02c", "\\"),
    ("sahi_slice512_ov25", "SAHI 512",     "#d62728", "x"),
    ("sahi_slice640_ov30", "SAHI 640",     "#9467bd", "."),
    ("tta_1280_custom",    "TTA 1280",     "#8c564b", "+"),
]


def use_document_font():
    """Load the report's bundled Times New Roman so the figure matches the page."""
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
        "font.size": PT_FIG,
        "axes.labelsize": PT_FIG,
        "xtick.labelsize": PT_FIG,
        "ytick.labelsize": PT_FIG,
        "legend.fontsize": PT_LEGEND,
        "axes.linewidth": 0.8,
    })


def main():
    use_document_font()
    gs = json.loads(SUMMARY.read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=600)
    n = len(SERIES)
    w = 0.86 / n

    for i, (key, label, colour, hatch) in enumerate(SERIES):
        rec = gs[key]["per_size_recall"]
        vals = [rec[b] for b in BUCKETS]
        xs = [xi - 0.43 + w * (i + 0.5) for xi in range(len(BUCKETS))]
        ax.bar(xs, vals, width=w, color=colour, label=label, hatch=hatch,
               edgecolor="black", linewidth=0.5, zorder=3)

    ax.set_xticks(range(len(BUCKETS)))
    ax.set_xticklabels(XLABELS)
    ax.set_xlim(-0.5, len(BUCKETS) - 0.5)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("recall")
    ax.grid(axis="y", color="0.88", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8, pad=2)

    # One row above the plot. Inside the axes a six-column legend is wider than
    # the axes itself and spills past both spines, so it sits outside instead,
    # where it also cannot cover a bar.
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
                    ncol=len(SERIES), frameon=False, handlelength=1.2,
                    handleheight=0.8, columnspacing=1.0, handletextpad=0.4,
                    borderpad=0.2)

    fig.tight_layout(pad=0.3)

    # Guard: a legend wider than the figure is the failure this figure had, and
    # matplotlib reports no error when it happens -- so measure it.
    fig.canvas.draw()
    lb = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    if lb.width > FIG_W_IN + 1e-3:
        raise SystemExit(f"legend is {lb.width:.3f} in wide, figure is only "
                         f"{FIG_W_IN:.3f} in -- reduce ncol or PT_LEGEND")

    fig.savefig(OUT)
    plt.close(fig)
    print(f"  legend {lb.width:.3f} in of {FIG_W_IN:.3f} in available")

    print(f"wrote {OUT}")
    print(f"  drawn at {FIG_W_IN:.4f} x {FIG_H_IN:.4f} in, text at {PT_FIG} pt "
          f"(body is {PT_BODY} pt); scale on the page is 1.0")
    for key, label, _, _ in SERIES:
        rec = gs[key]["per_size_recall"]
        print(f"  {label:14}", " ".join(f"{rec[b]:.4f}" for b in BUCKETS))
    counts = gs["baseline_640"]["per_size_gt_count"]
    print("  GT counts     ", " ".join(f"{counts[b]:,}" for b in BUCKETS))


if __name__ == "__main__":
    main()

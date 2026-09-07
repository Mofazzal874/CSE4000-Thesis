"""Shared typography for the report's matplotlib figures.

Two rules, both of which the figures used to break.

1. Use the document's typeface. matplotlib defaults to DejaVu, which does not
   match the report's Times New Roman. The bundled faces in ../fonts/ are
   registered here so the figures build anywhere the report itself builds.

2. Draw at the size the figure is printed. A figure enters the report as

       \\figorplaceholder{figures/<name>.png}{<height>}

   which scales the image into a box of \\linewidth by <height>, preserving
   aspect. A figure drawn at any other physical size is rescaled, and every
   label is rescaled with it -- so the printed text size is an accident of the
   figsize rather than a choice. `box()` returns the exact size of that box;
   draw at it and the scale is 1.0, making a point set in the script a point
   on the printed page.

   Note that compensating the other way -- inflating the font so that
   scale x size lands on target -- does not work. The canvas keeps its old
   proportions, so the larger text overruns it.

Geometry is measured from the built document (\\the\\textwidth), not assumed
from the paper size and margins.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT_DIR = Path(__file__).resolve().parent.parent / "fonts"

TEXTWIDTH_PT = 438.51411      # \the\textwidth of the built report
PT_PER_IN = 72.27             # TeX points per inch
TEXTWIDTH_IN = TEXTWIDTH_PT / PT_PER_IN

PT_BODY = 12                  # document body size
PT_CAPTION = 11               # caption size
PT_FIG = 10                   # in-figure text: one point under the caption


def box(height_cm, width_frac=1.0):
    """Size in inches of the box a figure is placed in.

    height_cm   the second argument of \\figorplaceholder
    width_frac  the minipage fraction if the figure shares a row (e.g. 0.49)
    """
    return (TEXTWIDTH_IN * width_frac, height_cm / 2.54)


def use_document_font(size=PT_FIG, **extra):
    """Load the report's Times New Roman and set text sizes in true points."""
    faces = ["times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"]
    missing = [f for f in faces if not (FONT_DIR / f).exists()]
    if missing:
        raise SystemExit(f"missing bundled font(s) in {FONT_DIR}: {missing}")
    for f in faces:
        font_manager.fontManager.addfont(str(FONT_DIR / f))
    rc = {
        "font.family": "Times New Roman",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "axes.unicode_minus": False,   # Times has no U+2212; use the hyphen
        "font.size": size,
        "axes.labelsize": size,
        "xtick.labelsize": size,
        "ytick.labelsize": size,
        "legend.fontsize": size,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
    }
    rc.update(extra)
    plt.rcParams.update(rc)


def check_fits(fig, artist, name="legend"):
    """Raise if an artist is wider than the figure.

    Overflow is silent in matplotlib -- the artist is simply clipped or spills
    past the canvas -- so it has to be measured. This is the fault that made
    figure 4.8's legend run past both spines.
    """
    fig.canvas.draw()
    bb = artist.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig_w = fig.get_size_inches()[0]
    if bb.width > fig_w + 1e-3:
        raise SystemExit(f"{name} is {bb.width:.3f} in wide but the figure is "
                         f"only {fig_w:.3f} in -- reduce columns or font size")
    return bb.width, fig_w

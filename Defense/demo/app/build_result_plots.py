"""
build_result_plots.py -- render the Results-tab figures for the defense demo.

Reads the frozen report metrics copied into <demo>/results/metrics/ and writes
four PNGs into <demo>/results/plots/. Run ONCE (offline, no GPU); the Gradio app
then just displays the PNGs, so the running demo needs no plotting stack.

    python app/build_result_plots.py

Data sources (CBAM+P2 model, official C2A test, run 20260707_062217):
    results/metrics/grand_summary.json          per-config P/R/F1/per-size/latency/confusion
    results/metrics/pr_f1_conf_<cfg>.csv         conf-swept precision/recall (PR curves)

Colours are the dataviz reference palette, CVD-validated in light mode
(validate_palette.js): configs use categorical slots 1-4; the FP/FN error chart
uses orange/violet. Every low-contrast fill carries a direct value label (the
palette's "relief" requirement), so identity never rests on hue alone.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

APP_DIR = Path(__file__).resolve().parent
DEMO_ROOT = APP_DIR.parent
METRICS = DEMO_ROOT / "results" / "metrics"
PLOTS = DEMO_ROOT / "results" / "plots"

# ---- palette (validated) --------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
EDGE = (0, 0, 0, 0.12)          # hairline border for fills (relief/definition)

# fixed colour PER CONFIG (entity), used across every chart
CFG = {
    "baseline_640":       {"label": "Baseline 640", "c": "#2a78d6"},   # blue
    "sahi_slice256_ov30": {"label": "SAHI 256",     "c": "#008300"},   # green
    "sahi_slice512_ov25": {"label": "SAHI 512",     "c": "#e87ba4"},   # magenta
    "tta_1280_custom":    {"label": "TTA 1280",     "c": "#eda100"},   # yellow
}
HEADLINE = list(CFG)                       # order shown in charts
FP_C, FN_C = "#eb6834", "#4a3aa7"          # orange / violet (error chart)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})


def _fmt(v):
    return f"{v:.3f}".lstrip("0") if v < 1 else f"{v:.3f}"


def _k(n):
    return f"{n/1000:.1f}k"


def _style(ax, ytitle=None, xtitle=None, title=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0)
    ax.xaxis.grid(False)
    if title:
        ax.set_title(title, color=INK, fontsize=13, fontweight="bold",
                     loc="left", pad=12)
    if ytitle:
        ax.set_ylabel(ytitle, color=INK2, fontsize=10)
    if xtitle:
        ax.set_xlabel(xtitle, color=INK2, fontsize=10)


def load_summary():
    return json.loads((METRICS / "grand_summary.json").read_text())


# ---------------------------------------------------------------- chart 1
def plot_per_size_recall(gs):
    """Grouped bars: per-size recall, one bar per config. THE key figure."""
    buckets = ["very_tiny", "tiny", "small", "medium"]
    counts = gs["baseline_640"]["per_size_gt_count"]
    xlabels = [f"{b.replace('_', '-')}\n(n={counts[b]:,})" for b in buckets]

    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=140)
    n = len(HEADLINE)
    w = 0.80 / n
    x = range(len(buckets))
    for i, cfg in enumerate(HEADLINE):
        rec = gs[cfg]["per_size_recall"]
        vals = [rec[b] for b in buckets]
        xs = [xi - 0.40 + w * (i + 0.5) for xi in x]
        ax.bar(xs, vals, width=w * 0.88, color=CFG[cfg]["c"],
               edgecolor=EDGE, linewidth=0.8, label=CFG[cfg]["label"], zorder=3)
        for xv, v in zip(xs, vals):
            ax.text(xv, v + 0.008, _fmt(v), ha="center", va="bottom",
                    fontsize=6.7, color=INK2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(xlabels, fontsize=9, color=INK2)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    _style(ax, ytitle="Recall @ IoU 0.5",
           title="Recall by object size — smaller = harder")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=4,
              frameon=False, fontsize=9, handlelength=1.1, columnspacing=1.4)
    fig.text(0.5, -0.02,
             "TTA 1280 lifts very-tiny recall most (.758 → .850); medium bucket "
             "n is tiny (317) so its swing is noisy.",
             ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(PLOTS / "fig1_per_size_recall.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- chart 2
def plot_recall_vs_latency(gs):
    """Scatter: overall recall vs latency (cost-of-accuracy frontier)."""
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=140)
    ax.set_xscale("log")

    # context SAHI settings in muted grey
    for cfg, name in [("sahi_slice320_ov25", "SAHI 320"),
                      ("sahi_slice640_ov30", "SAHI 640")]:
        if cfg in gs:
            lat = gs[cfg]["latency_ms"]["mean"]
            r = gs[cfg]["recall"]
            ax.scatter([lat], [r], s=70, color=MUTED, alpha=0.5, zorder=3,
                       edgecolor=SURFACE, linewidth=1.2)
            ax.annotate(name, (lat, r), textcoords="offset points",
                        xytext=(7, -9), fontsize=7.5, color=MUTED)

    for cfg in HEADLINE:
        lat = gs[cfg]["latency_ms"]["mean"]
        r = gs[cfg]["recall"]
        ax.scatter([lat], [r], s=150, color=CFG[cfg]["c"], zorder=4,
                   edgecolor=SURFACE, linewidth=1.5)
        ax.annotate(f"{CFG[cfg]['label']}\n{r:.3f} R · {lat:.0f} ms",
                    (lat, r), textcoords="offset points", xytext=(9, 6),
                    fontsize=8, color=INK, fontweight="bold")

    ax.set_xlim(10, 320)
    ax.set_ylim(0.825, 0.885)
    ax.set_xticks([15, 30, 60, 120, 240])
    ax.get_xaxis().set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    _style(ax, ytitle="Overall recall @ IoU 0.5",
           xtitle="Latency per image (ms, log scale) — RTX 4070, C2A 640 px",
           title="Cost of accuracy — recall vs latency")
    fig.text(0.5, -0.02,
             "Up = better recall, left = faster. TTA 1280 buys the most recall "
             "for the least time; SAHI clusters slower for a smaller gain.",
             ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(PLOTS / "fig2_recall_vs_latency.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- chart 3
def plot_pr_curves():
    """Overlaid precision-recall curves from the conf sweep, direct-labelled."""
    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=140)
    for cfg in HEADLINE:
        f = METRICS / f"pr_f1_conf_{cfg}.csv"
        if not f.is_file():
            continue
        P, R = [], []
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                P.append(float(row["precision"]))
                R.append(float(row["recall"]))
        ax.plot(R, P, color=CFG[cfg]["c"], linewidth=2.2, zorder=3,
                solid_capstyle="round", label=CFG[cfg]["label"])
    ax.set_xlim(0.78, 0.90)
    ax.set_ylim(0.74, 0.93)
    _style(ax, ytitle="Precision", xtitle="Recall",
           title="Precision–recall envelope (confidence swept 0.10 → 0.90)")
    # curves converge top-left; legend goes in the empty upper-right
    ax.legend(loc="upper right", frameon=False, fontsize=9.5, labelcolor=INK2,
              handlelength=1.5, borderaxespad=0.8)
    fig.text(0.5, -0.02,
             "Each curve is one config across confidence thresholds. TTA pushes "
             "recall right; SAHI holds precision; the plain model sits inside both.",
             ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(PLOTS / "fig3_pr_curves.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- chart 4
def plot_error_tradeoff(gs):
    """Grouped bars: false positives vs false negatives per config."""
    fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=140)
    x = range(len(HEADLINE))
    w = 0.36
    fp = [gs[c]["confusion_TP_FP_FN"][1] for c in HEADLINE]
    fn = [gs[c]["confusion_TP_FP_FN"][2] for c in HEADLINE]
    xs_fp = [xi - w / 2 - 0.01 for xi in x]
    xs_fn = [xi + w / 2 + 0.01 for xi in x]
    ax.bar(xs_fp, fp, width=w, color=FP_C, edgecolor=EDGE, linewidth=0.8,
           label="False positives (false alarms)", zorder=3)
    ax.bar(xs_fn, fn, width=w, color=FN_C, edgecolor=EDGE, linewidth=0.8,
           label="False negatives (missed people)", zorder=3)
    for xv, v in zip(xs_fp, fp):
        ax.text(xv, v + 200, _k(v), ha="center", va="bottom", fontsize=8,
                color=INK2)
    for xv, v in zip(xs_fn, fn):
        ax.text(xv, v + 200, _k(v), ha="center", va="bottom", fontsize=8,
                color=INK2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([CFG[c]["label"] for c in HEADLINE], fontsize=9.5,
                       color=INK2)
    ax.set_ylim(0, max(fp + fn) * 1.18)
    ax.set_yticks([0, 5000, 10000, 15000, 20000])
    ax.get_yaxis().set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    _style(ax, ytitle="Count on C2A test (72,523 GT boxes)",
           title="Error trade-off — false alarms vs missed people")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.text(0.5, -0.02,
             "TTA 1280 cuts misses (FN 12.0k → 9.0k) but nearly doubles false "
             "alarms (FP 10.1k → 18.5k) — the precision it trades for recall.",
             ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(PLOTS / "fig4_error_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    if not (METRICS / "grand_summary.json").is_file():
        raise SystemExit(f"missing {METRICS/'grand_summary.json'} -- copy the "
                         "report metrics into results/metrics/ first")
    gs = load_summary()
    plot_per_size_recall(gs)
    plot_recall_vs_latency(gs)
    plot_pr_curves()
    plot_error_tradeoff(gs)
    print("wrote:")
    for p in sorted(PLOTS.glob("fig*.png")):
        print("  ", p.relative_to(DEMO_ROOT), f"({p.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()

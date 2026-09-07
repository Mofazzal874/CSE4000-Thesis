"""Regenerate the five report figures that had no saved generator.

`Defense/REPORT_OVERHAUL_PLAN.md` lines 207-209 record that these were
"regenerated from CSVs" during the font overhaul, but the script was not kept,
so the PNGs had no traceable source. Each one is rebuilt here from the archived
run record, so every point in every plot can be checked against the run that
produced it.

    Figure 4.7  fig_training_worst.png       <- Mamba run, ultra/results.csv
    Figure 4.13 cbam_p2_pr_curve.png         <- CBAM+P2 run, metrics/pr_curve.csv
                cbam_p2_f1_conf.png          <- CBAM+P2 run, metrics/f1_vs_conf.csv
                cbam_p2_calibration.png      <- CBAM+P2 run, metrics/calibration.csv
                cbam_p2_confusion.png        <- CBAM+P2 run, metrics/confusion.csv

The CBAM+P2 run is identified by its test AP50 of 0.85326, which is the 0.8533
of the ablation table; the script asserts this rather than trusting the path.

The four diagnostics each sit in a 0.49\\linewidth minipage with a 6.6 cm box,
and the training figure spans the full width in a 5.3 cm box, so each is drawn
at exactly those dimensions -- see report_style for why that matters.

Run from the report folder:  python scripts/make_diagnostic_figs.py
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_style import box, check_fits, use_document_font  # noqa: E402

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
FIGS = REPORT / "figures"
ROOT = REPORT.parent.parent          # the thesis folder

CBAM_P2 = (ROOT / "Last Month" / "24_01_26- Benchmarking YOLOs" / "CBAM_P2Head"
           / "runs" / "20260602_063759_yolo11m_cbam_p2head_s0_nogit")
MAMBA = (ROOT / "Last Month" / "02-06-26-Mamba_CBAM_P2Head" / "runs"
         / "20260609_205717_mamba_cbam_p2head_s0_nogit")

BLUE, ORANGE, GREEN, RED = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"
GREY = "#888888"

# The two box geometries these figures are placed in.
PANEL = box(6.6, 0.49)      # the 2x2 diagnostic grid
WIDE = box(5.3, 1.0)        # the full-width training figure


def rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def verify_run():
    """Confirm CBAM_P2 is the run behind the ablation table's 0.8533."""
    ap50 = json.loads((CBAM_P2 / "metrics" / "coco_test.json").read_text())["AP50"]
    if round(ap50, 4) != 0.8533:
        raise SystemExit(f"wrong run: test AP50 is {ap50:.5f}, expected 0.8533")
    return ap50


def grid(ax):
    ax.grid(color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8, pad=2)


# ------------------------------------------------------------------ (a) PR
def pr_curve():
    d = rows(CBAM_P2 / "metrics" / "pr_curve.csv")
    # Above conf 0.95 the model returns nothing, so the file records
    # precision = recall = 0. Precision is undefined there, not zero; keeping
    # those rows draws a false vertical line down the recall = 0 axis.
    d = [r for r in d if int(r["TP"]) > 0]
    rec = [float(r["recall"]) for r in d]
    prec = [float(r["precision"]) for r in d]

    fig, ax = plt.subplots(figsize=PANEL, dpi=400)
    ax.plot(rec, prec, color=BLUE, linewidth=1.6)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    grid(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGS / "cbam_p2_pr_curve.png")
    plt.close(fig)
    return f"{len(d)} conf steps, recall {max(rec):.4f} at precision {prec[0]:.4f}"


# ------------------------------------------------------------- (b) F1/F2
def f1_conf():
    d = rows(CBAM_P2 / "metrics" / "f1_vs_conf.csv")
    conf = [float(r["conf"]) for r in d]
    f1 = [float(r["F1"]) for r in d]
    f2 = [float(r["F2"]) for r in d]
    # The dashed line is the F2-optimal threshold, taken from the data rather
    # than typed in; it is the 0.16 of the operating-threshold table.
    best = max(d, key=lambda r: float(r["F2"]))
    thr, best_f2 = float(best["conf"]), float(best["F2"])

    fig, ax = plt.subplots(figsize=PANEL, dpi=400)
    ax.plot(conf, f1, color=BLUE, linewidth=1.6, label="$F_1$")
    ax.plot(conf, f2, color=ORANGE, linewidth=1.6, label="$F_2$")
    ax.axvline(thr, color=GREY, linewidth=0.9, linestyle="--")
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("score")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.02, 0.92)
    grid(ax)
    leg = ax.legend(loc="upper right", frameon=True, framealpha=0.9,
                    handlelength=1.4, borderpad=0.35, labelspacing=0.25)
    fig.tight_layout(pad=0.3)
    check_fits(fig, leg, "F1/F2 legend")
    fig.savefig(FIGS / "cbam_p2_f1_conf.png")
    plt.close(fig)
    return f"F2 optimum {best_f2:.4f} at conf {thr:.2f} (dashed line)"


# --------------------------------------------------------- (c) calibration
def calibration():
    d = rows(CBAM_P2 / "metrics" / "calibration.csv")
    # Empty bins carry no meaningful mean confidence, so drop them.
    d = [r for r in d if int(r["count"]) > 0]
    mc = [float(r["mean_conf"]) for r in d]
    ma = [float(r["mean_acc"]) for r in d]

    fig, ax = plt.subplots(figsize=PANEL, dpi=400)
    ax.plot([0, 1], [0, 1], color=GREY, linewidth=0.9, linestyle="--",
            label="perfect")
    ax.plot(mc, ma, "-o", color=BLUE, linewidth=1.6, markersize=4, label="model")
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("empirical precision")
    ax.set_xlim(-0.03, 1.0)
    ax.set_ylim(-0.03, 1.05)
    grid(ax)
    leg = ax.legend(loc="upper left", frameon=True, framealpha=0.9,
                    handlelength=1.4, borderpad=0.35, labelspacing=0.25)
    fig.tight_layout(pad=0.3)
    check_fits(fig, leg, "calibration legend")
    fig.savefig(FIGS / "cbam_p2_calibration.png")
    plt.close(fig)
    return f"{len(d)} populated bins of 10"


# ------------------------------------------------------------ (d) confusion
def confusion():
    d = rows(CBAM_P2 / "metrics" / "confusion.csv")
    m = [[int(d[0]["pred_person"]), int(d[0]["pred_bg"])],
         [int(d[1]["pred_person"]), int(d[1]["pred_bg"])]]

    # constrained layout, not tight_layout: a colorbar takes space from the
    # axes, and tight_layout then under-measures the left margin and clips the
    # y-axis label.
    fig, ax = plt.subplots(figsize=PANEL, dpi=400, layout="constrained")
    # aspect="auto" so the cells fill the panel; imshow's default square cells
    # would leave a dead band above the matrix and make this panel read smaller
    # than the three beside it.
    im = ax.imshow(m, cmap="Blues", vmin=0, aspect="auto")
    peak = max(max(r) for r in m)
    for i in range(2):
        for j in range(2):
            # White on the dark cell, ink on the light ones.
            ax.text(j, i, f"{m[i][j]:,}", ha="center", va="center",
                    color="white" if m[i][j] > 0.6 * peak else "black")
    # Short tick labels with the axis carrying "predicted"/"actual": the long
    # form ("actual background") is wider than the panel and gets clipped.
    ax.set_xticks([0, 1], ["person", "background"])
    ax.set_yticks([0, 1], ["person", "background"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.tick_params(length=0, pad=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("count")
    cb.outline.set_linewidth(0.8)
    fig.savefig(FIGS / "cbam_p2_confusion.png")
    plt.close(fig)
    return f"TP {m[0][0]:,}  FN {m[0][1]:,}  FP {m[1][0]:,}"


# ------------------------------------------- (e) state-space training curves
def training_worst():
    d = rows(MAMBA / "ultra" / "results.csv")
    ep = [int(float(r["epoch"])) for r in d]
    # 154 logged rows but only 150 distinct epochs: the crash-and-resume
    # repeated epochs 2-4. Plotted in log order, which is what makes the
    # small backtrack visible near the start -- that is the real record.
    m50 = [float(r["metrics/mAP50(B)"]) for r in d]
    m5095 = [float(r["metrics/mAP50-95(B)"]) for r in d]
    best_f2_ep = json.loads(
        (MAMBA / "metrics" / "summary.json").read_text())["f2_early_stop"]["best_epoch"]

    fig, (axl, axr) = plt.subplots(1, 2, figsize=WIDE, dpi=400)

    for col, colour, label in (("train/box_loss", BLUE, "box (train)"),
                               ("val/box_loss", ORANGE, "box (val)"),
                               ("train/cls_loss", GREEN, "cls (train)"),
                               ("val/cls_loss", RED, "cls (val)")):
        axl.plot(ep, [float(r[col]) for r in d], color=colour, linewidth=1.2,
                 label=label)
    axl.set_xlabel("epoch")
    axl.set_ylabel("loss")
    grid(axl)
    ll = axl.legend(loc="upper right", frameon=True, framealpha=0.9,
                    handlelength=1.4, borderpad=0.35, labelspacing=0.2)

    axr.plot(ep, m50, color=BLUE, linewidth=1.2, label="mAP50")
    axr.plot(ep, m5095, color=ORANGE, linewidth=1.2, label="mAP50-95")
    axr.axvline(best_f2_ep, color=GREY, linewidth=0.9, linestyle="--")
    axr.annotate(f"best $F_2$\n(epoch {best_f2_ep})", (best_f2_ep, 0.42),
                 xytext=(-4, 0), textcoords="offset points", ha="right",
                 va="center", color=GREY)
    axr.set_xlabel("epoch")
    axr.set_ylabel("validation mAP")
    axr.set_ylim(-0.03, 0.95)
    grid(axr)
    rl = axr.legend(loc="lower right", frameon=True, framealpha=0.9,
                    handlelength=1.4, borderpad=0.35, labelspacing=0.2)

    fig.tight_layout(pad=0.3)
    check_fits(fig, ll, "loss legend")
    check_fits(fig, rl, "mAP legend")
    fig.savefig(FIGS / "fig_training_worst.png")
    plt.close(fig)
    return (f"{len(d)} rows over {len(set(ep))} distinct epochs "
            f"(1-{max(ep)}), best F2 epoch {best_f2_ep}")


def main():
    ap50 = verify_run()
    use_document_font()
    print(f"CBAM+P2 run verified: test AP50 = {ap50:.5f}")
    print(f"panel box {PANEL[0]:.4f} x {PANEL[1]:.4f} in, "
          f"wide box {WIDE[0]:.4f} x {WIDE[1]:.4f} in, scale 1.0\n")
    for fn in (pr_curve, f1_conf, calibration, confusion, training_worst):
        print(f"  {fn.__name__:16} {fn()}")


if __name__ == "__main__":
    main()

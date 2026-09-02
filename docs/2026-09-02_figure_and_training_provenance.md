# Provenance hunt: Figure 4.8 and the training scripts (2026-09-02)

Purpose: make Figure 4.8 of the report reproducible, and record where the script that
produced each trained configuration actually lives. Written because the figure had no
traceable generator, which is the kind of gap a reviewer can legitimately object to.

## 1. Figure 4.8 — `per_size_recall_all_configs.png`

### What the file is

| property | value |
|---|---|
| path | `Defense/draft3_01_09_26/figures/per_size_recall_all_configs.png` |
| former md5 | `917a7bbbb35e…` (3677x1361, 125 KB) |
| appears in | draft1, draft2, the draft2 backup, draft3, `Defense/external` — byte-identical |
| body page | 41 (PDF page 53) |

### The generator was never saved

A repo-wide search for the exact filename returns only consumers, never a producer:
the four chapter `.tex` copies, their `main.log`s, and `Defense/REPORT_OVERHAUL_PLAN.md`.
No `.py`, `.ipynb`, or shell script anywhere in the tree writes this name.

`REPORT_OVERHAUL_PLAN.md` lines 207-209 explain why:

> D: regenerated >=10pt-font figures from CSVs: cbam_p2_pr_curve, cbam_p2_f1_conf,
> cbam_p2_calibration, cbam_p2_confusion (NOW WITH COLORBAR - rule 7),
> per_size_recall_all_configs (clean legend names), fig_training_worst (mamba).

So the figure was regenerated ad hoc during the font-size overhaul and the script was
discarded. The PNG was the only surviving artefact.

### The data behind it, recovered

Source of record: `Defense/demo/results/metrics/grand_summary.json`.

Every one of the 24 bar heights in the original PNG matches this file to four decimals,
which is what confirms the identification:

| configuration | very-tiny (<8 px) | tiny (8-16) | small (16-32) | medium (32-96) |
|---|---|---|---|---|
| `baseline_640` | 0.7576 | 0.8644 | 0.8844 | 0.7981 |
| `sahi_slice256_ov30` | 0.7884 | 0.8340 | 0.9109 | 0.7950 |
| `sahi_slice320_ov25` | 0.7823 | 0.8342 | 0.9096 | 0.7950 |
| `sahi_slice512_ov25` | 0.7629 | 0.8369 | 0.9068 | 0.7950 |
| `sahi_slice640_ov30` | 0.7561 | 0.8365 | 0.9033 | 0.7855 |
| `tta_1280_custom` | 0.8496 | 0.8849 | 0.8975 | 0.6972 |

Ground-truth counts per band: 25,072 / 20,520 / 26,614 / 317 (large band = 0 instances,
which is why the figure has four groups and not five).

The JSON holds a seventh configuration, `sahi_tta_256`, that the figure deliberately
omits. It duplicates `sahi_slice256_ov30` to four decimals in all four bands, and the
combined SAHI+TTA path was dropped from the report, so leaving it out is correct.

### Not to be confused with

`31-03-26(Mamba-ViT-CNN)/SAHI+TTA/plots/per_size_recall_sahi_tta_noaug.png` is a
different render (md5 `a878346c4625…`, 3200x1400) produced by
`31-03-26(Mamba-ViT-CNN)/sahi_tta_eval_noaug.py:1206`. Similar chart, different file.

### The fix

`Defense/draft3_01_09_26/scripts/make_per_size_recall.py` now rebuilds the figure from
`grand_summary.json`. Nothing in it is typed in by hand, so the plot is auditable against
the evaluation record. It also adds a hatch to each series, so the six bars stay separable
in greyscale printing and for readers with colour-vision deficiency.

Closest prior art for the plotting code, kept as the template:
`Defense/demo/app/build_result_plots.py:99` (`plot_per_size_recall`), which reads the same
JSON keys (`per_size_recall`, `per_size_gt_count`).

## 1a. Why the figure text did not match the page

Two separate faults, both of which apply to every matplotlib figure in the report.

**Wrong typeface.** matplotlib defaults to DejaVu, so the plots were set in a different
face from the Times New Roman of the body text. Fixed by loading the report's own bundled
faces from `Defense/draft3_01_09_26/fonts/` through `font_manager.addfont`, which keeps
the figures reproducible anywhere the report itself builds.

**Wrong size, for a non-obvious reason.** A figure is included with

```
\figorplaceholder{figures/<name>.png}{<height>}
```

which fits the image into a box `\textwidth` wide (438.514 pt = 6.0677 in, measured with
`\the\textwidth`, not assumed) and `<height>` tall. If the figure is drawn at any other
physical size, LaTeX rescales it and every label scales with it. The effect was large:

| figure | drawn | scale applied | text set | text printed |
|---|---|---|---|---|
| Fig 4.8 per-size recall | 11.49 x 4.25 in | 0.528 | 19 pt | 10.0 pt |
| Fig 4.12 sim-to-real bars | 6.0 x 3.7 in | 0.745 | 10.5 pt | 7.8 pt |
| dose-response | 5.0 x 3.2 in | 0.861 | 10.5 pt | 9.0 pt |

Three figures, three different printed sizes, none of them chosen.

The fix is to draw each figure at exactly the size of the box it lands in, which makes the
scale 1.0 so a point set in the script is a point on the page. Figure text is then set at
10 pt against 11 pt captions and 12 pt body.

Note the trap: compensating instead by inflating the font (setting 13.4 pt so that
0.745 x 13.4 = 10) does **not** work. The figure keeps its old proportions, so the larger
text overruns it -- the legend ran off the canvas and the value labels collided. Only
resizing the canvas is correct.

Changing a figure's aspect ratio this way does not disturb page layout, because
`\figorplaceholder` reserves a fixed-height box regardless of what it contains. It only
changes how much of the box width the image uses.

## 1b. The other five figures, also recovered

`REPORT_OVERHAUL_PLAN.md` lines 207-209 lists these in the same ad hoc regeneration batch
as Figure 4.8, and none had a saved generator either. All five sources are now located and
the figures are rebuilt by `Defense/draft3_01_09_26/scripts/make_diagnostic_figs.py`.

| figure | source of record |
|---|---|
| `cbam_p2_pr_curve.png` | CBAM+P2 run, `metrics/pr_curve.csv` |
| `cbam_p2_f1_conf.png` | CBAM+P2 run, `metrics/f1_vs_conf.csv` |
| `cbam_p2_calibration.png` | CBAM+P2 run, `metrics/calibration.csv` |
| `cbam_p2_confusion.png` | CBAM+P2 run, `metrics/confusion.csv` |
| `fig_training_worst.png` | Mamba run, `ultra/results.csv` |

CBAM+P2 run =
`Last Month/24_01_26- Benchmarking YOLOs/CBAM_P2Head/runs/20260602_063759_yolo11m_cbam_p2head_s0_nogit`.
The script asserts its test AP50 is 0.85326 -- the 0.8533 of the ablation table -- rather
than trusting the path.

Values that cross-check against the report:

- confusion matrix: TP 60,518 / FN 12,005 / FP 10,135, identical to the previous figure
- F2 optimum 0.8442 at confidence 0.16, matching the operating-threshold table's
  0.16 and 0.844
- best-F2 epoch 149, read from the Mamba run's `summary.json`, not typed in

Two faults in the originals were fixed while rebuilding:

- The PR curve included the six rows above confidence 0.95 where the model returns nothing
  and the file records precision = recall = 0. Precision is undefined there, not zero, and
  plotting them drew a false vertical line down the recall = 0 axis. Those rows are dropped
  (95 of 101 remain).
- The confusion matrix's long tick labels were wider than the panel and were being clipped
  ("actual background" rendered as "ackground"). Short tick labels now carry axis labels
  "predicted" and "actual".

## 2. A number in the report that the run record does not support

Found while identifying what the training figure's marker line should be. Chapter 4 says,
at two places (`04_implementation_results.tex` lines 224 and 411), that the state-space run
continued to **epoch 154**. No run reached an epoch 154.

What the archived record shows:

| run | configured `epochs` | last epoch | best F2 | best fitness | what ended it |
|---|---|---|---|---|---|
| baseline | 300 | 300 | 270 | 260 | ran out the cap |
| CBAM | 300 | 300 | 239 | 250 | ran out the cap |
| CBAM+P2 | 300 | 263 | 218 | 213 | patience: 213 + 50 = 263 |
| Mamba | **150** | **150** | 149 | 149 | ran out its cap |

Three specific problems:

1. **There is no epoch 154.** The Mamba run's `ultra/results.csv` holds 154 *rows* but only
   150 distinct epochs -- epochs 2, 3 and 4 each repeat after the crash-and-resume the text
   describes. The 154 is a row count that was read as an epoch number.
2. **Patience did not end that run.** `args.yaml` sets `patience: 50`, and the best fitness
   is at epoch 149, so patience would have allowed training to epoch 199. The run stopped
   because it hit its own `epochs: 150`.
3. **That budget was half the others'.** The state-space variant was configured for 150
   epochs against 300 for the other three, and its best score falls at epoch 149 -- at the
   end of its allowance, not early within it. "Peaked earliest" reads as evidence the model
   saturated; the record shows it ran out of budget.

The substantive conclusion still holds: the late curve is genuinely flat. Validation mAP50
goes 0.86413 at epoch 140, 0.86450 at 149, 0.86466 at 150 -- a gain of 0.00053 over the
final ten epochs. So "bought no later improvement" is supported; only the epoch number and
the stopping mechanism are wrong, and the halved budget should be stated rather than left
for a reviewer to find in `args.yaml`.

One further item to check, not asserted here: line 220 says "the $F_2$ stop ended the CBAM
run after its best score at epoch 239 and the CBAM+P2 run after epoch 218." Both runs record
`f2_early_stop.stopped = true`, but CBAM ran to its full 300-epoch cap and CBAM+P2 stopped
at 263, which is exactly its fitness-patience bound (213 + 50). The flag appears to record
that the criterion was met rather than that training halted, so the attribution may need the
same correction.

## 2. Training scripts — archived per run, not centrally

Each run directory carries a frozen copy of the exact script that produced it under
`code/`. That copy, not any working file, is the reproducible source.

| configuration | archived script |
|---|---|
| YOLO11m baseline | `Last Month/24_01_26- Benchmarking YOLOs/Yolo11m/runs/20260615_230315_yolo11m_baseline_s0_nogit/code/yolo11m_thesis.py` |
| + CBAM | `Last Month/24_01_26- Benchmarking YOLOs/CBAM/runs/20260601_232929_yolo11m_cbam_s0_nogit/code/yolo11m_cbam_thesis.py` |
| + CBAM + P2 | `Last Month/24_01_26- Benchmarking YOLOs/CBAM_P2Head/runs/20260602_063759_*/code/yolo11m_cbam_p2head_thesis.py` |
| + Mamba + CBAM + P2 | `Last Month/02-06-26-Mamba_CBAM_P2Head/runs/20260609_205717_*/code/mamba_cbam_p2head_thesis.py` |
| scene-split retrain | `05-07-2026-Novelty-Lap/results/pc1/2026-07-24_G1_scenesplit_cbam_p2/*/code/yolo11m_cbam_p2head_thesis.py` |

Per-run measured data used by the report's size-band figures and tables lives in
`metrics/per_size.csv` inside each run folder, with columns
`bin,lo_px,hi_px,gt_total,matched,recall`.

### One trap

`Last Month/24_01_26- Benchmarking YOLOs/Yolo11m/runs_musgd_old/` holds three earlier
baseline runs from the MuSGD optimiser that diverged on the P2 configuration. Those are
superseded and must not be used to recreate anything. The valid baseline is
`runs/20260615_230315_yolo11m_baseline_s0_nogit`.

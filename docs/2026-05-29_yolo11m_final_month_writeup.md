# 2026-05-29 — `yolov11m_final_month.py` write-up

## What this file is
A spec-compliant replacement for the Kaggle notebook
`24_01_26- Benchmarking YOLOs/Yolov11m/yolov11m.ipynb`. It re-trains and
evaluates **YOLO11m baseline** on C2A under the full final-month metric
protocol in `Last Month/system_spec.md`. This run is **Phase A row 1**
(`yolo11m_baseline`, P0) of the headline ablation matrix in Section 9.6 /
Section 25 — it is the reference baseline that CBAM / P2Head / Mamba
variants are compared against.

File location:
`Last Month/24_01_26- Benchmarking YOLOs/Yolov11m/yolov11m_final_month.py`

## Notebook ↔ spec conflicts, and how they were resolved

| Conflict | Notebook | Spec | Resolution |
|---|---|---|---|
| Output path | `/kaggle/working/...` hardcoded | Sec 5.2 — derive from `SCRIPT_DIR` | `OUTPUT_ROOT = Path(__file__).parent` |
| Dataset path | `/kaggle/input/c2a-dataset/...` hardcoded | Sec 5.3 — env-var + candidate probe | `find_dataset_root("C2A_ROOT", "train/images", C2A_CANDIDATES)` |
| Data fraction | `SAMPLE_RATIO=0.5` | Sec 23 — frozen full split with md5 | Full dataset, md5-frozen via `common/splits/splits.md5`; subsampling only inside smoke (`SMOKE_FRACTION=0.01`) |
| Epochs / patience | `EPOCHS=25, PATIENCE=5` | Sec 18 — `NUM_EPOCHS=300` upper, `PATIENCE=30`, custom F2 patience, cos LR | `NUM_EPOCHS=300, PATIENCE=50, F2_PATIENCE=40, cos_lr=True, lrf=0.01` (raised from spec — see "Patience values" section below) |
| Batch / workers / AMP | `batch=10, workers=8`, no AMP | Sec 3/7 — batch 16–24 for YOLO11m, AMP on, workers=4 on Windows | `BATCH_SIZE=16, NUM_WORKERS=4, amp=True`; OOM retry ladder `[16, 8, 4, 4]` |
| Device | `device=0` with T4×2 comments | Sec 3 — single 4070 Ti SUPER | `cuda:0` only, sanity-checked at startup |
| Seeds | Single unseeded run | Sec 12.1 — 5 seeds | `SEEDS=[0,1,2,3,4]`. Cross-seed mean/std/min/max/median + BCa CI written to `runs/yolo11m_baseline_multi_seed_rollup/cross_seed_metrics.csv` after all seeds complete. |
| Metrics | Ultralytics CSV + a few scalars | Sec 11 — full catalog | Implemented (see below) |
| Provenance | None | Sec 14, 11.7 | `PLOTS_INDEX.md`, `PLOTS_INDEX.csv`, `env.json`, `manifest.json`, weights sha256 |
| Smoke | None | Sec 17 | 8-step smoke harness, <5 min, auto-cleanup; 24-h freshness marker enforced by pre-flight |
| Resume / OOM | None | Sec 20 | OOM retry ladder with `nbs` (Ultralytics 8.4.x), last-safety checkpoint copy every epoch, resume detection |
| Image format | `.png .jpg .jpeg` | Sec 5.1 — `.png` only | `.png` only |

## What the script outputs (per run)

Under `Yolov11m/runs/<run_id>/` where `run_id = <YYYYMMDD>_<HHMMSS>_yolo11m_baseline_s<seed>_<short_hash>`:

- `code/` — snapshot of this script at launch
- `env.json` — Python/torch/CUDA/ultralytics versions, git commit, dataset/data-yaml/weights md5, hyperparameters
- `hyperparams.yaml` — flat dump of every knob
- `weights/best.pt`, `weights/last.pt`, `weights/last_safety.pt`
- `results.csv` — Ultralytics per-epoch (box/cls/dfl loss, P/R/mAP, lr/pg*)
- `metrics/`
  - `summary.json`, `summary.xlsx` — headline numbers in one place
  - `per_image_test.csv`, `per_image_val.csv` — TP/FP/FN/P/R/F1/F2/avg_conf/inference_ms per image (paired-bootstrap input)
  - `pr_curve.csv`, `f1_vs_conf.csv`, `confidence_hist.csv`
  - `per_size.csv` — very-tiny / tiny / small / medium / large recall bins
  - `confusion.csv`, `calibration.csv` — ECE/MCE/Brier
  - `coco_test.json`, `coco_val.json` — AP_small/medium/large + AR_1/10/100 + AR_small/med/large (if pycocotools present)
  - `latency_by_res.csv`, `per_image_latency.csv`, `throughput.csv`
  - `epoch_metrics.csv` — per-epoch wall-clock, samples/sec, VRAM peak, grad-norm proxy
- `plots/` — every CSV has a matching PNG (Sec 13.1); `PLOTS_INDEX.md` + `PLOTS_INDEX.csv` provenance
- `architecture/` — `model_summary.txt` (torchinfo), `module_table.csv`, `flops_breakdown.csv`, exported ONNX
- `significance/` — paired-bootstrap JSON when `BASELINE_PER_IMAGE_CSV` is set (default off — per-image scores still saved)
- `energy/emissions.csv` — CodeCarbon (offline tracker, Bangladesh grid)
- `logs/` — `train.log`, `train.err`, `nvidia_smi_loop.csv`, `psutil_loop.csv`, `skipped_metrics.txt`
- `MODEL_CARD.md`, `manifest.json`

## Things that explicitly CANNOT be computed for this row (logged as `[SKIPPED]`)

Vanilla YOLO11m has no CBAM / P2 / SSM, so per Sec 11.6 the following are
N/A and recorded explicitly (NOT silently dropped):

- `attention_map_examples`, `channel_attention_weights`, `spatial_attention_entropy` — no CBAM
- `per-stride_AP`, `tiny_obj_recall_by_head` — no P2 head
- `ssm_state_norm`, `forward_vs_backward_scan_disagreement`, `window_size_per_layer`, `dilation_branch_contribution`, `injection_layer_indices` — no SSM
- SAHI-specific metrics, TTA-specific metrics — handled by Phase C scripts, not this one
- Paired-bootstrap p-values vs another model — `BASELINE_PER_IMAGE_CSV=None` by default; per-image scores are still saved so Phase D can run them later

## What still needs YOUR attention before publishing

1. **Set `C2A_ROOT`** to the dataset root on the AnyDesk PC if the
   candidates in `C2A_CANDIDATES` don't include the current path.
   PowerShell: `$env:C2A_ROOT = "E:\Thesis_mofazzal_2007074\common\c2a\new_dataset3"`.
2. After running the other Phase A rows, point `BASELINE_PER_IMAGE_CSV`
   at this run's `metrics/per_image_test.csv` so CBAM / P2 / Mamba runs
   can compute paired-bootstrap deltas against `yolo11m_baseline`.

## Patience values — 2026-05-29 decision

The user flagged a concern that `PATIENCE=30 / F2_PATIENCE=20` (the spec's
defaults) might be too strict. Literature check before changing:

| Source | Value |
|---|---|
| Ultralytics current default (`cfg/default.yaml`) | **100** (was 50, raised in 8.1) |
| HIT-UAV thermal human detection paper (Sci Reports 2024) — closest to our setup | **50** patience, 300-epoch cap |
| SOD-YOLO (VisDrone, arXiv 2507.12727) | Fixed 200, **no** early stopping |
| AI-TOD benchmark, most VisDrone papers | Fixed schedule (12 / 24 / 40 / 300 epochs), no early stopping |

`30 / 20` is below all published values. Small-object mAP and F2 are noisy
in the last ~30 % of cosine annealing — a single bad late-stage epoch can
trip a tight stopper while the model is still genuinely improving.

**Decision:** `PATIENCE=50, F2_PATIENCE=40`. The 50 matches the closest
published paper and the historic Ultralytics default. The custom F2
stopper is kept (spec Sec 18.1 mandates two simultaneous criteria) but
raised in lockstep, sitting just under the primary so it can still trip
first if F2 specifically stalls while fitness drifts.

## First-smoke findings on the AnyDesk PC (2026-05-29 evening)

Smoke ran cleanly: dataset auto-found at `E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3` (6129 / 2043 / 2043, all three COCO JSONs present), 4070 Ti SUPER detected, AMP ran fine, peak VRAM at batch=16 was 7.87 GB (well below the 14.5 GB headroom target in Sec 19.3).

Two warts discovered and fixed:

1. **Class name auto-detected as `'0'`**. The C2A `train_annotations.json` has category `{"id": 0, "name": "0"}` — a numeric placeholder, not the real class name. The original `detect_class_names()` accepted any non-empty string, so it propagated `'0'` into the data YAML and the MODEL_CARD. Fixed by adding `_is_placeholder_name()` — digit-only / empty / `"none"` / `"null"` names are now rejected and the script falls back to the spec default `['person']` (Sec 5.1) with a warning.

2. **Module-level setup re-ran ~13× during training.** Windows uses `spawn` for DataLoader workers (no `fork()`), so every worker re-imports the entry script and re-runs top-level code: `[paths]` / `[sanity]` / `[splits]` prints, `system_sanity()`, `check_free_space()`, `freeze_or_verify_splits()`, the git probe. Functionally harmless (idempotent), but huge log noise + ~1-2 s worker spawn overhead each. Fixed by gating verbose calls on `_IS_MAIN = multiprocessing.current_process().name == "MainProcess"`. Constants (SCRIPT_DIR, OUTPUT_ROOT, DATASET_ROOT, CLASS_NAMES, SPLIT_MD5, SANITY) still compute in every worker because the dataset/eval functions need them — but silently.

3. **User had `SMOKE_TEST=True` on the AnyDesk PC copy** (sensible — first sanity check before a 30-50 h run). To start the full 5-seed training, set `SMOKE_TEST = False`. The smoke-passed marker (`smoke/_PASSED_yolo11m_baseline.txt`) is already written, so `smoke_recent()` returns True and the full run won't re-trigger smoke.

## Cross-seed aggregation (new in this run)

After all 5 seeds complete, `main()` writes:

- `runs/yolo11m_baseline_multi_seed_rollup/per_seed_summaries.json`
- `runs/yolo11m_baseline_multi_seed_rollup/cross_seed_metrics.csv` — per
  metric: `n_seeds, mean, std, min, max, median, bca_ci_lo, bca_ci_hi`
  for val/test P/R, mAP50, mAP50-95, AP_small/med/large, AR_1/10/100 +
  AR_small/med/large.
- `runs/yolo11m_baseline_multi_seed_rollup/cross_seed_efficiency.csv` —
  params, GFLOPs, weights size, latency p50/p95/p99 mean ± std.

This is the table you cite in the paper's headline-baseline row.

## How to run

```powershell
# from the script's folder
cd "Last Month\24_01_26- Benchmarking YOLOs\Yolov11m"

# Option A: smoke only (Sec 17)
python yolov11m_final_month.py   # with SMOKE_TEST=True in the config block

# Option B: full run (will auto-run a fresh smoke first if marker is >24 h old)
python yolov11m_final_month.py
```

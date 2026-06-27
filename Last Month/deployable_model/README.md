# Deployable model — Joint C2A + SARD training (`joint_c2a_sard_train.py`)

Trains ONE CBAM+P2 model that is strong on **both** disaster scenes (C2A) **and** real
search-and-rescue humans (SARD) — the model a drone could actually use. Starts from your
C2A-trained CBAM+P2, trains on C2A+SARD together, then evaluates the single model on BOTH
test sets with the full `system_spec_thesis.md` §6 metric suite.

---

## 0. Datasets — you ALREADY have both. No new download.
This script uses the two datasets already on your remote PC. It auto-detects them.
| Dataset | Where it must be (E:) | Source link (if you ever need it) |
|---|---|---|
| **C2A** | `E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3\` (train/val/test) | Kaggle `rgbnihal/c2a-dataset` · GitHub `Ragib-Amin-Nihal/C2A` |
| **SARD** | `E:\Thesis_mofazzal_2007074\common\sard\search-and-rescue\` (train/valid/test) | the Roboflow "search-and-rescue" export you already downloaded |
- Nothing to download for this script. If auto-detect ever misses them:
  `$env:C2A_ROOT="...new_dataset3"` and `$env:SARD_ROOT="...search-and-rescue"`.
- (The NITC *disaster* dataset discussed separately is for a FUTURE cross-dataset test — NOT
  needed here.)

## 1. What the script does (internally, step by step)
1. Auto-finds the C2A root, the SARD root, and your newest completed **C2A CBAM+P2 `best.pt`**.
2. Collapses SARD labels to a single `person` class (no-op if already class 0).
3. Builds an explicit **joint image list**: all C2A-train + SARD-train×`SARD_OVERSAMPLE`
   (default 1 — balanced). Same for val.
4. **Inits from the C2A `best.pt`** (CBAM is YAML-native → survives Ultralytics' rebuild; no
   patch needed) and **trains on the joint set** (AdamW lr0=0.001, cosine, 100 ep + patience 50).
5. Evaluates the final `best.pt` **separately on C2A-test and SARD-test** with the §6 suite.
6. Writes weights + metrics + plots + env.json + manifest.json.

## 2. Run it (on E:)
```powershell
E:\Thesis_mofazzal_2007074\mofazzal1\Scripts\Activate.ps1
cd "E:\Thesis_mofazzal_2007074\deployable_model"          # copy this folder D:->E: first
python joint_c2a_sard_train.py        # SMOKE_TEST=True by default -> 2-epoch dry run
# if the SMOKE prints a "DEPLOYABLE MODEL RESULT" block with no error:
#   open joint_c2a_sard_train.py, set  SMOKE_TEST = False , save, then:
python joint_c2a_sard_train.py        # the real run (~3-6 h, early-stops when converged)
```
**Failproof / power cuts:** if it dies mid-run, just **re-run the same command** — it auto-detects
the incomplete run and **resumes from `last.pt`** (with corrupt-checkpoint recovery → falls back to
the newest healthy `epoch*.pt`). It does NOT restart from scratch. `save_period=25`.

## 3. Config knobs (top of the file)
- `SARD_OVERSAMPLE = 1` — SARD train (4041) vs C2A (6129) is already ~40:60 balanced. **Only bump
  to 2 if SARD-test underperforms** while C2A-test stays high (oversampling risks SARD overfit).
- `INIT_FROM_C2A_BEST = True` — start from your C2A model (recommended). `False` = fresh from yolo11m.pt.
- `NUM_EPOCHS = 100`, `PATIENCE = 50`, `BATCH_SIZE = 8` (P2 head is VRAM-heavy; OOM ladder 8→4→2).

## 4. Metrics produced — full `system_spec_thesis.md` §6 set, on BOTH test sets
Per test set (C2A **and** SARD), written under `runs_joint/<id>/metrics/` + `plots/`:
- **Detection quality:** precision, recall, **F1, F2**, mAP@0.5, mAP@0.5:0.95, mAP@0.75,
  COCO **AP_small/medium/large**, AR, **per-size recall** (very-tiny…large), **optimal-threshold F1/F2**.
- **Curves (CSV+PNG):** PR curve, F1-vs-confidence, confidence histogram, per-size-recall bar.
- **Calibration:** ECE, MCE, Brier (CSV).
- **Confusion counts:** TP/FP/FN at conf=0.25.
- **Efficiency (once, dataset-invariant):** params, GFLOPs, layers, weights size, latency
  mean/p50/p95, **FPS@640**.
- **Reproducibility:** `env.json` (versions, GPU, hyperparams, dataset roots), `manifest.json`.
- Thresholds match the chain exactly: conf=0.001/iou=0.7 (AP), conf=0.25/iou=0.5 (F1/F2).

(Deliberately omitted: CBAM attention-map / per-stride-AP architecture plots — those characterize
the *model* and are already captured on the C2A chain runs; they don't change between datasets.)

## 5. Outputs
```
runs_joint/<timestamp>_cbam_p2head_joint_c2a_sard/
  weights/best.pt              <- THE deployable model
  metrics/deployable_summary.json   <- C2A-test + SARD-test + efficiency (read this)
  metrics/per_image_{c2a,sard}.csv, per_size_*, pr_curve_*, f1_vs_conf_*, calibration_*, ...
  plots/pr_curve_*.png, f1_vs_conf_*.png, per_size_recall_*.png
  env.json, manifest.json
```

## 6. Success criterion
C2A-test stays ~0.85 mAP50 **and** SARD-test jumps far above the ~0.006 zero-shot (it trains on
the full SARD train, so expect ≥ your few-shot N=200 point). One model, strong on both = your
deployable artifact. Send me `deployable_summary.json` and I'll confirm.

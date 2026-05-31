# System Specification — THESIS scope (final month)

> **Read this first when writing any code for the thesis report.** This is the
> *trimmed, thesis-month* contract. The exhaustive journal-grade protocol lives in
> `system_spec.md` — this file says which parts of it apply NOW and which are
> deferred to the paper. Hardware facts (CPU/GPU/RAM/OS) are in `system_spec.md`
> §1–§8 and are NOT repeated here; they still bind.
>
> Strategic rationale for everything below: `pivot point_31_5_26.md`.

**Owner:** Mofazzal (2007074) · **Scope:** thesis report, ~1 month · **Last updated:** 2026-05-31
**Machine:** single NVIDIA RTX 4070 Ti SUPER (16 GB, Ada, cap 8.9), i7-14700K, 128 GB RAM, Windows 11 + PowerShell.

---

## 1. Mission for the month

Produce a **complete, honest, defensible** thesis on **tiny-human detection in aerial disaster/SAR imagery**, with:
1. A clean **additive ablation** (4 models, full epochs, this machine).
2. **Cross-dataset validation on a real SAR dataset** (the key upgrade over the current single, semi-synthetic dataset).
3. A consistent metric set + figures, captured once so nothing needs re-running while writing.

**Writing is the critical path.** Compute serves the report, not the other way around. Do not block writing on training.

---

## 2. The final / headline model

**`Mamba + CBAM + P2Head`** is the thesis headline model.
- Best mAP50 / mAP50-95 / precision of all variants; same params as CBAM+P2 (Mamba injected post-init → "gain at zero parameter cost").
- Reported **honestly**: P2 head is the dominant driver; CBAM adds operational gains; the SSM neck improves localization/precision with a small very-tiny-recall trade-off. See `pivot point_31_5_26.md` §3–§4.
- Do **not** reduce the thesis to CBAM+P2 (that abandons novelty → plug-and-play, which is explicitly unwanted). The SSM angle stays; its limits are stated plainly.
- The genuine-novelty redesign (SSM that *drives* small-object gains) is a **paper-phase** task, out of scope this month.

---

## 3. The ablation chain (what to train, in this order)

Each module is added so its contribution is isolated independently (CBAM-alone, P2-alone, then combined, then SSM). Train **one after another** (never two trainings concurrently on one GPU). The presentation order follows the historical run order verified 2026-05-31 from `01-02-2026- ablation study/` (CBAM was run before P2: CBAM 2026-02-02..06, P2Head 2026-02-14..16) and the progress-PDF generation order (attention → detection head).

| Order | model_tag | Script | What it adds | Isolates |
|---|---|---|---|---|
| 1 | `yolo11m_baseline` | `Yolo11m/yolo11m_thesis.py` | — (stock YOLO11m, C2PSA) | reference (**DONE**) |
| 2 | `yolo11m_cbam` | `CBAM/yolo11m_cbam_thesis.py` | CBAM replaces backbone C2PSA | **CBAM contribution (alone)** |
| 3 | `yolo11m_p2head` | `P2Head/…` (to build) | P2 detection head (4-scale, stride-4) | **P2 contribution (alone; expected main driver)** |
| 4 | `yolo11m_cbam_p2head` | `CBAM_P2Head/…` (to build) | CBAM + P2 together | **combined CBAM+P2** |
| 5 | `mamba_cbam_p2head` | `Mamba/…` (to build) | C3k2→C3K2Mamba in neck, + CBAM + P2 | **SSM contribution** (headline model) |

Notes:
- This is a **5-model ablation** (updated 2026-05-31 — the earlier 4-model version omitted standalone `yolo11m_cbam`; the user reinstated it so CBAM-alone and P2-alone are each isolated against the baseline). `yolo11m_eca` stays optional — re-add only if time permits; cite the earlier ECA run otherwise.
- Steps 2–5 reuse the existing implementation patterns: CBAM build = extract YAML → replace `C2PSA` with `CBAM [16,7]` → register module → `YOLO(yaml).load(yolo11m.pt)` (lazy-init CBAM, verbatim from `6-2-26/results/cbam_module.py`); **post-init Mamba injection** into neck C3k2 layers (`idx≥11, c_out≤512`); the custom callbacks (NaNStop, F2EarlyStop, GradientMonitor, checkpoint manager). Clone the reference structure; change only the architecture-building step.
- Each model gets its own sibling folder under `Benchmarking YOLOs/` (`Yolo11m/`, `CBAM/`, `P2Head/`, `CBAM_P2Head/`, `Mamba/`) so the dataset auto-probe (`SCRIPT_DIR.parent.parent/common/c2a/…`) keeps working.
- Single-class `person`. If the C2A annotation category name is the placeholder `"0"`, fall back to `['person']` (known data wart).

### 3.1 Seed policy (thesis)
- Steps 1–4 (`baseline`, `cbam`, `p2head`, `cbam_p2head`): **1 seed** (seed=0).
- Step 5 (`mamba_cbam_p2head`, headline): **3 seeds** (0,1,2) → report mean±std for the headline row only.
- Each `*_thesis.py` script is hard-set to `SEEDS=[0]` (headline → `[0,1,2]`). 5 seeds × all models is infeasible this month (see `pivot` §2). Full 5-seed + significance lives in the `*_paper.py` variants, deferred to the paper.

### 3.2 Training config (this machine)
- `imgsz=640`, `batch=16` (OOM ladder `[16,8,4,4]`; smoke showed ~7.9 GB peak at 16 — headroom exists, may try 24), `amp=True`, `workers=4` (Windows `spawn`), `cache='ram'`.
- `NUM_EPOCHS=300` upper bound, `PATIENCE=50`, `F2_PATIENCE=40`, `cos_lr=True`, `lrf=0.01`, AdamW `lr0=0.001`, `close_mosaic=10`. Default augmentation only (no copy_paste/mixup/flipud — that was a negative result).
- `device='cuda:0'` only. **No DDP** (single GPU). Sanity-check the device is the 4070 Ti SUPER at startup.

---

## 4. Cross-dataset generalization (the key thesis upgrade)

After the chain trains, validate on **real** SAR imagery (C2A is semi-synthetic — this is what makes the generalization claim credible).

- **Primary: SARD** (Search And Rescue Dataset). Real UAV RGB, ~1,981 frames, single `person` (collapse pose sub-labels). Get from Kaggle `nikolasgegenava/sard-search-and-rescue` or Roboflow (YOLO format).
- **Secondary (if time): VisDrone-person** — map `pedestrian`+`people`→`person`, ignore other classes.
- **Skip: HIT-UAV** (thermal; modality gap too large) — future-work mention only.

**Protocol (inference only — no retraining for the headline finding):**
1. **Zero-shot transfer:** each C2A-trained `best.pt` → evaluate on SARD test with the *same* eval pipeline + metrics (§6). Headline generalization result.
2. Report the **C2A→SARD metric drop** per model; smallest drop = best generalizer (may differ from best-C2A model — report either way).
3. **Optional adaptability:** fine-tune the headline model on a small SARD train split, re-evaluate (shows the architecture adapts to real data).
4. **Document class-mapping explicitly** (SARD pose→`person`; VisDrone person-like→`person`).

---

## 5. Evaluation protocol — keep comparisons fair

- **Primary comparison table = no-TTA, no-SAHI**, apples-to-apples, all models same pipeline/thresholds. This is the table that goes head-to-head with Nihal et al.
- **TTA and SAHI are separate, clearly-labeled inference-time ablations** — never mixed into the primary mAP column. Footnote that SAHI uses a per-image matching protocol where standard mAP isn't computable.
- Fixed thresholds: `conf=0.001, iou=0.7` for AP-style metrics; `conf=0.25, iou=0.5` for operational F1/F2.
- **Re-measure all latency on the 4070 Ti SUPER** (old numbers were T4×2 — don't mix hardware in efficiency claims).
- Same frozen test split across all models (md5 in `common/splits/`); per-image scores saved so the paper can run significance later.

---

## 6. Metrics — thesis MUST-have set

Capture all of these for every model (val + test). Most come from **evaluating saved weights**, not retraining. If a metric is N/A for a variant, write `[SKIPPED] <metric> — <reason>` to `skipped_metrics.txt` (never silently drop).

**Detection quality:** precision, recall, F1, **F2** (primary — SAR favors recall), mAP@0.5, mAP@0.5:0.95, **AP_small / AP_medium / AP_large** (COCO), AR@1/10/100, per-size recall (very-tiny <8², tiny 8–16², small 16–32², medium 32–96², large ≥96² — keep for continuity with prior slides), optimal-threshold F1/F2.

**Curves/diagnostics (data + PNG each):** PR curve, F1-vs-confidence, confidence histogram, confusion matrix, calibration/reliability diagram (ECE, MCE, Brier).

**Efficiency (on the 4070):** params (total/trainable), GFLOPs, layers, weights size, latency mean/p50/p95, FPS at 640 (one number is enough for the thesis).

**Training dynamics (per epoch):** box/cls/dfl loss (train+val), total loss, P/R/mAP, lr, epoch time, VRAM peak; which early-stop criterion fired and when.

**Architecture-specific (they tell the story — cheap, keep):**
- CBAM: attention-map overlays on a few test images.
- P2: per-stride AP (which head catches the <8px detections).
- Mamba: forward-vs-reverse scan disagreement, injection-layer indices, SSM state-norm sanity.

**Qualitative (high value for the report):** detection grid, **failure-case grid** (16 worst-FN images) + a small failure taxonomy (occlusion/scale/lighting/crowd), success grid.

**Reproducibility (once per run):** `env.json` (versions, git commit, dataset/yaml/weights md5, seed, hyperparameters), `manifest.json`.

### 6.1 Deferred to the paper (do NOT spend month-time on)
5-seed-everything; paired bootstrap / McNemar / Holm-Bonferroni significance; CodeCarbon energy/CO₂; model cards & dataset datasheets; ONNX export; multi-resolution FPS sweep + multi-batch throughput. (All specified in `system_spec.md` §11–§12 for later.)

---

## 7. Visualization & provenance

Every metric that lands in a CSV also gets a PNG (DPI≥300 for paper figures). Consistent per-model colors across all plots. Maintain `plots/PLOTS_INDEX.md` (one line: `plot ← data_file (columns) — producing_script`) so any figure can be regenerated while writing. Cross-model figures: PR overlay, metric bars, params-vs-mAP Pareto, C2A→SARD generalization-drop bars, improvement waterfall (baseline→+P2→+CBAM→+Mamba).

---

## 8. Folder structure

Use `system_spec.md` §15 layout under `Last Month/`: `common/` (dataset, weights, yamls, frozen `splits/`), `runs/<run_id>/` (code snapshot, env.json, weights, metrics/, plots/, architecture/, logs/), `ablation_master/` (cross-run tables + figures + `paper_tables/`). Add `common/sard/` for the cross-dataset data. Run-ID format: `<YYYYMMDD>_<HHMMSS>_<model_tag>_s<seed>_<short_hash>` (`system_spec.md` §16).

---

## 9. Infra rules that stay ON (cheap insurance — load-shedding country, single GPU)

From `system_spec.md` §17–§21, keep all of these:
- **Smoke test before every full run** (GPU, ~1% data, 2 epochs, all metric/plot paths, <5 min, auto-cleanup on pass, abort full run on fail; 24-h freshness marker).
- **OOM auto-retry ladder** `[16,8,4,4]` with grad-accum compensation.
- **Power-failure resilience:** `save_period=5` + last-safety checkpoint copy every epoch + resume-from-`last.pt` on restart.
- **Dynamic epochs + dual early stop** (fitness patience 50, F2 patience 40).
- **GPU utilization target ≥85%** — run the `nvidia-smi -l 2` sampler; if util is low, apply the §19.2 fixes (workers, ram-cache, batch). Don't let the GPU sit at 0%.
- **Package check at startup** (`ensure_packages()`); never silently skip a metric because a package is missing — log it.

---

## 10. Pre-run checklist (thesis-trimmed)

Before launching any full run:
- [ ] GPU sanity = 4070 Ti SUPER, 16 GB, cap 8.9.
- [ ] Packages verified.
- [ ] Seed set (per §3.1); `env.json` written.
- [ ] Split md5 matches `common/splits/`.
- [ ] Smoke passed in last 24 h; smoke artifacts cleaned.
- [ ] OOM ladder + checkpoint cadence + resource sampler on.
- [ ] All §6 MUST-have metric writers initialized; `PLOTS_INDEX.md` started.
- [ ] Disk free ≥ 20 GB.
- [ ] Confirm GPU util ≥85% after epoch 1; fix if not.

---

## 11. Out of scope this month (explicitly)

New architecture/module design; backbone replacement; 5-seed significance; energy/carbon; model cards/datasheets; additional datasets beyond SARD (+ optional VisDrone); ONNX/deployment. All of these are **paper-phase** — see `pivot point_31_5_26.md` §4, §7, §8.

---

## 12. Scripts & how to run

### 12.1 Two-file convention (thesis vs paper) — applies to EVERY model

For each model in the §3 chain there are two sibling scripts that share **identical metric/eval code** (the full `system_spec.md` §11 catalog). They differ ONLY in `SEEDS` and doc references:

| Script | Scope | SEEDS | Use |
|---|---|---|---|
| `<model>_thesis.py` | thesis report (this month) | `[0]` chain / `[0,1,2]` headline | run NOW |
| `<model>_paper.py` | journal paper (later) | `[0,1,2,3,4]` | run post-course |

Per-model scripts (D: edit paths; on the AnyDesk PC the parent is `E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\` and filenames drop the `v`):
- **Built:** `Yolov11m/yolov11m_thesis.py` + `yolov11m_paper.py` (baseline) — `SEEDS=[0]` / `[0,1,2,3,4]`. **DONE.**
- **Built:** `CBAM/yolo11m_cbam_thesis.py` (standalone CBAM, step 2) — `SEEDS=[0]`; CBAM §11.6 metrics ACTIVE.
- **To build:** `P2Head/…`, `CBAM_P2Head/…`, `Mamba/…` — same `_thesis.py` / `_paper.py` split.

**Headline (`mamba_cbam_p2head_thesis.py`) uses `SEEDS=[0,1,2]`** (3 seeds); all other thesis chain scripts use `[0]`. A `*_paper.py` variant per model is the only place 5 seeds live.

### 12.2 Shared-output behavior (do not "fix" this — it's intended)

`*_thesis.py` and `*_paper.py` for the same model share `MODEL_TAG`, the `runs/` output dir, the smoke marker, and the `<MODEL_TAG>_multi_seed_rollup/` dir, with `SKIP_COMPLETED_SEEDS=True`. So running the thesis script first (seed 0) and the paper script later will **skip seed 0** and only train seeds 1–4 — no recompute — and the shared rollup aggregates all five once present.

### 12.3 Run mechanics (per script)

- **Run from the script's own folder.** Output lands in `<script_dir>/runs/<run_id>/`; the dataset is auto-probed (`C2A_ROOT` env var overrides).
- **Smoke runs automatically** if no passing smoke marker exists from the last 24 h (`system_spec.md` §17.4). It then proceeds to the full run. No separate step needed.
- One training at a time on the single GPU. Launch the next chain model only after the current one finishes.
- Power-loss safe: re-running the same command resumes from `weights/last.pt`.

### 12.4 Commands (PowerShell)

```powershell
# 0. (once, optional) point the script at the dataset if auto-probe misses it
$env:C2A_ROOT = "E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3"

# 1. go to the baseline script's folder
cd "D:\Academics\thesis folder\Last Month\24_01_26- Benchmarking YOLOs\Yolov11m"

# 2. THESIS baseline — 1 seed, full metrics. Auto-smokes first if needed.
python yolov11m_thesis.py

# (optional) smoke-only dry run first: set SMOKE_TEST=True in the CONFIG block, then:
#   python yolov11m_thesis.py    # runs smoke, exits without full training

# 3. LATER, post-course — paper rigor (5 seeds). Skips seed 0 if thesis already ran it.
python yolov11m_paper.py
```

To run unattended and capture a log:
```powershell
python yolov11m_thesis.py *>&1 | Tee-Object -FilePath ".\thesis_baseline_run.log"
```

---

## Agent reminder

When writing code this month: target the §3 chain in order, single GPU, seed policy §3.1; produce the §6 MUST metrics (prefer re-evaluating saved weights over retraining where possible); keep §5 comparisons fair; run the §4 cross-dataset eval after training; keep §9 infra on. Defer everything in §6.1 and §11. When a choice isn't covered here, check `system_spec.md`; if still unclear, **ask** — and never silently drop a metric or mix evaluation protocols. Writing the report is the priority the compute serves.

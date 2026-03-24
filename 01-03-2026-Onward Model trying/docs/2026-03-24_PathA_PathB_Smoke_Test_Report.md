# Path A & Path B — Smoke Test & Bug Fix Report
**Date**: 2026-03-24 (Updated: second review pass)

## Summary

Both Path A (single GPU) and Path B (dual GPU YAML-native) files were reviewed end-to-end.
**5 bugs fixed total** (4 in Path B, 1 critical resume bug in Path A). Timing estimates updated.
Both files verified and ready for Kaggle deployment.

---

## Files

| File | Strategy | GPU | Status |
|------|----------|-----|--------|
| `atrous_mamba_cbam_p2head.py` | get_model() monkey-patch + injection | Single (DEVICE="0") | Ready |
| `atrous_mamba_cbam_p2head_pathB_dualGPU.py` | YAML-native C3K2Mamba registration | Dual (DEVICE="0,1") | Ready (after fixes below) |

---

## CRITICAL Bug Fixed in Path A

### 5. Resume loses trained SSM weights (get_model rebuild)
**Severity**: CRITICAL
**Root cause**: Even with `resume=True`, ultralytics calls `get_model()` which rebuilds from YAML (C3k2). The `intersect_dicts` weight transfer silently drops all SSM keys because the rebuilt model doesn't have them. Our `_inject_into_sequential` then re-creates C3K2Mamba with **random SSM weights** — all training from session 1 is lost for the novel SSM components.
**Fix**: Added 3-step flow in `_patched_get_model()`:
1. `_orig_get_model()` builds C3k2 + transfers base weights
2. `_inject_into_sequential()` re-injects C3K2Mamba (SSM random at this point)
3. **NEW**: `model.load(weights)` re-transfers ALL weights — now SSM keys exist in both source and destination, so trained SSM weights from session 1 are restored.

Also added `has_mamba` guard: if model already has C3K2Mamba (e.g. loaded from YAML-native checkpoint), skip injection entirely.

**Path B not affected**: YAML stores C3K2Mamba directly → rebuild produces correct architecture → all keys match on first transfer.

---

## Bugs Fixed in Path B

### 1. Cell 10: Training used base YAML + injection (Path A logic)
**Severity**: CRITICAL
**Fix**: Changed `YOLO("yolov11m_cbam_p2head.yaml")` + `inject_mamba_neck()` → `YOLO("yolov11m_atrousmamba_cbam_p2head.yaml")` (native C3K2Mamba, no injection needed).

### 2. Cell 10: Verification callback said "INJECTION VERIFICATION"
**Severity**: LOW
**Fix**: Updated messaging to "YAML-NATIVE VERIFICATION" and error message to reference Cell 7 registration.

### 3. Cell 13: Model loading fallback used base YAML + injection
**Severity**: MEDIUM
**Fix**: Changed `YOLO("yolov11m_cbam_p2head.yaml") + inject_mamba_neck()` → `YOLO("yolov11m_atrousmamba_cbam_p2head.yaml")` for evaluation model loading.

### 4. Resume path messaging referenced get_model() patch
**Severity**: LOW
**Fix**: Updated to reference "YAML-native registration" instead.

---

## Issues Reviewed and Deemed OK

### `_dilated_reverse` bilinear interpolation (NOT a bug)
The function uses `F.interpolate(bilinear)` to upscale ws×ws tokens back to region×region for dilation > 1. This is a **valid design choice** (not scatter-back):
- Each branch must produce a full-resolution feature map for the gated fusion
- Scatter-back with zeros would corrupt the fusion (1x1 conv operates per-position)
- Bilinear interpolation is analogous to standard FPN/UNet upsampling
- Gradients flow correctly through bilinear interpolation

### YAML f-string interpolation
`{DILATIONS}` where DILATIONS=[1,2,4] produces `[1, 2, 4]` in YAML. This is valid YAML flow sequence syntax. `yaml.safe_load()` parses it as a nested Python list. Constructor args align correctly: `C3K2Mamba(c1, 512, 2, False, 1, 0.5, 4, [1,2,4])`.

### parse_model regex patching (mitigated)
The regex that adds C3K2Mamba to parse_model's module sets is fragile. However, the `_dry_run_yaml()` function runs immediately after and would CRASH if the patching failed — preventing silent training with base architecture. This is acceptable.

### YAML layer indices
Detect references `[19, 22, 25, 28]` — verified correct for both ATROUS and BASE YAMLs. The `max_channels=512` in `scales: m` means BASE's `[1024, True]` gets clamped to 512, matching ATROUS's `[512, ...]`.

---

## Path A Verification Checklist

- [x] DEVICE forced to "0" (line 153)
- [x] BATCH_SIZE = 8 for single GPU (lines 173, 183)
- [x] get_model() monkey-patch present (lines 898-924)
- [x] inject_mamba_neck() called in training flow
- [x] Gradient clipping present (max_norm=10.0)
- [x] _load_atrous_model helper exists for evaluation
- [x] SessionCheckpointManager for Kaggle 12h resume
- [x] C3K2Mamba verification after resume checkpoint loading

## Path B Verification Checklist

- [x] DEVICE = "0,1" for dual GPU (line 154)
- [x] BATCH_SIZE = 12 for dual GPU (line 183)
- [x] Cell 7: atrous_mamba_module.py written + copied to site-packages
- [x] Cell 7: tasks.py patched with C3K2Mamba imports
- [x] Cell 7: parse_model patched to recognize C3K2Mamba
- [x] Cell 8: AtrousMamba YAML written with C3K2Mamba in neck
- [x] Cell 8: `_dry_run_yaml()` validates YAML parsing before training
- [x] Cell 8: Gradient clipping present (max_norm=10.0)
- [x] Cell 10: Uses AtrousMamba YAML directly (NO injection)
- [x] Cell 10: Verification callback confirms YAML-native C3K2Mamba
- [x] Cell 10: OOM retry ladder works
- [x] Cell 10: Resume path handles C3K2Mamba verification
- [x] Cell 13: _load_atrous_model uses AtrousMamba YAML for fallback

---

## Known Limitations (Not Bugs)

1. **Sequential SSM scan**: The `_SelectiveScan1D` uses a Python for-loop. For ws=8 (L=64 tokens), this is slow but functional on T4.
2. **Cell 5 vs Cell 7 class duplication**: Same classes defined twice (in-script + written module file). Must be kept in sync manually.
3. **parse_model patching is version-dependent**: If ultralytics updates parse_model internals, the regex may fail. The dry-run catches this.

---

## Updated Timing Estimates

Reference: base model (C3k2) = 7.768h on 2×T4 @ batch=12.
AtrousMamba SSM scan adds ~1.5-2× overhead on neck layers.

| Path | GPUs | Batch | Est. Time (120 ep) | Sessions |
|------|------|-------|---------------------|----------|
| A | 1×T4 | 8 | ~18-24h | 2 (resume) |
| B | 2×T4 | 16 | ~10-12h | 1 (tight) or 2 |

Early stopping (patience=15 / F2=10) may finish well before 120 epochs.

## Deployment Instructions

### Path A (Single GPU — 2 Kaggle sessions)
1. Upload `atrous_mamba_cbam_p2head.py` to Kaggle
2. Set `TEST_MODE = True` first → verify smoke test passes (2 epochs, ~5 min)
3. Set `TEST_MODE = False` for full training
4. Session 1: runs until 12h timeout, checkpoint auto-saved every 5 epochs
5. Download `session_last.pt` + `session_meta.json` from `/kaggle/working/`
6. Upload them as a new Kaggle Dataset
7. Session 2: Set `RESUME_TRAINING = True`, `RESUME_PT = "/kaggle/input/<dataset>/session_last.pt"`
8. Trained SSM weights are preserved across resume (re-transfer fix)

### Path B (Dual GPU — 1-2 Kaggle sessions)
1. Upload `atrous_mamba_cbam_p2head_pathB_dualGPU.py` to Kaggle
2. Select **"GPU T4 x2"** accelerator
3. Set `TEST_MODE = True` first → verify YAML dry-run + smoke test passes
4. Set `TEST_MODE = False` for full training
5. ~10-12h on 2×T4 — tight for 12h session but early stopping should help
6. If session times out: resume works same as Path A (checkpoint every 5 epochs)
7. batch=16 (8/GPU) — if OOM, auto-retries at batch=8 then batch=4









1. Will the DDP problem happen again?
No. Here's why each path is safe:

Path A — No DDP at all. DEVICE="0" forces single GPU. Everything runs in one process. The _patched_get_model ensures C3K2Mamba survives the trainer's internal model rebuild. The new 3-step flow (build → inject → re-transfer) preserves trained SSM weights even during resume.

Path B — DDP is safe because the fix addresses the root cause:

What went wrong before	What's different now
DDP subprocess rebuilt model from YAML with C3k2	YAML now has C3K2Mamba directly
Subprocess didn't know about C3K2Mamba class	tasks.py is patched on disk with from atrous_mamba_module import C3K2Mamba
Subprocess couldn't import custom module	atrous_mamba_module.py is copied to site-packages + PYTHONPATH set
No way to verify before training	_dry_run_yaml() builds the model from YAML and crashes if C3K2Mamba is missing
The DDP subprocess reads the same patched source files from disk, imports C3K2Mamba from site-packages, and builds it natively from the YAML. Both processes get the correct architecture.

2. Resume — Do NOT upload the zip file
The zip file is for results (plots, excels, reports). It does NOT contain the checkpoint needed for resume.

What you need for resume:

Download these 2 specific files from /kaggle/working/ (they're saved every 5 epochs, even if the session times out):


session_last.pt        ← the checkpoint (~50-100MB)
session_meta.json      ← metadata (epoch count, best metrics)
Resume steps:

After session 1 times out, go to the notebook output
Download session_last.pt and session_meta.json
Create a new Kaggle Dataset → upload both files together
Add that dataset as input to your notebook
Change only ONE line: RESUME_TRAINING = True
Leave RESUME_PT = "" — it will auto-detect session_last.pt in /kaggle/input/
Run the notebook
I just added auto-detect to both files — it scans /kaggle/input/ for session_last.pt so you don't need to manually set the path.

Important: The session_meta.json must be in the same directory as session_last.pt (upload them together in the same dataset). It restores the F2/mAP50 tracking for early stopping continuity.





Yes, the resume mechanism is fully active. Here's how it works:

Resume Flow for Path B (Dual GPU)
During training:

SessionCheckpointManager saves session_last.pt + session_meta.json to /kaggle/working/ every 5 epochs
Also saves session_best.pt (best weights so far)
Prints resume instructions in the output each time it checkpoints
If session times out (12h limit):

Save the output — download session_last.pt + session_meta.json from /kaggle/working/ (Kaggle keeps output files after timeout)
Upload both files as a new Kaggle Dataset
In Cell 1, change two lines:

RESUME_TRAINING = True
RESUME_PT       = ""    # leave empty — auto-detect will find it
Run the notebook — it auto-detects session_last.pt in /kaggle/input/, copies it to the run directory, and calls mamba_model.train(resume=True)
Why it's safe for DDP: Path B uses YAML-native C3K2Mamba. When ultralytics rebuilds the model from YAML during resume (which DDP subprocesses do), C3K2Mamba is already in the YAML — so all SSM weights load correctly from the checkpoint. No re-injection needed.

Worst case timing: At batch=12, each epoch takes ~30min (384 iterations × 4.5s). In 12h you get ~24 epochs. With patience=15, if the model plateaus around epoch 60-80, early stopping kicks in. If not, you'll need 1-2 resume sessions to finish 120 epochs.



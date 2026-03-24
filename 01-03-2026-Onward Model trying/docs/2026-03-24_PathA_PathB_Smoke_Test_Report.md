# Path A & Path B — Smoke Test & Bug Fix Report
**Date**: 2026-03-24

## Summary

Both Path A (single GPU) and Path B (dual GPU YAML-native) files were reviewed end-to-end.
Four bugs were fixed in Path B. Both files are now ready for Kaggle deployment.

---

## Files

| File | Strategy | GPU | Status |
|------|----------|-----|--------|
| `atrous_mamba_cbam_p2head.py` | get_model() monkey-patch + injection | Single (DEVICE="0") | Ready |
| `atrous_mamba_cbam_p2head_pathB_dualGPU.py` | YAML-native C3K2Mamba registration | Dual (DEVICE="0,1") | Ready (after fixes below) |

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

## Deployment Instructions

### Path A (Single GPU — 2 Kaggle sessions)
1. Upload `atrous_mamba_cbam_p2head.py` to Kaggle
2. Set `TEST_MODE = False` for full training
3. First session: ~7-8h training, checkpoint auto-saved
4. Download `session_last.pt` + `session_meta.json`
5. Upload as new Kaggle Dataset
6. Second session: Set `RESUME_TRAINING = True`, `RESUME_PT = "/kaggle/input/<dataset>/session_last.pt"`

### Path B (Dual GPU — 1 Kaggle session)
1. Upload `atrous_mamba_cbam_p2head_pathB_dualGPU.py` to Kaggle
2. Set `TEST_MODE = False` for full training
3. Select "GPU T4 x2" accelerator
4. ~8h for 120 epochs — fits in single 12h session
5. No resume needed (but available if session crashes)

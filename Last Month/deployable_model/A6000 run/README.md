# Joint C2A+SARD deployable model — A6000 PC-2 build

Same logic as `../joint_c2a_sard_train.py` (full §6 metrics, dual-eval on C2A+SARD, resume-safe),
**optimized for PC-2** and **auto-pinned to whichever GPU is free**.

## Go / No-Go verdict: ✅ GO (no blocking problem)
- **Auto-GPU pick (shared box).** At launch the script queries both GPUs via `nvidia-smi` and:
  both free → picks your **assigned GPU 1**; only GPU 0 free → picks GPU 0; **neither free → stays
  on GPU 1 and warns** (never auto-grabs the other user's GPU when both are busy). "Free" =
  <1 GB memory in use **and** <10% util. It then sets `CUDA_VISIBLE_DEVICES=<picked>` *before*
  importing torch, so the picked GPU is the ONLY visible device (`cuda:0` in-process) and the
  other GPU is invisible — you cannot touch it. To force a fixed GPU, set `GPU_AUTO_SELECT=False`
  (uses `GPU_ID_FALLBACK`, default 1). The picked physical id is logged to `env.json`.
- **256 GB RAM is shared.** Not a problem: `cache='ram'` for this dataset is only ~8–12 GB vs
  245 GB free. (If the other user ever hogs RAM, set `CACHE='disk'`.)

## PC-2 specs (confirmed from Task Manager)
- 2× **NVIDIA RTX A6000**, 47.5 GB each (Ampere, TF32-capable). **You: GPU 1.**
- **Intel Xeon Gold 6346**, 2 sockets, **32 cores / 64 logical**, 3.6 GHz base.
- **256 GB** DDR4-3200 (245 GB free). Windows 11.

## What's optimized for this box (and why)
| Setting | PC-1 (4070, 16 GB) | **PC-2 (A6000, 47.5 GB)** | why |
|---|---|---|---|
| `BATCH_SIZE` | 8 | **40** (ladder 40→32→24→16) | fills ~44 GB, pushes util >90% |
| `NOMINAL_BATCH` | 16 | **40** | effective batch = 40 (no accumulation) |
| `NUM_WORKERS` | 4 | **16** | 64 logical cores — keep the big GPU fed |
| `CACHE` | ram | **ram** | ~10 GB, trivial vs 256 GB |
| TF32 | – | **on** | Ampere matmul/conv speedup (deterministic-safe) |
| cudnn.benchmark | – | **off** | kept off so `deterministic=True` + seed stays reproducible |
| GPU pin | device 0 | **auto-pick free GPU** (both free→GPU 1; none free→GPU 1) | isolate GPU 1, only touch a free one |
- To push closer to 48 GB: try `BATCH_SIZE=44` or `48` (the OOM ladder catches it if it doesn't fit).
- **Do NOT let it spill into "shared GPU memory"** (the 128 GB line) — that's system RAM over PCIe and
  is slow. Keep batch within ~47 GB dedicated; `nvidia-smi` should show ~44–47 GB, not more.
- **Windows worker caveat:** if the DataLoader hangs (WinError 1455), drop `NUM_WORKERS` to 8.
- `lr0=0.001` kept (AdamW is robust; this is a fine-tune from the converged C2A model).

## FILES TO TRANSFER to PC-2 (`D:\student_2k20\2007074\`)
1. **This code folder** → e.g. `D:\student_2k20\2007074\A6000_run\` (any name; outputs land beside it).
2. **C2A dataset** → `D:\student_2k20\2007074\common\c2a\C2A_Dataset\new_dataset3\` (train/ val/ test/).
3. **SARD dataset** → `D:\student_2k20\2007074\common\sard\search-and-rescue\` (train/ valid/ test/).
4. **C2A CBAM+P2 init checkpoint** → copy `best.pt` to exactly
   `D:\student_2k20\2007074\c2a_cbam_p2head_best.pt`
   (source on PC-1: `…\Benchmarking YOLOs\CBAM_P2Head\runs\20260602_063759_yolo11m_cbam_p2head_s0_nogit\weights\best.pt`).
   This matches `EXPLICIT_C2A_BEST_PT` in the script. (yolo11m.pt is NOT needed — the checkpoint
   already carries the weights.)

The script auto-detects 2 + 3 (PC-2 root is in its search list); 4 is used directly.

## SET UP the `2007074` Python env on PC-2 (PowerShell)
```powershell
cd D:\student_2k20\2007074
python -m venv 2007074
.\2007074\Scripts\Activate.ps1
python -m pip install --upgrade pip
# CUDA 12.x torch (driver is 2024-era, supports it). cu124 is safe; use cu126 to mirror PC-1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics pandas pyyaml matplotlib opencv-python pycocotools openpyxl thop
# verify the GPUs are visible and it's the A6000:
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```
(If `pycocotools` fails to build on Windows, the script still runs — COCO AP is skipped, the
per-size recall + ultralytics mAP still compute.)

## RUN
```powershell
cd D:\student_2k20\2007074\A6000_run
# smoke first (SMOKE_TEST=True by default): 2-epoch dry run, confirms GPU 1 + data + eval
python joint_c2a_sard_train.py
#   look for: [gpu] ... NVIDIA RTX A6000 | 47.x GB   and a DEPLOYABLE MODEL RESULT block, no errors
# then set SMOKE_TEST = False in the script, save, and run the real thing:
python joint_c2a_sard_train.py
```
- **Failproof:** power-cut / crash → just re-run the same command; it resumes from `last.pt`
  (corrupt-checkpoint recovery built in). `save_period=25`.
- **Confirm utilization** in a second terminal: `nvidia-smi -i 1 -l 2` → GPU 1 util should be
  >90% and memory ~44–47 GB during training. If util is low, raise workers; if memory < 40 GB,
  raise BATCH_SIZE to 44/48.

## Outputs (same as PC-1 build, full §6 suite)
`runs_joint/<id>/weights/best.pt` (deployable) + `metrics/deployable_summary.json` (C2A-test +
SARD-test + efficiency) + per-image/per-size/PR/F1/calibration CSVs + plots + env.json + manifest.json.

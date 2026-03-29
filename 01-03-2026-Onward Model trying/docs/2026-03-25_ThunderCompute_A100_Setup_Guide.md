# ThunderCompute A100 — AtrousMamba Training Setup Guide
**Date:** 2026-03-25

---

## Your Instance Specs
| Resource | Value |
|----------|-------|
| GPU | NVIDIA A100 80GB VRAM (1 GPU) |
| CPU | 8 vCPUs / 64GB RAM |
| Storage | 100GB |
| Mode | Prototyping ($1.02/hr) |
| Budget | $20 (~19.6 hours) |

## Estimated Training Time
| Epochs | Est. Time | Est. Cost |
|--------|-----------|-----------|
| 80 (default) | ~4-7h | $4-7 |
| 60 (with early stop) | ~3-5h | $3-5 |

Architecture is optimized: 3 C3K2Mamba layers at deep levels only + frozen backbone. Early stopping (patience=15) will likely end training at 50-70 epochs.

---

## Step-by-Step Setup

### Step 1: Create Instance
1. Open VSCode → Thunder Compute sidebar panel
2. Click **Create Instance** (+ icon)
3. Select:
   - GPU: **NVIDIA A100 (80GB)**
   - GPUs: **1**
   - CPU: **8 vCPUs / 64GB RAM**
   - Storage: **100GB**
   - Mode: **Prototyping** (Production toggle OFF)
   - Template: **base**
4. Click **Create Instance**
5. Wait ~60 seconds for it to spin up

### Step 2: Connect
1. Click **Connect** button next to your instance in the Thunder Compute panel
2. A new VSCode window opens — you're now working on the remote A100 machine
3. Every terminal, file explorer, and extension now runs remotely

### Step 3: Set Up Project Directory
Open a terminal in the remote VSCode window (Ctrl+`):

```bash
mkdir -p ~/atrousmamba
mkdir -p ~/data
cd ~/atrousmamba
```

### Step 4: Upload Dataset (C2A)

**Dataset:** https://www.kaggle.com/datasets/rgbnihal/c2a-dataset

**Option A: Kaggle API (recommended)**
```bash
pip install kaggle
mkdir -p ~/.kaggle

# Create your API key from https://www.kaggle.com/settings → API → Create New Token
# This downloads kaggle.json. Upload it:
# (drag-and-drop kaggle.json from local to remote ~/atrousmamba/ in VSCode file explorer)

cp ~/atrousmamba/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download -d rgbnihal/c2a-dataset -p ~/data/
cd ~/data && unzip c2a-dataset.zip
```

**Option B: Direct upload via VSCode drag-and-drop**
1. Download the C2A dataset ZIP from Kaggle to your local machine
2. In the remote VSCode file explorer, navigate to `~/data/`
3. Drag-and-drop the ZIP file from your local machine into `~/data/`
4. In terminal: `cd ~/data && unzip c2a-dataset.zip`

After extraction, verify the structure:
```bash
ls ~/data/
# Should contain a folder with train/images/, val/images/, test/images/
# Example: ~/data/C2A_Dataset/new_dataset3/train/images/
```

> **Note:** You do NOT need to upload any separate C2A files — the Kaggle dataset ZIP contains everything (images + labels). The script auto-detects the dataset path under `~/data/`.

### Step 5: Upload Training Script
In the remote VSCode file explorer:
1. Navigate to `~/atrousmamba/`
2. Drag-and-drop `atrous_mamba_a100_thundercompute.py` from your local machine

Or use terminal:
```bash
# If you have the script locally and tnr CLI installed:
# From your LOCAL terminal (not remote):
tnr scp ./atrous_mamba_a100_thundercompute.py 0:/home/user/atrousmamba/
```

### Step 6: Upload Comparison Models (your previous runs)
Upload your previous `best.pt` files for automatic comparison. These are **optional** — the script trains fine without them, but if uploaded, it generates side-by-side comparison plots and tables.

```bash
mkdir -p ~/atrousmamba/comparison_models
# Drag-and-drop your previous best.pt files into ~/atrousmamba/comparison_models/
# Expected filenames (script auto-detects these):
#   cbam_p2head_mamba.pt    — your old Mamba+CBAM+P2 run
#   cbam_p2head_best.pt     — your CBAM+P2 ablation baseline
```

**What to upload:**
- Only the `best.pt` weight files from your previous Kaggle runs
- You do NOT need to upload the C2A dataset separately — Step 4 handles that
- You do NOT need to upload any YAML configs — the script generates them

The script scans `~/atrousmamba/comparison_models/` automatically. If filenames don't match, set paths manually in the script:
```python
OLD_MAMBA_BEST = os.path.expanduser("~/atrousmamba/comparison_models/your_mamba.pt")
CBAM_P2_BEST   = os.path.expanduser("~/atrousmamba/comparison_models/your_cbam.pt")
```

### Step 7: Install Dependencies
```bash
cd ~/atrousmamba
# CRITICAL: Uninstall GUI opencv first, then install headless (server has no display libs)
pip uninstall -y opencv-python opencv-contrib-python
pip install opencv-python-headless
pip install ultralytics timm thop openpyxl scikit-learn tqdm
pip install "pandas<3.0" "matplotlib<3.10"
```
This takes ~2-3 minutes. PyTorch + CUDA are already pre-installed.
> **Note:** The script handles dependencies automatically on run, but pre-installing avoids timeout issues.
> **Important:** You MUST uninstall `opencv-python` before installing `opencv-python-headless`. Otherwise the GUI version stays and causes `libxcb.so.1` errors.

### Step 8: Run Test Mode First
Edit the script to ensure `TEST_MODE = True` (it should be by default):
```bash
cd ~/atrousmamba
python atrous_mamba_a100_thundercompute.py 2>&1 | tee test_run.log
```
This runs 2 epochs on 5% data (~5-10 minutes). Verify:
- All 8/8 smoke tests pass
- YAML dry-run passes (3 C3K2Mamba layers, ~22M params)
- Training completes without errors
- No NaN losses
- Evaluation cells run

### Step 9: Run Full Training
Edit the script: change `TEST_MODE = False`
```bash
cd ~/atrousmamba
nohup python atrous_mamba_a100_thundercompute.py 2>&1 | tee full_run.log &
```

**Why `nohup`?** If your VSCode connection drops, training continues in the background.

**Monitor progress:**
```bash
tail -f full_run.log
```

### Step 10: Download Results When Done
After training completes, download the results:

**Via VSCode:** Right-click files in remote explorer → Download

**Key files to download:**
```
~/atrousmamba/runs/detect/yolo11m_atrousmamba_cbam_p2head/weights/best.pt
~/atrousmamba/runs/detect/yolo11m_atrousmamba_cbam_p2head/weights/last.pt
~/atrousmamba/checkpoints/session_last.pt
~/atrousmamba/checkpoints/session_meta.json
~/atrousmamba/atrousmamba_results.zip  (all plots, excels, reports)
~/atrousmamba/full_run.log
```

**Via CLI (from local terminal):**
```bash
tnr scp 0:/home/user/atrousmamba/atrousmamba_results.zip ./
tnr scp 0:/home/user/atrousmamba/runs/detect/yolo11m_atrousmamba_cbam_p2head/weights/best.pt ./
```

### Step 11: Delete Instance (IMPORTANT — stops billing!)
After downloading all results:
1. Thunder Compute panel → click Delete on your instance
2. Or from local terminal: `tnr delete 0`

**WARNING:** All data on the instance is permanently deleted. Download everything first!

---

## Resuming Training (If Session Interrupted)

If training was interrupted (or you need to continue on a new instance):

1. Download `checkpoints/session_last.pt` + `checkpoints/session_meta.json` from the old instance
2. Create a new instance (same specs)
3. Upload dataset + script + checkpoint files
4. In the script config:
   ```python
   RESUME_TRAINING = True
   RESUME_PT = os.path.expanduser("~/atrousmamba/checkpoints/session_last.pt")
   ```
5. Run: `python atrous_mamba_a100_thundercompute.py`

---

## Troubleshooting

### `ImportError: libxcb.so.1` (or libGL, libgthread)
The server has no display. `ultralytics` pulls in GUI `opencv-python` which needs X11 libs. Fix: **uninstall** the GUI version first, then install headless:
```bash
pip uninstall -y opencv-python opencv-contrib-python
pip install opencv-python-headless
```
The script now does this automatically (uninstall + reinstall). If you still hit this after running the script, run the commands above manually before re-running.

### NaN losses appearing at epoch 2-3
This was caused by the SSM's `exp(dt*A)` exploding. Fixed in the A100 script by:
1. Clamping `dt*A` to `[-20, 0]` before `exp()` — prevents exponential blowup
2. Tightening gradient clipping from `max_norm=10` → `max_norm=1.0`

> **Note:** The 2xT4 dual-GPU version does NOT have these fixes. If you see NaN on the T4 version, switch to the A100 script — it's faster and more stable.

If you still see NaN on A100, try lowering `lr0` from `0.0005` to `0.0002`.

### "CUDA out of memory"
The script auto-retries with smaller batches (32 → 16 → 8). Default batch=32 uses ~20-30GB VRAM on A100 80GB with the optimized architecture (3 C3K2Mamba layers + frozen backbone).

> **Why not batch=64?** The previous 6-layer C3K2Mamba architecture OOM'd at batch=48 and batch=64 due to massive activation memory at P2 resolution (160×160). The optimized architecture removes C3K2Mamba from P2/P3 levels, making batch=32 comfortable.

### Training seems slow
Check `nvidia-smi` in another terminal:
```bash
watch -n 2 nvidia-smi
```
GPU utilization should be 80-100% during training. If low (<50%), the bottleneck is CPU data loading — reduce `workers` or enable `cache='disk'`.

### JIT compilation warning
If you see "JIT unavailable — using Python scan", training still works but is ~30% slower. This can happen in prototyping mode. If speed is critical, recreate the instance in **Production mode** ($1.79/hr).

### Connection dropped during training
If you used `nohup`, training continues. Reconnect via VSCode and `tail -f full_run.log`.

### "This function is not implemented" error
This means a CUDA operation isn't supported in prototyping mode. Options:
1. Recreate in **Production mode**
2. Contact support@thundercompute.com

### `unsupported function cuLaunchCooperativeKernel` + `CUDNN_STATUS_EXECUTION_FAILED`
This specific pair usually appears in **Prototyping mode** when cuDNN picks a kernel path Thunder does not support.

If you do **not** want to switch to Production mode, use a safe CUDA backend path:
```python
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
```

Trade-off: training becomes slower, but runs reliably in prototyping instances.

The latest `atrous_mamba_a100_thundercompute.py` now applies this fallback automatically when it detects this error pattern.

---

## Cost Summary
| Phase | Time | Cost |
|-------|------|------|
| Setup + test run | ~30 min | $0.51 |
| Full training (80 epochs) | ~4-7h | $4-7 |
| **Total** | **~5-8h** | **$5-8** |

Fits within a single 4-8 hour A100 session. Buffer remaining for re-runs or debugging.

## Architecture Notes

### Why only 3 C3K2Mamba layers (not 6)?
The original architecture had C3K2Mamba at ALL 6 neck positions, including the P2 level (128ch, 160×160 spatial resolution). This caused:
- **OOM at batch ≥ 48** — P2-level AtrousSSM creates massive intermediate tensors (3 branches × 2 directions × 400+ windows at 160×160)
- **~30+ min/epoch** — 72 sequential SSM scan operations per forward pass
- **Training instability** — too many randomly-initialized SSM parameters

The optimized architecture places C3K2Mamba only at deep layers (512ch, 40×40 and 20×20) where:
1. Spatial dims are small → fast scanning, low memory
2. Long-range context matters most for semantic understanding
3. Standard C3k2 handles P2/P3 fine-grained details efficiently

### Why freeze backbone?
The backbone (layers 0-10) loads pretrained yolo11m.pt weights. The novel contribution is the AtrousSSM neck. Freezing the backbone:
1. Cuts gradient computation by ~50% → faster epochs
2. Prevents backbone weight corruption from unstable SSM gradients
3. Focuses all learning capacity on the neck + head

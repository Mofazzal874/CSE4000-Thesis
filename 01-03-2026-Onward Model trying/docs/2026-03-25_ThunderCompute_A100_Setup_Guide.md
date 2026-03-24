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
| 100 (default) | ~12-16h | $12-16 |
| 80 (with early stop) | ~10-12h | $10-12 |
| 120 (full) | ~16-20h | $16-20 |

Early stopping (patience=20) will likely end training at 60-80 epochs.

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

**Option A: Kaggle API (recommended if dataset is on Kaggle)**
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
1. Download the C2A dataset ZIP to your local machine first
2. In the remote VSCode file explorer, navigate to `~/data/`
3. Drag-and-drop the ZIP file from your local machine into `~/data/`
4. In terminal: `cd ~/data && unzip c2a-dataset.zip`

**Option C: If dataset is on Google Drive**
```bash
pip install gdown
gdown --id YOUR_GOOGLE_DRIVE_FILE_ID -O ~/data/c2a-dataset.zip
cd ~/data && unzip c2a-dataset.zip
```

After extraction, verify the structure:
```bash
ls ~/data/
# Should contain a folder with train/images/, val/images/, test/images/
# Example: ~/data/C2A_Dataset/new_dataset3/train/images/
```

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

### Step 6: (Optional) Upload Comparison Models
If you want to compare with previous models, upload their `best.pt` files:
```bash
mkdir -p ~/atrousmamba/comparison_models
# Drag-and-drop best.pt files into ~/atrousmamba/comparison_models/
```
Then set the paths in the script's config section:
```python
OLD_MAMBA_BEST = os.path.expanduser("~/atrousmamba/comparison_models/mamba_best.pt")
CBAM_P2_BEST   = os.path.expanduser("~/atrousmamba/comparison_models/cbam_p2_best.pt")
```

### Step 7: Install Dependencies
```bash
cd ~/atrousmamba
pip install ultralytics timm thop openpyxl scikit-learn tqdm
pip install "pandas<3.0" "matplotlib<3.10"
```
This takes ~2-3 minutes. PyTorch + CUDA are already pre-installed.

### Step 8: Run Test Mode First
Edit the script to ensure `TEST_MODE = True` (it should be by default):
```bash
cd ~/atrousmamba
python atrous_mamba_a100_thundercompute.py 2>&1 | tee test_run.log
```
This runs 2 epochs on 5% data (~5-10 minutes). Verify:
- All 8/8 smoke tests pass
- YAML dry-run passes (6 C3K2Mamba layers, ~25M params)
- Training completes without errors
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

### "CUDA out of memory" at batch=48
The script auto-retries with smaller batches [48 → 32 → 16 → 8]. If batch=48 OOMs, it'll fall back automatically.

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

---

## Cost Summary
| Phase | Time | Cost |
|-------|------|------|
| Setup + test run | ~30 min | $0.51 |
| Full training (100 epochs) | ~12-16h | $12-16 |
| **Total** | **~13-17h** | **$13-17** |

Buffer: ~$3-7 remaining for re-runs or debugging.

"""
================================================================================
CBAM + P2 HEAD — CONTINUATION RUN (Stage 2 Fine-Tune)
================================================================================
PURPOSE:
  Load the best.pt from your completed 70-epoch CBAM+P2 run and continue
  training for up to 30 more epochs with full early stopping, all metrics,
  and session-resumable checkpoints.

WHAT THIS IS NOT:
  This is NOT a resume (which would reload optimizer/LR state from last.pt).
  This is a second-stage fine-tune: the model starts from your best weights
  and trains with a lower LR and fresh schedule. This is intentional —
  it gives the model a second annealing pass from a strong starting point.

HOW TO FIND THE INPUT:
  Upload your saved working directory zip to a Kaggle Dataset.
  Kaggle auto-extracts zips. This script will walk /kaggle/input/ and find
  best.pt automatically — no manual path editing needed.

NEW vs ORIGINAL SCRIPT:
  ✓ F1 / F2 computed per epoch from results.csv
  ✓ F2-based early stopping (recall-weighted, correct for SAR/disaster)
  ✓ NaN/Inf detection callback (stops + saves emergency checkpoint)
  ✓ Gradient norm monitoring per epoch
  ✓ Session checkpoint every N epochs (survive 12h Kaggle reset)
  ✓ OOM retry with automatic batch halving
  ✓ ECE calibration plot
  ✓ Confidence distribution histogram
  ✓ Per-size recall breakdown (very tiny → large)
  ✓ 6-panel metric plot (P, R, mAP50, mAP50-95, F1, F2)
  ✓ Stitched training curve (ep1-70 from original + ep71-100 from continuation)
  ✓ Failure mode analysis (dangerous high-conf low-recall images)
  ✓ Speed benchmark at 4 resolutions
  ✓ Master report + JSON export for reproducibility

TRAINING STRATEGY (Stage 2):
  lr0         = 0.0001  (10× lower than original — fine-tune, not re-train)
  warmup      = 2 epochs (shorter — weights already well-trained)
  patience    = 15 epochs (mAP50, Ultralytics built-in)
  F2 patience = 10 epochs (custom recall-weighted early stop)
  epochs      = 50 continuation epochs (total ≈ 100)
================================================================================
"""

# ============================================================================
# CELL 1: Control Flags & Dependencies
# ============================================================================

# ==================== CONTROL FLAGS ====================
TEST_MODE       = False   # True = 5% data, 2 epochs (smoke test)
                         # False = full 30-epoch continuation
RESUME_TRAINING = False  # True = resume continuation from a mid-run checkpoint
RESUME_PT       = ""     # Only used if RESUME_TRAINING=True. Point to:
                         # "/kaggle/input/<your-dataset>/session_last.pt"
# =======================================================

import subprocess, sys, re

def _pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", pkg])

# ── CUDA / PyTorch version alignment ─────────────────────────────────────────
# Kaggle T4 driver = CUDA 12.6, but default PyTorch ships as cu124.
# This mismatch can cause AMP instability. Detect and fix before any imports.
print("── Checking CUDA / PyTorch version alignment ──")
_smi        = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
_drv_cuda   = re.search(r"CUDA Version:\s*([\d.]+)", _smi)
_drv_cuda   = _drv_cuda.group(1) if _drv_cuda else "unknown"
_torch_str  = subprocess.run(
    [sys.executable, "-c", "import torch; print(torch.__version__)"],
    capture_output=True, text=True).stdout.strip()
print(f"  Driver CUDA : {_drv_cuda}  |  PyTorch : {_torch_str}")

if _drv_cuda.startswith("12.6") and "cu124" in _torch_str:
    print("  ⚠  Mismatch detected — reinstalling PyTorch cu126 (~3 min) …")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu126"
    ])
    print("  ✓ PyTorch cu126 installed")
else:
    print("  ✓ No mismatch")

# ── Packages ─────────────────────────────────────────────────────────────────
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "-q",
                       "ultralytics"], stderr=subprocess.DEVNULL)
for _pkg in ["ultralytics", "timm", "thop", "openpyxl", "scikit-learn"]:
    _pip(_pkg)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "pandas<3.0", "matplotlib<3.10", "tqdm"])
print("✓ All dependencies installed")


# ============================================================================
# CELL 2: Imports & Configuration
# ============================================================================
import os, sys, time, yaml, shutil, gc, json, math, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ── GPU fingerprint ───────────────────────────────────────────────────────────
num_gpus = torch.cuda.device_count()
gpu_name = torch.cuda.get_device_name(0) if num_gpus > 0 else "CPU"
gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3 if num_gpus > 0 else 0
DEVICE   = "0,1" if num_gpus >= 2 else "0" if num_gpus == 1 else "cpu"
BF16_OK  = torch.cuda.is_bf16_supported() if num_gpus > 0 else False
print(f"GPUs: {num_gpus}  |  {gpu_name} ({gpu_mem:.1f} GB)  |  DEVICE: {DEVICE}")
print(f"AMP dtype: {'BF16' if BF16_OK else 'FP16 (P100 safe)'}")

# ── Training config ───────────────────────────────────────────────────────────
if TEST_MODE:
    CONTINUE_EPOCHS = 2
    TRAIN_FRACTION  = 0.05
    PATIENCE_MAP    = 2       # Ultralytics mAP patience
    F2_PATIENCE     = 2       # Custom F2 recall-weighted patience
    TEST_IMAGES     = 10      # Images used in custom eval loop
    VAL_IMAGES      = 20
    SAVE_PERIOD     = 1       # Save checkpoint every N epochs
    CHECKPOINT_EVERY= 1       # Copy to /working/ every N epochs
    BATCH_SIZE      = 4
    print("⚠  TEST MODE — 5% data, 2 epochs")
else:
    CONTINUE_EPOCHS = 50      # 70 already done → target ≈ 100 total
    TRAIN_FRACTION  = 1.0
    PATIENCE_MAP    = 15
    F2_PATIENCE     = 10
    TEST_IMAGES     = None    # Full test set
    VAL_IMAGES      = None    # Full val set
    SAVE_PERIOD     = 5
    CHECKPOINT_EVERY= 5
    BATCH_SIZE      = 8 if gpu_mem >= 14 else 4
    print(f"🚀  FULL MODE — {CONTINUE_EPOCHS} continuation epochs | batch={BATCH_SIZE}")

# OOM retry ladder: if batch=8 OOMs, try 4 then 2
OOM_RETRY_BATCHES = [BATCH_SIZE, max(BATCH_SIZE // 2, 2), max(BATCH_SIZE // 4, 1)]

# Fine-tune LR (10× lower than the original 0.001 — DO NOT change this)
FINETUNE_LR = 0.0001
RUN_NAME    = "yolo11m_cbam_p2head_continued"

print(f"  Fine-tune LR: {FINETUNE_LR}  |  OOM retry batches: {OOM_RETRY_BATCHES}")


# ============================================================================
# CELL 3: Dataset Configuration
# ============================================================================
# Auto-discover C2A dataset root — Kaggle mounts at unpredictable paths
# based on dataset owner/slug, so we walk /kaggle/input/ to find it.
print("Searching for C2A dataset in /kaggle/input/ …")
DATASET_ROOT = None
for _root, _dirs, _files in os.walk("/kaggle/input"):
    # Look for the folder that has train/images, val/images, test/images
    if (os.path.isdir(os.path.join(_root, "train", "images")) and
        os.path.isdir(os.path.join(_root, "val",   "images"))):
        DATASET_ROOT = _root
        print(f"  ✓ C2A dataset found at: {DATASET_ROOT}")
        break

if DATASET_ROOT is None:
    # Print what IS available to help diagnose
    print("  ❌ C2A dataset NOT found. Contents of /kaggle/input/:")
    for _root, _dirs, _files in os.walk("/kaggle/input"):
        _lvl = _root.replace("/kaggle/input", "").count(os.sep)
        if _lvl < 4:
            print("  " * _lvl + f"📁 {os.path.basename(_root)}/")
            for _f in _files[:3]:
                print("  " * _lvl + f"  📄 {_f}")
    raise FileNotFoundError(
        "C2A dataset not found under /kaggle/input/.\n"
        "Make sure you added the 'C2A Dataset' as input to this notebook."
    )

TEST_IMG_DIR = f"{DATASET_ROOT}/test/images"
TEST_LBL_DIR = f"{DATASET_ROOT}/test/labels"
VAL_IMG_DIR  = f"{DATASET_ROOT}/val/images"
VAL_LBL_DIR  = f"{DATASET_ROOT}/val/labels"

_yaml = f"""train: {DATASET_ROOT}/train/images
val:   {DATASET_ROOT}/val/images
test:  {DATASET_ROOT}/test/images
nc: 1
names: ['person']
"""
with open("c2a.yaml", "w") as f:
    f.write(_yaml.strip())
print("✓ c2a.yaml written")


# ============================================================================
# CELL 4: CBAM Module (must be IDENTICAL to your original run)
# ============================================================================
# The saved best.pt contains CBAM layer weights. The CBAM class definition
# must match exactly — any change breaks weight loading.

cbam_code = '''
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        reduced_channels = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced_channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_channels, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return x * self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(concat))

class CBAM(nn.Module):
    """CBAM with lazy initialisation — identical to original study."""
    def __init__(self, *args, **kwargs):
        super(CBAM, self).__init__()
        self.reduction = 16
        self.kernel_size = 7
        if len(args) == 0:
            pass
        elif len(args) == 1:
            if isinstance(args[0], int) and args[0] <= 32:
                self.reduction = args[0]
        elif len(args) == 2:
            if isinstance(args[0], int) and args[0] <= 32:
                self.reduction = args[0]
                self.kernel_size = args[1] if isinstance(args[1], int) else 7
        elif len(args) >= 4:
            self.reduction   = args[2] if isinstance(args[2], int) else 16
            self.kernel_size = args[3] if isinstance(args[3], int) else 7
        self.reduction   = kwargs.get("reduction",   self.reduction)
        self.kernel_size = kwargs.get("kernel_size", self.kernel_size)
        if self.kernel_size not in (3, 7):
            self.kernel_size = 7
        self._initialized      = False
        self.channel_attention = None
        self.spatial_attention = None
        self._channels         = None

    def _lazy_init(self, channels, device, dtype):
        self._channels         = channels
        self.channel_attention = ChannelAttention(channels, self.reduction).to(device=device, dtype=dtype)
        self.spatial_attention = SpatialAttention(self.kernel_size).to(device=device, dtype=dtype)
        self._initialized      = True

    def forward(self, x):
        if not self._initialized:
            self._lazy_init(x.size(1), x.device, x.dtype)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
'''

with open("/kaggle/working/cbam_module.py", "w") as f:
    f.write(cbam_code)
exec(cbam_code)

# Smoke test
_t = torch.randn(2, 512, 20, 20)
assert CBAM(16, 7)(_t).shape == _t.shape
print("✓ CBAM module OK")


# ============================================================================
# CELL 5: Register CBAM in Ultralytics (DDP-safe)
# ============================================================================
# WHY THIS IS NEEDED:
#   Ultralytics DDP spawns completely fresh Python subprocesses via
#   torch.distributed.run. Those processes re-import ultralytics from scratch.
#   Two things must happen for each DDP worker:
#     1. `import cbam_module` must work  → install to site-packages
#     2. `globals()["CBAM"]` in tasks.py must exist  → patch the file on disk
#
# setattr(ult_tasks, "CBAM", ...) only affects the current process's memory.
# It has NO effect on DDP subprocesses. We must patch the actual .py files.

import site as _site

# ── Step 1: Install cbam_module to site-packages ─────────────────────────────
_cbam_src = "/kaggle/working/cbam_module.py"
_installed = False
for _sp in _site.getsitepackages():
    try:
        shutil.copy(_cbam_src, os.path.join(_sp, "cbam_module.py"))
        print(f"  ✓ cbam_module.py → site-packages: {_sp}")
        _installed = True
        break
    except Exception:
        continue
if not _installed:
    print("  ⚠  Could not install to site-packages; DDP may still fail")

# Also keep /kaggle/working on path for the main process
if "/kaggle/working" not in sys.path:
    sys.path.insert(0, "/kaggle/working")

# Set PYTHONPATH so DDP child processes inherit the path too
_existing_pypath = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = "/kaggle/working" + (
    ":" + _existing_pypath if _existing_pypath else "")

# ── Step 2: Physically patch ultralytics/nn/tasks.py on disk ─────────────────
# parse_model() uses globals() of tasks.py to resolve YAML layer names like
# "CBAM". We must inject the import INTO that file so every subprocess that
# does `import ultralytics.nn.tasks` gets CBAM in its global namespace.
import ultralytics.nn.tasks as _ult_tasks_mod
_tasks_file = _ult_tasks_mod.__file__
_inject      = "from cbam_module import CBAM, ChannelAttention, SpatialAttention\n"

with open(_tasks_file, "r") as _f:
    _tasks_src = _f.read()

if "from cbam_module import" not in _tasks_src:
    # Inject right after the first import block (after the first 'import' line)
    _tasks_src = _inject + _tasks_src
    with open(_tasks_file, "w") as _f:
        _f.write(_tasks_src)
    print(f"  ✓ Patched ultralytics/nn/tasks.py with CBAM import")
else:
    print(f"  ✓ ultralytics/nn/tasks.py already patched")

# ── Step 3: Also patch ultralytics/nn/modules/__init__.py ────────────────────
import ultralytics.nn.modules as _ult_modules_mod
_modules_init = os.path.join(os.path.dirname(_ult_modules_mod.__file__), "__init__.py")
with open(_modules_init, "r") as _f:
    _modules_src = _f.read()

if "from cbam_module import" not in _modules_src:
    _modules_src = _inject + _modules_src
    with open(_modules_init, "w") as _f:
        _f.write(_modules_src)
    print(f"  ✓ Patched ultralytics/nn/modules/__init__.py with CBAM import")
else:
    print(f"  ✓ ultralytics/nn/modules/__init__.py already patched")

# ── Step 4: Reload modules so the main process picks up the patch too ─────────
import importlib
importlib.reload(_ult_tasks_mod)
importlib.reload(_ult_modules_mod)

from cbam_module import CBAM, ChannelAttention, SpatialAttention
import ultralytics.nn.modules as ult_modules
import ultralytics.nn.tasks  as ult_tasks

# Belt-and-suspenders: setattr as well for the main process
for _name, _obj in [("CBAM", CBAM),
                     ("ChannelAttention", ChannelAttention),
                     ("SpatialAttention", SpatialAttention)]:
    setattr(ult_modules, _name, _obj)
    setattr(ult_tasks,   _name, _obj)

assert hasattr(ult_modules, "CBAM"), "CBAM registration failed"
print("✓ CBAM registered in ultralytics namespace (DDP-safe)")


# ============================================================================
# CELL 6: CBAM+P2 YAML (must match original architecture exactly)
# ============================================================================
cbam_p2_yaml = """# YOLO11m + CBAM + P2 Head — identical to original ablation study
nc: 1
scales:
  m: [0.50, 1.00, 512]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 1, CBAM, [16, 7]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, False]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]
  - [[19, 22, 25, 28], 1, Detect, [nc]]
"""
with open("yolov11m_cbam_p2head.yaml", "w") as f:
    f.write(cbam_p2_yaml)

# Sanity check YAML
_cfg = yaml.safe_load(cbam_p2_yaml)
assert len(_cfg["head"][-1][0]) == 4, "P2 YAML: wrong number of detection scales"
assert any(l[2] == "CBAM" for l in _cfg["backbone"] if len(l) >= 3), "CBAM missing from backbone"
print("✓ CBAM+P2 YAML verified (4 detection scales, CBAM in backbone)")


# ============================================================================
# CELL 7: Auto-detect Previous Run & Validate Paths
# ============================================================================
# Searches /kaggle/input/ for the best.pt and results.csv from your
# original 70-epoch CBAM+P2 run. Upload your working dir zip as a
# Kaggle Dataset — it will be auto-extracted and found here.

print("\nSearching for previous CBAM+P2 run in /kaggle/input/ …")

PREV_BEST_PT      = None
PREV_RESULTS_CSV  = None

_TARGET_BEST = os.path.join("runs", "detect", "yolo11m_cbam_p2head",
                              "weights", "best.pt")
_TARGET_CSV  = os.path.join("runs", "detect", "yolo11m_cbam_p2head",
                              "results.csv")

for _root, _dirs, _files in os.walk("/kaggle/input"):
    _cand = os.path.join(_root, _TARGET_BEST)
    if os.path.isfile(_cand):
        PREV_BEST_PT     = _cand
        PREV_RESULTS_CSV = os.path.join(_root, _TARGET_CSV)
        print(f"  ✓ Found previous run at: {_root}")
        break

if PREV_BEST_PT is None:
    # Fallback: search for any best.pt named cbam_p2head
    print("  Standard path not found. Trying fallback search …")
    for _root, _dirs, _files in os.walk("/kaggle/input"):
        for _f in _files:
            if _f == "best.pt" and "cbam_p2" in _root:
                PREV_BEST_PT = os.path.join(_root, _f)
                _csv_try = os.path.join(Path(_root).parent.parent, "results.csv")
                PREV_RESULTS_CSV = _csv_try if os.path.exists(_csv_try) else None
                print(f"  ✓ Fallback found: {PREV_BEST_PT}")
                break
        if PREV_BEST_PT:
            break

if PREV_BEST_PT is None:
    # Print what IS in /kaggle/input to help diagnose
    print("\n  ❌ CBAM+P2 best.pt NOT FOUND. Contents of /kaggle/input/:")
    for _root, _dirs, _files in os.walk("/kaggle/input"):
        _lvl = _root.replace("/kaggle/input", "").count(os.sep)
        if _lvl < 4:
            _indent = "  " * _lvl
            print(f"{_indent}📁 {os.path.basename(_root)}/")
            for _f in _files[:4]:
                print(f"{_indent}  📄 {_f}")
    raise FileNotFoundError(
        "Previous CBAM+P2 best.pt not found.\n"
        "Steps:\n"
        "  1. Go to kaggle.com/datasets → New Dataset\n"
        "  2. Upload your saved working directory zip\n"
        "  3. Add it as input to this notebook under 'Add Data'\n"
        "The zip must contain the path: runs/detect/yolo11m_cbam_p2head/weights/best.pt"
    )

print(f"  ✓ best.pt     : {PREV_BEST_PT}")
if PREV_RESULTS_CSV and os.path.exists(PREV_RESULTS_CSV):
    print(f"  ✓ results.csv : {PREV_RESULTS_CSV}")
else:
    print("  ⚠  results.csv not found — stitched training curve will be skipped")
    PREV_RESULTS_CSV = None


# ============================================================================
# CELL 8: Custom Callbacks
# ============================================================================

class NaNStopCallback:
    """Stop immediately on NaN/Inf loss. Save emergency checkpoint first."""
    def __init__(self):
        self.triggered = False

    def on_train_batch_end(self, trainer):
        if self.triggered:
            return
        loss = getattr(trainer, "loss", None)
        if loss is not None and not torch.isfinite(loss):
            self.triggered = True
            print(f"\n🚨 NaN/Inf loss at epoch {trainer.epoch+1}! Stopping.")
            _emg = Path(trainer.save_dir) / "weights" / "emergency_nan.pt"
            try:
                trainer.model.save(str(_emg))
                print(f"   Emergency weights → {_emg}")
            except Exception:
                pass
            try:
                trainer.stop = True
            except AttributeError:
                trainer.epoch = trainer.epochs


class F2EarlyStopCallback:
    """
    Early stopping based on F2 score (β=2).
    F2 = 5·P·R / (4·P + R) — weights recall 2× more than precision.
    This is the correct stopping criterion for disaster survivor detection
    where a missed person is far worse than a false alarm.
    """
    def __init__(self, patience: int = 10, min_delta: float = 5e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best_f2   = 0.0
        self.counter   = 0
        self.history   = []   # list of (epoch, f2)

    def on_fit_epoch_end(self, trainer):
        m  = trainer.metrics
        P  = float(m.get("metrics/precision(B)", 0))
        R  = float(m.get("metrics/recall(B)",    0))
        f2 = 5 * P * R / (4 * P + R + 1e-9)
        self.history.append((trainer.epoch, f2))

        if f2 > self.best_f2 + self.min_delta:
            self.best_f2 = f2
            self.counter = 0
        else:
            self.counter += 1
            left = self.patience - self.counter
            if self.counter >= self.patience:
                print(f"\n⏹  F2 Early Stop triggered — no improvement "
                      f"for {self.patience} epochs. Best F2={self.best_f2:.4f}")
                try:
                    trainer.stop = True
                except AttributeError:
                    trainer.epoch = trainer.epochs
            elif left <= 3:
                print(f"  ⚠  F2 patience: {left} epoch(s) left "
                      f"(best={self.best_f2:.4f}, now={f2:.4f})")


class GradientMonitorCallback:
    """Log gradient norm per epoch. Warn on explosion or vanishing."""
    def __init__(self, save_dir: str = "/kaggle/working"):
        self.save_dir = save_dir
        self.records  = []

    def on_train_epoch_end(self, trainer):
        total_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in trainer.model.parameters()
            if p.grad is not None
        ) ** 0.5
        self.records.append({"epoch": trainer.epoch + 1, "grad_norm": total_norm})
        if total_norm > 200:
            print(f"  ⚠  Grad explosion: norm={total_norm:.1f} @ ep{trainer.epoch+1}")
        elif total_norm < 1e-7 and len(self.records) > 2:
            print(f"  ⚠  Vanishing grads: norm={total_norm:.2e} @ ep{trainer.epoch+1}")

    def on_train_end(self, trainer):
        if self.records:
            pd.DataFrame(self.records).to_csv(
                f"{self.save_dir}/grad_norms.csv", index=False)
            print(f"  ✓ Gradient norms → {self.save_dir}/grad_norms.csv")


class SessionCheckpointManager:
    """
    Kaggle 12-hour session survival kit.

    Every CHECKPOINT_EVERY epochs:
      • Copies last.pt → /kaggle/working/session_last.pt
      • Copies best.pt → /kaggle/working/session_best.pt
      • Writes session_meta.json with epoch, best metrics, train args

    HOW TO RESUME after session reset:
      1. Download session_last.pt + session_meta.json from /kaggle/working/
      2. Upload both to a NEW Kaggle Dataset
      3. In CELL 1 of next session:
           RESUME_TRAINING = True
           RESUME_PT = "/kaggle/input/<your-dataset>/session_last.pt"
      4. Run — training continues from saved epoch

    Use last.pt (not best.pt) for resuming — last.pt has optimizer state.
    """
    def __init__(self, checkpoint_every: int = 5,
                 working_dir: str = "/kaggle/working"):
        self.every       = checkpoint_every
        self.wdir        = Path(working_dir)
        self.best_f2     = 0.0
        self.best_map50  = 0.0
        self._args       = {}

    def register_args(self, args: dict):
        self._args = {k: str(v) for k, v in args.items()}

    def _copy(self, src: Path, dst: Path) -> bool:
        if src.exists():
            shutil.copy(str(src), str(dst))
            return True
        return False

    def _write_meta(self, epoch: int, save_dir: Path):
        meta = {
            "completed_epochs": epoch,
            "best_f2":          round(self.best_f2,    4),
            "best_map50":       round(self.best_map50, 4),
            "run_name":         RUN_NAME,
            "train_args":       self._args,
            "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.wdir / "session_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def on_fit_epoch_end(self, trainer):
        m    = trainer.metrics
        P    = float(m.get("metrics/precision(B)", 0))
        R    = float(m.get("metrics/recall(B)",    0))
        f2   = 5 * P * R / (4 * P + R + 1e-9)
        m50  = float(m.get("metrics/mAP50(B)", 0))
        if f2  > self.best_f2:    self.best_f2    = f2
        if m50 > self.best_map50: self.best_map50 = m50

        epoch = trainer.epoch + 1
        if epoch % self.every != 0:
            return

        save_dir = Path(trainer.save_dir)
        ok = self._copy(save_dir / "weights" / "last.pt",
                        self.wdir / "session_last.pt")
        self._copy(save_dir / "weights" / "best.pt",
                   self.wdir / "session_best.pt")
        self._write_meta(epoch, save_dir)

        if ok:
            print(f"\n{'─'*60}")
            print(f"  📥 CHECKPOINT @ epoch {epoch}")
            print(f"     Best F2={self.best_f2:.4f}  mAP50={self.best_map50:.4f}")
            print(f"     Files in /kaggle/working/:")
            print(f"       • session_last.pt  ← use this to resume")
            print(f"       • session_best.pt")
            print(f"       • session_meta.json")
            print(f"{'─'*60}\n")

    def on_train_end(self, trainer):
        save_dir = Path(trainer.save_dir)
        self._copy(save_dir / "weights" / "last.pt",
                   self.wdir / "session_last.pt")
        self._copy(save_dir / "weights" / "best.pt",
                   self.wdir / "session_best.pt")
        self._write_meta(trainer.epoch + 1, save_dir)
        print(f"\n📥 Final checkpoint saved")
        print(f"   Best F2={self.best_f2:.4f}  Best mAP50={self.best_map50:.4f}")


print("✓ Callbacks defined: NaNStop, F2EarlyStop, GradientMonitor, SessionCheckpointManager")


# ============================================================================
# CELL 9: Training (OOM-safe, session-resumable)
# ============================================================================
from ultralytics import YOLO

print("\n" + "=" * 80)
print(f"STAGE 2 FINE-TUNE: CBAM+P2 Head ({CONTINUE_EPOCHS} continuation epochs)")
print(f"Starting weights: {PREV_BEST_PT}")
print("=" * 80)

CONT_RUN_DIR = f"runs/detect/{RUN_NAME}"
CONT_BEST    = f"{CONT_RUN_DIR}/weights/best.pt"

# ── Instantiate callbacks ────────────────────────────────────────────────────
nan_cb   = NaNStopCallback()
f2_cb    = F2EarlyStopCallback(patience=F2_PATIENCE)
grad_cb  = GradientMonitorCallback()
ckpt_mgr = SessionCheckpointManager(CHECKPOINT_EVERY)

def _attach_callbacks(model):
    model.add_callback("on_train_batch_end", nan_cb.on_train_batch_end)
    model.add_callback("on_fit_epoch_end",   f2_cb.on_fit_epoch_end)
    model.add_callback("on_fit_epoch_end",   ckpt_mgr.on_fit_epoch_end)
    model.add_callback("on_train_epoch_end", grad_cb.on_train_epoch_end)
    model.add_callback("on_train_end",       grad_cb.on_train_end)
    model.add_callback("on_train_end",       ckpt_mgr.on_train_end)


# ── RESUME PATH (mid-continuation checkpoint) ────────────────────────────────
if RESUME_TRAINING and RESUME_PT:
    _src = Path(RESUME_PT)
    if not _src.exists():
        raise FileNotFoundError(
            f"RESUME_PT not found: {RESUME_PT}\n"
            "Add the checkpoint dataset as Kaggle input first."
        )
    # Copy into expected run directory (Ultralytics resume=True needs it there)
    _wts = Path(CONT_RUN_DIR) / "weights"
    _wts.mkdir(parents=True, exist_ok=True)
    _local = _wts / "last.pt"
    if not _local.exists():
        shutil.copy(str(_src), str(_local))
        print(f"  ✓ Copied {_src.name} → {_local}")

    # Restore best metrics from metadata if available
    _meta_src = _src.parent / "session_meta.json"
    if _meta_src.exists():
        _meta = json.load(open(_meta_src))
        ckpt_mgr.best_f2    = float(_meta.get("best_f2",    0))
        ckpt_mgr.best_map50 = float(_meta.get("best_map50", 0))
        print(f"  Resuming from epoch {_meta.get('completed_epochs','?')}  "
              f"Best F2={ckpt_mgr.best_f2:.4f}")

    cont_model = YOLO(str(_local))
    _attach_callbacks(cont_model)
    print(f"\n⚡ RESUMING continuation from {_local}")
    cont_model.train(resume=True)
    print("✓ Resumed training complete")

# ── FRESH STAGE-2 FINE-TUNE with OOM retry ───────────────────────────────────
else:
    _trained_ok = False

    for _batch in OOM_RETRY_BATCHES:
        print(f"\n→ Training attempt: batch_size={_batch}")

        cont_model = YOLO(PREV_BEST_PT)   # loads best weights from original run
        print(f"  ✓ Loaded best weights from: {PREV_BEST_PT}")
        cont_model.info(verbose=False)
        _attach_callbacks(cont_model)

        _train_kwargs = dict(
            data          = "c2a.yaml",
            epochs        = CONTINUE_EPOCHS,
            imgsz         = 640,
            batch         = _batch,
            device        = DEVICE,
            optimizer     = "AdamW",
            lr0           = FINETUNE_LR,   # 0.0001 — fine-tune rate
            lrf           = 0.01,
            weight_decay  = 0.0005,
            momentum      = 0.937,
            warmup_epochs = 2,             # shorter warmup — weights already good
            close_mosaic  = 5,
            amp           = True,
            patience      = PATIENCE_MAP,
            save          = True,
            save_period   = SAVE_PERIOD,
            plots         = True,          # auto-generates PR curve, F1 curve
            verbose       = True,
            fraction      = TRAIN_FRACTION,
            cache         = True,
            workers       = 2,
            name          = RUN_NAME,
            exist_ok      = True,
        )
        ckpt_mgr.register_args(_train_kwargs)

        try:
            cont_model.train(**_train_kwargs)
            _trained_ok = True
            print(f"\n✓ Continuation training complete (batch={_batch})")
            break

        except torch.cuda.OutOfMemoryError as _oom:
            print(f"\n💥 OOM at batch={_batch}: {_oom}")
            # save partial checkpoint if it exists
            _partial = Path(CONT_RUN_DIR) / "weights" / "last.pt"
            if _partial.exists():
                shutil.copy(str(_partial), "/kaggle/working/session_last.pt")
                ckpt_mgr._write_meta(-1, Path(CONT_RUN_DIR))
                print(f"  ⚠  Partial checkpoint saved → session_last.pt (download now!)")
            del cont_model
            torch.cuda.empty_cache(); gc.collect(); time.sleep(3)
            if _batch == OOM_RETRY_BATCHES[-1]:
                raise RuntimeError(
                    "OOM on all retry attempts.\n"
                    "Options: reduce imgsz to 512, or switch to P100 GPU."
                ) from _oom

    if not _trained_ok:
        raise RuntimeError("Training did not complete — check OOM messages above.")

del cont_model; torch.cuda.empty_cache(); gc.collect()


# ============================================================================
# CELL 10: Output Directories
# ============================================================================
BASE_DIR   = "/kaggle/working"
EXCEL_DIR  = f"{BASE_DIR}/excel_reports"
PLOT_DIR   = f"{BASE_DIR}/plots"
REPORT_DIR = f"{BASE_DIR}/benchmark_reports"
for _d in [EXCEL_DIR, PLOT_DIR, REPORT_DIR]:
    os.makedirs(_d, exist_ok=True)
print("✓ Output directories ready")


# ============================================================================
# CELL 11: Training Curves
# ============================================================================
# Produces three things:
#  A) Individual per-run plots (loss + 6-panel metrics for the new run)
#  B) Stitched curve (ep1-70 original + ep71-100 continuation)
#     This shows the reviewer the full training trajectory in one plot.
#  C) Overlay comparison (original vs continuation) where available

def _load_csv(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    P = df.get("metrics/precision(B)", pd.Series(dtype=float))
    R = df.get("metrics/recall(B)",    pd.Series(dtype=float))
    df["metrics/F1(B)"] = 2 * P * R / (P + R + 1e-9)
    df["metrics/F2(B)"] = 5 * P * R / (4 * P + R + 1e-9)
    for suffix in ("train", "val"):
        loss_cols = [c for c in df.columns if suffix in c and "loss" in c]
        if loss_cols:
            df[f"{suffix}/total_loss"] = df[loss_cols].sum(axis=1)
    return df

orig_df = _load_csv(PREV_RESULTS_CSV)
cont_df = _load_csv(f"{CONT_RUN_DIR}/results.csv")

# Save to Excel
for tag, df in [("original_70ep", orig_df), ("continuation", cont_df)]:
    if not df.empty:
        df.to_excel(f"{EXCEL_DIR}/training_{tag}.xlsx", index=False)

# ── A) Individual continuation run plot (6-panel metrics) ───────────────────
def plot_6panel(df: pd.DataFrame, tag: str, title: str):
    if df.empty:
        return
    ep  = df["epoch"]
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    panel_map = [
        ("metrics/precision(B)", "Precision",   axes[0, 0]),
        ("metrics/recall(B)",    "Recall",       axes[0, 1]),
        ("metrics/mAP50(B)",     "mAP@0.5",      axes[0, 2]),
        ("metrics/mAP50-95(B)",  "mAP@0.5:0.95", axes[1, 0]),
        ("metrics/F1(B)",        "F1",            axes[1, 1]),
        ("metrics/F2(B)",        "F2",            axes[1, 2]),
    ]
    for col, label, ax in panel_map:
        if col in df.columns:
            ax.plot(ep, df[col], lw=2, marker="o", markersize=3)
            best_v = df[col].max()
            best_e = df.loc[df[col].idxmax(), "epoch"]
            ax.axhline(best_v, color="red", ls=":", lw=1.2, alpha=0.7,
                       label=f"Best={best_v:.4f} @ep{best_e:.0f}")
        ax.set(xlabel="Epoch", ylabel=label, title=f"{label}")
        ax.set_ylim([0, 1.05]); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{tag}_metrics_6panel.png", dpi=300); plt.close()

plot_6panel(cont_df, "continuation", f"CBAM+P2 Continuation Run — Metrics")

# ── B) Stitched curve (only if original CSV was found) ───────────────────────
if not orig_df.empty and not cont_df.empty:
    # Offset continuation epochs to follow original
    orig_ep_max  = int(orig_df["epoch"].max())
    cont_stitched = cont_df.copy()
    cont_stitched["epoch"] = cont_stitched["epoch"] + orig_ep_max

    for metric_col, label, color_orig, color_cont in [
        ("metrics/mAP50(B)",    "mAP@0.5",   "#1E88E5", "#E53935"),
        ("metrics/recall(B)",   "Recall",     "#1E88E5", "#E53935"),
        ("metrics/F2(B)",       "F2",         "#1E88E5", "#E53935"),
        ("val/total_loss",      "Val Loss",   "#1E88E5", "#E53935"),
    ]:
        fig, ax = plt.subplots(figsize=(14, 5))
        if metric_col in orig_df.columns:
            ax.plot(orig_df["epoch"],         orig_df[metric_col],
                    color=color_orig, lw=2, label="Original (ep 1–70)")
        if metric_col in cont_stitched.columns:
            ax.plot(cont_stitched["epoch"],   cont_stitched[metric_col],
                    color=color_cont, lw=2, ls="--", label="Continuation (ep 71+)")
        ax.axvline(orig_ep_max, color="grey", ls=":", lw=1.5, label="Stage boundary")
        ax.set(xlabel="Epoch", ylabel=label,
               title=f"Full Training History — {label}")
        ax.legend(); ax.grid(True, alpha=0.3)
        if "loss" not in metric_col.lower():
            ax.set_ylim([0, 1.05])
        _safe = metric_col.replace("/", "_").replace("(B)", "")
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/stitched_{_safe}.png", dpi=300); plt.close()
    print("  ✓ Stitched training history plots saved")

# ── C) Loss plot for continuation ────────────────────────────────────────────
if not cont_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    ax = axes[0]
    for k in ["box", "cls", "dfl"]:
        tc, vc = f"train/{k}_loss", f"val/{k}_loss"
        if tc in cont_df.columns:
            ax.plot(cont_df["epoch"], cont_df[tc], label=f"Train {k}", alpha=0.8)
            ax.plot(cont_df["epoch"], cont_df[vc], label=f"Val {k}",
                    ls="--", alpha=0.8)
    ax.set(xlabel="Epoch", ylabel="Loss", title="Individual Losses — Continuation")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax = axes[1]
    if "val/total_loss" in cont_df.columns:
        ax.plot(cont_df["epoch"], cont_df["train/total_loss"],
                label="Train Total", lw=2)
        ax.plot(cont_df["epoch"], cont_df["val/total_loss"],
                label="Val Total",   lw=2, ls="--")
    ax.set(xlabel="Epoch", ylabel="Total Loss", title="Total Loss — Continuation")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/continuation_loss_curves.png", dpi=300); plt.close()

print("✓ Training curve analysis complete")


# ============================================================================
# CELL 12: Copy Ultralytics Auto-Generated Plots
# ============================================================================
# Ultralytics auto-generates PR_curve.png, F1_curve.png, confusion_matrix.png
# when plots=True. Copy them into our consolidated plots folder.
for _fname in ("PR_curve.png", "F1_curve.png", "confusion_matrix.png",
               "confusion_matrix_normalized.png", "results.png"):
    _src = Path(CONT_RUN_DIR) / _fname
    if _src.exists():
        _dst = Path(PLOT_DIR) / f"cont_{_fname}"
        shutil.copy(str(_src), str(_dst))
        print(f"  ✓ {_fname} → plots/cont_{_fname}")


# ============================================================================
# CELL 13: Model Complexity
# ============================================================================
from ultralytics import YOLO

def get_complexity(weights_path: str, label: str) -> dict:
    _m = YOLO(weights_path)
    n_p = sum(p.numel() for p in _m.model.parameters())
    try:
        from thop import profile as thop_profile
        _d   = torch.zeros(1, 3, 640, 640)
        _dev = next(_m.model.parameters()).device
        macs, _ = thop_profile(_m.model, inputs=(_d.to(_dev),), verbose=False)
        gf = macs / 1e9
    except Exception:
        gf = 0.0
    sz = os.path.getsize(weights_path) / 1024
    del _m; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return {"Model": label, "Params(M)": round(n_p/1e6, 3),
            "GFLOPs": round(gf, 1), "Size(KB)": round(sz, 0)}

complexity_rows = []
if os.path.exists(CONT_BEST):
    complexity_rows.append(get_complexity(CONT_BEST,     "CBAM+P2 (continued)"))
if os.path.exists(PREV_BEST_PT):
    complexity_rows.append(get_complexity(PREV_BEST_PT,  "CBAM+P2 (original)"))
if complexity_rows:
    cdf = pd.DataFrame(complexity_rows)
    cdf.to_excel(f"{EXCEL_DIR}/model_complexity.xlsx", index=False)
    print("\nModel Complexity:")
    print(cdf.to_string(index=False))


# ============================================================================
# CELL 14: Helper Functions
# ============================================================================
import cv2
from ultralytics.utils.metrics import box_iou

def get_image_list(img_dir: str) -> list:
    return sorted(f for f in os.listdir(img_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png")))

def parse_yolo_label(lbl_path: str, W: int, H: int) -> torch.Tensor:
    if not os.path.exists(lbl_path):
        return torch.empty((0, 4), dtype=torch.float32)
    boxes = []
    for line in open(lbl_path):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        _, xc, yc, w, h = (float(v) for v in parts[:5])
        xc *= W; yc *= H; w *= W; h *= H
        boxes.append([xc - w/2, yc - h/2, xc + w/2, yc + h/2])
    return (torch.tensor(boxes, dtype=torch.float32)
            if boxes else torch.empty((0, 4), dtype=torch.float32))

def match_preds_gt(pred_boxes, gt_boxes, iou_thr=0.5):
    if len(pred_boxes) == 0: return 0, 0, len(gt_boxes)
    if len(gt_boxes)   == 0: return 0, len(pred_boxes), 0
    ious      = box_iou(pred_boxes.float(), gt_boxes.float())
    matched   = set()
    tp        = 0
    for pi in range(len(pred_boxes)):
        max_iou, max_j = ious[pi].max(dim=0)
        if max_iou >= iou_thr and max_j.item() not in matched:
            tp += 1; matched.add(max_j.item())
    return tp, len(pred_boxes) - tp, len(gt_boxes) - tp

def categorise_boxes(boxes: torch.Tensor) -> dict:
    if len(boxes) == 0:
        return {k: torch.zeros(0, dtype=torch.bool)
                for k in ("very_tiny", "tiny", "small", "medium", "large")}
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return {
        "very_tiny": areas < 8**2,
        "tiny":      (areas >= 8**2)  & (areas < 16**2),
        "small":     (areas >= 16**2) & (areas < 32**2),
        "medium":    (areas >= 32**2) & (areas < 96**2),
        "large":      areas >= 96**2,
    }

print("✓ Helper functions loaded")


# ============================================================================
# CELL 15: Comprehensive Evaluation Function
# ============================================================================

def evaluate_model_comprehensive(model, model_name: str, img_dir: str,
                                   lbl_dir: str, split_name: str = "test",
                                   n_images=None, conf: float = 0.25) -> tuple:
    print(f"\n{'='*70}\nEVALUATING: {model_name}  [{split_name.upper()}]\n{'='*70}")
    image_list = get_image_list(img_dir)
    if n_images:
        image_list = image_list[:min(n_images, len(image_list))]

    records   = []
    inf_times = []
    all_confs = []
    total_tp = total_fp = total_fn = 0
    size_stats = {k: {"tp": 0, "fn": 0, "count": 0}
                  for k in ("very_tiny", "tiny", "small", "medium", "large")}

    for img_file in tqdm(image_list, desc=model_name, ncols=80):
        img_path = f"{img_dir}/{img_file}"
        lbl_path = f"{lbl_dir}/{img_file.rsplit('.', 1)[0]}.txt"
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        gt   = parse_yolo_label(lbl_path, W, H)

        t0   = time.perf_counter()
        pred = model.predict(img_path, conf=conf, verbose=False)
        inf_times.append((time.perf_counter() - t0) * 1000)

        pb = (pred[0].boxes.xyxy.cpu().float() if len(pred[0].boxes) > 0
              else torch.empty((0, 4), dtype=torch.float32))
        pc = (pred[0].boxes.conf.cpu().float().tolist()
              if len(pred[0].boxes) > 0 else [])
        all_confs.extend(pc)

        tp, fp, fn = match_preds_gt(pb, gt)
        total_tp += tp; total_fp += fp; total_fn += fn

        if len(gt) > 0:
            for cat, mask in categorise_boxes(gt).items():
                n_gt = int(mask.sum())
                size_stats[cat]["count"] += n_gt
                if n_gt > 0:
                    stp, _, sfn = match_preds_gt(pb, gt[mask] if mask.any()
                                                  else gt[:0])
                    size_stats[cat]["tp"] += stp
                    size_stats[cat]["fn"] += sfn

        P_i  = tp / (tp + fp + 1e-9)
        R_i  = tp / (tp + fn + 1e-9)
        F1_i = 2 * P_i * R_i / (P_i + R_i + 1e-9)
        F2_i = 5 * P_i * R_i / (4 * P_i + R_i + 1e-9)
        records.append({
            "Image": img_file, "GT": len(gt), "Pred": len(pb),
            "TP": tp, "FP": fp, "FN": fn,
            "Precision": P_i, "Recall": R_i, "F1": F1_i, "F2": F2_i,
            "Avg_Conf":      float(np.mean(pc)) if pc else 0.0,
            "Inference_ms":  inf_times[-1],
        })

    df = pd.DataFrame(records)
    df.to_excel(f"{EXCEL_DIR}/{model_name}_{split_name}_detailed.xlsx", index=False)

    P_ov = total_tp / (total_tp + total_fp + 1e-9)
    R_ov = total_tp / (total_tp + total_fn + 1e-9)
    F1   = 2 * P_ov * R_ov / (P_ov + R_ov + 1e-9)
    F2   = 5 * P_ov * R_ov / (4 * P_ov + R_ov + 1e-9)

    size_recalls = {}
    for cat, s in size_stats.items():
        tot = s["tp"] + s["fn"]
        size_recalls[f"{cat}_recall"] = s["tp"] / tot if tot > 0 else 0.0
        size_recalls[f"{cat}_count"]  = s["count"]

    summary = {
        "Model": model_name, "Split": split_name, "N_images": len(image_list),
        "Total_GT": total_tp + total_fn, "Total_Pred": total_tp + total_fp,
        "TP": total_tp, "FP": total_fp, "FN": total_fn,
        "Precision": P_ov, "Recall": R_ov, "F1": F1, "F2": F2,
        **size_recalls,
        "Avg_Inf_ms": float(np.mean(inf_times)),
        "Std_Inf_ms": float(np.std(inf_times)),
        "P95_Inf_ms": float(np.percentile(inf_times, 95)),
    }

    print(f"  P={P_ov:.4f}  R={R_ov:.4f}  F1={F1:.4f}  F2={F2:.4f}")
    for cat in ("very_tiny", "tiny", "small", "medium"):
        print(f"  {cat:12s}: {size_recalls[f'{cat}_recall']:.4f}"
              f"  (n={size_recalls[f'{cat}_count']})")
    print(f"  Latency: {summary['Avg_Inf_ms']:.1f}±{summary['Std_Inf_ms']:.1f} ms")

    return df, summary, inf_times, all_confs


# ============================================================================
# CELL 16: Visualisation Functions
# ============================================================================

def visualise_predictions(model, img_dir: str, lbl_dir: str,
                           model_name: str, split: str, n: int = 15,
                           conf: float = 0.25):
    imgs  = get_image_list(img_dir)[:n]
    cols  = 5
    rows  = (len(imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, f in enumerate(imgs):
        lbl  = f"{lbl_dir}/{f.rsplit('.', 1)[0]}.txt"
        gt   = len(open(lbl).readlines()) if os.path.exists(lbl) else 0
        p    = model.predict(f"{img_dir}/{f}", conf=conf, verbose=False)
        pc   = len(p[0].boxes)
        axes[i].imshow(cv2.cvtColor(p[0].plot(), cv2.COLOR_BGR2RGB))
        axes[i].axis("off")
        col  = "green" if pc == gt else "orange" if abs(pc - gt) <= 2 else "red"
        axes[i].set_title(f"GT:{gt}|Pred:{pc}", fontsize=10,
                           color=col, fontweight="bold")
    for j in range(len(imgs), len(axes)):
        axes[j].axis("off")
    plt.suptitle(f"{model_name} — {split}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{model_name}_{split}_predictions.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_confidence_distribution(confs: list, model_name: str, split: str):
    if not confs:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(confs, bins=50, range=(0, 1), color="#1E88E5",
            alpha=0.7, density=True, label="Confidence")
    ax.axvline(np.mean(confs),   color="red",    ls="--", lw=1.5,
               label=f"Mean={np.mean(confs):.3f}")
    ax.axvline(np.median(confs), color="orange", ls="--", lw=1.5,
               label=f"Median={np.median(confs):.3f}")
    ax.set(xlabel="Confidence", ylabel="Density",
           title=f"{model_name} — Confidence Distribution ({split})")
    ax.legend(); ax.set_xlim([0, 1]); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{model_name}_{split}_conf_dist.png", dpi=200)
    plt.close()


def plot_calibration_ece(results_df: pd.DataFrame, model_name: str,
                          split: str, n_bins: int = 10):
    df = results_df[results_df["Pred"] > 0].copy()
    if df.empty:
        return None
    bins     = np.linspace(0, 1, n_bins + 1)
    cal_rows = []
    for i in range(n_bins):
        m = (df["Avg_Conf"] >= bins[i]) & (df["Avg_Conf"] < bins[i+1])
        if m.sum() > 0:
            cal_rows.append({
                "Conf_mid": (bins[i] + bins[i+1]) / 2,
                "Avg_Conf": df.loc[m, "Avg_Conf"].mean(),
                "Avg_Prec": df.loc[m, "Precision"].mean(),
                "Count":    int(m.sum()),
            })
    if not cal_rows:
        return None
    cdf  = pd.DataFrame(cal_rows)
    ece  = float(np.average(np.abs(cdf["Avg_Conf"] - cdf["Avg_Prec"]),
                             weights=cdf["Count"]))
    cdf.to_excel(f"{EXCEL_DIR}/{model_name}_{split}_calibration.xlsx", index=False)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.bar(cdf["Conf_mid"], cdf["Avg_Prec"],
           width=1/n_bins, alpha=0.6, color="#4CAF50", label="Precision@bin")
    ax.bar(cdf["Conf_mid"], cdf["Avg_Conf"] - cdf["Avg_Prec"],
           width=1/n_bins, alpha=0.4, color="#F44336",
           bottom=cdf["Avg_Prec"], label="Calibration gap")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect")
    ax.set(xlabel="Confidence", ylabel="Precision",
           title=f"{model_name} — ECE={ece:.4f} ({split})")
    ax.legend(); ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{model_name}_{split}_calibration.png", dpi=200)
    plt.close()
    return ece


def analyze_failure_modes(results_df: pd.DataFrame, model_name: str,
                           split: str) -> dict:
    dangerous = results_df[(results_df["Avg_Conf"] > 0.7) &
                            (results_df["Recall"] < 0.5)]
    high_fn   = results_df[results_df["FN"] > 3]
    high_fp   = results_df[results_df["FP"] > 3]
    if len(dangerous) > 0:
        dangerous.to_excel(
            f"{EXCEL_DIR}/{model_name}_{split}_dangerous.xlsx", index=False)
    print(f"  Dangerous (hi-conf lo-recall): {len(dangerous)}"
          f"  | High-FN: {len(high_fn)}"
          f"  | High-FP: {len(high_fp)}")
    return {"Dangerous": len(dangerous), "High_FN": len(high_fn),
            "High_FP": len(high_fp)}


def benchmark_speed(model, sample_img: str, model_name: str) -> pd.DataFrame:
    rows = []
    for sz in [320, 480, 640, 800]:
        for _ in range(3):   # warmup
            model.predict(sample_img, imgsz=sz, verbose=False)
        times = []
        for _ in range(15):
            t0 = time.perf_counter()
            model.predict(sample_img, imgsz=sz, verbose=False)
            times.append((time.perf_counter() - t0) * 1000)
        mean_ms = float(np.mean(times[3:]))
        rows.append({"Resolution": sz,
                     "Avg_ms": round(mean_ms, 2),
                     "FPS": round(1000 / mean_ms, 1)})
    sdf = pd.DataFrame(rows)
    sdf.to_excel(f"{EXCEL_DIR}/{model_name}_speed.xlsx", index=False)
    print(sdf.to_string(index=False))
    return sdf


def plot_per_size_recall(summaries: list, split: str):
    cats   = ["very_tiny", "tiny", "small", "medium", "large"]
    labels = ["Very Tiny\n(<8²px)", "Tiny\n(8–16px)", "Small\n(16–32px)",
              "Medium\n(32–96px)", "Large\n(>96px)"]
    x      = np.arange(len(cats))
    n      = len(summaries)
    width  = 0.7 / n
    colors = ["#E53935", "#1E88E5"]

    fig, ax = plt.subplots(figsize=(16, 7))
    for si, s in enumerate(summaries):
        recalls = [s.get(f"{c}_recall", 0) for c in cats]
        bars    = ax.bar(x + (si - n/2 + 0.5) * width, recalls, width,
                         label=s["Model"], color=colors[si % len(colors)],
                         alpha=0.85)
        for bar, rc in zip(bars, recalls):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{rc:.3f}", ha="center", va="bottom", fontsize=8)

    counts = [summaries[0].get(f"{c}_count", 0) for c in cats]
    for i, c in enumerate(counts):
        ax.text(i, max(s.get(f"{cats[i]}_recall", 0) for s in summaries) + 0.05,
                f"n={c}", ha="center", fontsize=9, color="grey")

    ax.set(xlabel="Object Size", ylabel="Recall",
           title=f"Per-Size Recall — {split.upper()}")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend(fontsize=11); ax.set_ylim([0, 1.18])
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/per_size_recall_{split}.png", dpi=300)
    plt.close()
    print(f"  ✓ Per-size recall chart ({split}) saved")


print("✓ All visualisation functions loaded")


# ============================================================================
# CELL 17: Load Models for Evaluation
# ============================================================================
from ultralytics import YOLO

models_eval = {}

if os.path.exists(CONT_BEST):
    models_eval["CBAM+P2_continued"] = YOLO(CONT_BEST)
    print(f"✓ Loaded: CBAM+P2 continued  ({CONT_BEST})")
else:
    print(f"⚠  Continued model not found: {CONT_BEST}")

if os.path.exists(PREV_BEST_PT):
    models_eval["CBAM+P2_original"] = YOLO(PREV_BEST_PT)
    print(f"✓ Loaded: CBAM+P2 original   ({PREV_BEST_PT})")

assert models_eval, "No models loaded — cannot evaluate!"


# ============================================================================
# CELL 18: Official Ultralytics Val Metrics
# ============================================================================
print("\n" + "=" * 80 + "\nOFFICIAL VAL METRICS (Ultralytics)\n" + "=" * 80)

official_results = {}
for name, model in models_eval.items():
    res = model.val(data="c2a.yaml", split="test", verbose=False, plots=True)
    official_results[name] = {
        "mAP50":    round(float(res.box.map50), 4),
        "mAP50-95": round(float(res.box.map),   4),
        "Precision":round(float(res.box.mp),    4),
        "Recall":   round(float(res.box.mr),    4),
    }
    r = official_results[name]
    P, R = r["Precision"], r["Recall"]
    r["F1"] = round(2 * P * R / (P + R + 1e-9), 4)
    r["F2"] = round(5 * P * R / (4 * P + R + 1e-9), 4)
    print(f"  {name}: mAP50={r['mAP50']:.4f}  mAP50-95={r['mAP50-95']:.4f}"
          f"  F2={r['F2']:.4f}")

pd.DataFrame(official_results).T.to_excel(
    f"{EXCEL_DIR}/official_test_metrics.xlsx")
print("✓ Official metrics saved")


# ============================================================================
# CELL 19: Test Set Evaluation
# ============================================================================
print("\n" + "=" * 80)
print(f"TEST SET EVALUATION ({TEST_IMAGES or 'ALL'} images)")
print("=" * 80)

test_summaries = []
test_dfs       = {}
test_confs_all = {}

for name, model in models_eval.items():
    df, summary, _, confs = evaluate_model_comprehensive(
        model, name, TEST_IMG_DIR, TEST_LBL_DIR, "test", TEST_IMAGES)
    test_dfs[name]       = df
    test_summaries.append(summary)
    test_confs_all[name] = confs
    visualise_predictions(model, TEST_IMG_DIR, TEST_LBL_DIR,
                          name, "test", min(15, TEST_IMAGES or 15))


# ============================================================================
# CELL 20: Validation Set Evaluation
# ============================================================================
print("\n" + "=" * 80)
print(f"VALIDATION SET EVALUATION ({VAL_IMAGES or 'ALL'} images)")
print("=" * 80)

val_summaries = []
val_dfs       = {}

for name, model in models_eval.items():
    df, summary, _, confs = evaluate_model_comprehensive(
        model, name, VAL_IMG_DIR, VAL_LBL_DIR, "val", VAL_IMAGES)
    val_dfs[name]       = df
    val_summaries.append(summary)
    visualise_predictions(model, VAL_IMG_DIR, VAL_LBL_DIR,
                          name, "val", min(15, VAL_IMAGES or 15))


# ============================================================================
# CELL 21: Advanced Metrics
# ============================================================================
print("\n" + "=" * 80 + "\nADVANCED METRICS\n" + "=" * 80)

sample_img  = f"{TEST_IMG_DIR}/{get_image_list(TEST_IMG_DIR)[0]}"
ece_results = {}
speed_dfs   = {}

for name, model in models_eval.items():
    print(f"\n── {name} ──")
    print("  Calibration (ECE):")
    ece = plot_calibration_ece(test_dfs[name], name, "test")
    ece_results[name] = ece
    print(f"  ECE = {ece:.4f}" if ece else "  ECE = N/A")
    print("  Confidence distribution:")
    plot_confidence_distribution(test_confs_all.get(name, []), name, "test")
    print("  Failure modes:")
    analyze_failure_modes(test_dfs[name], name, "test")
    print("  Speed benchmark:")
    speed_dfs[name] = benchmark_speed(model, sample_img, name)

# Per-size recall charts
if test_summaries: plot_per_size_recall(test_summaries, "test")
if val_summaries:  plot_per_size_recall(val_summaries,  "val")


# ============================================================================
# CELL 22: Comparison Table (Original vs Continued)
# ============================================================================
_metric_keys = [
    ("Precision",       "Precision"),
    ("Recall",          "Recall"),
    ("F1",              "F1"),
    ("F2",              "F2"),
    ("very_tiny_recall","Very Tiny Recall (<8²px)"),
    ("tiny_recall",     "Tiny Recall (8–16px)"),
    ("small_recall",    "Small Recall (16–32px)"),
    ("medium_recall",   "Medium Recall (32–96px)"),
    ("Avg_Inf_ms",      "Avg Latency (ms)"),
]

for split_name, summaries in [("test", test_summaries), ("val", val_summaries)]:
    rows = []
    base = next((s for s in summaries if "original" in s["Model"]), summaries[-1])
    for key, label in _metric_keys:
        row = {"Metric": label}
        for s in summaries:
            row[s["Model"]] = round(float(s.get(key, 0)), 4)
        if len(summaries) > 1:
            bv = float(base.get(key, 0))
            for s in summaries:
                if s is not base:
                    row[f"Δ vs Original"] = round(float(s.get(key, 0)) - bv, 4)
        rows.append(row)
    cdf = pd.DataFrame(rows)
    cdf.to_excel(f"{EXCEL_DIR}/comparison_{split_name}.xlsx", index=False)
    print(f"\n{'='*70}\nCOMPARISON — {split_name.upper()}\n{'='*70}")
    print(cdf.to_string(index=False))


# ============================================================================
# CELL 23: F2 Early Stopping History Plot
# ============================================================================
if f2_cb.history:
    epochs_f2, f2_vals = zip(*f2_cb.history)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(epochs_f2, f2_vals, lw=2, marker="o", markersize=4,
            color="#E53935", label="F2 per epoch")
    ax.axhline(f2_cb.best_f2, color="grey", ls=":", lw=1.5,
               label=f"Best F2={f2_cb.best_f2:.4f}")
    ax.set(xlabel="Epoch", ylabel="F2 Score",
           title="F2 Early Stopping Monitor — Continuation Run")
    ax.set_ylim([0, 1.05]); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/f2_early_stop_history.png", dpi=200); plt.close()
    print("✓ F2 early stopping history plot saved")

    # Also plot gradient norms if available
    _gnorm_path = "/kaggle/working/grad_norms.csv"
    if os.path.exists(_gnorm_path):
        gdf = pd.read_csv(_gnorm_path)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(gdf["epoch"], gdf["grad_norm"], lw=1.5, color="#1E88E5")
        ax.axhline(200, color="red", ls="--", lw=1, alpha=0.6, label="Explosion threshold")
        ax.set(xlabel="Epoch", ylabel="Gradient L2 Norm",
               title="Gradient Norm per Epoch — Continuation Run")
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_yscale("log")
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/gradient_norms.png", dpi=200); plt.close()


# ============================================================================
# CELL 24: Master Report
# ============================================================================
ts = {s["Model"]: s for s in test_summaries}
vs = {s["Model"]: s for s in val_summaries}

def _f(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)

# Best epoch from continuation results.csv
best_ep = -1
best_f2_ep = 0.0
if not cont_df.empty and "metrics/F2(B)" in cont_df.columns:
    best_ep    = int(cont_df.loc[cont_df["metrics/mAP50(B)"].idxmax(), "epoch"])
    best_f2_ep = float(cont_df["metrics/F2(B)"].max())

report_lines = [
    "=" * 80,
    "CBAM+P2 HEAD — CONTINUATION RUN REPORT",
    "=" * 80,
    f"Generated  : {time.strftime('%Y-%m-%d %H:%M:%S')}",
    f"TEST MODE  : {TEST_MODE}",
    f"Stage 1    : 70 epochs (original ablation study)",
    f"Stage 2    : {CONTINUE_EPOCHS} continuation epochs (lr={FINETUNE_LR})",
    f"Best epoch : {best_ep}  |  Best F2={best_f2_ep:.4f}",
    "",
    "MODEL COMPLEXITY:",
]
if complexity_rows:
    for r in complexity_rows:
        report_lines.append(
            f"  {r['Model']:30s}: {r['Params(M)']:.3f}M params"
            f"  {r['GFLOPs']:.1f} GFLOPs")

report_lines += ["", "OFFICIAL mAP (Ultralytics, test split):"]
for name, m in official_results.items():
    report_lines.append(
        f"  {name:30s}: mAP50={m['mAP50']:.4f}  "
        f"mAP50-95={m['mAP50-95']:.4f}  F2={m['F2']:.4f}")

report_lines += ["", "TEST SET (custom eval loop):"]
for name, s in ts.items():
    report_lines.append(
        f"  {name:30s}: P={_f(s['Precision'])}  R={_f(s['Recall'])}"
        f"  F1={_f(s['F1'])}  F2={_f(s['F2'])}"
        f"  Lat={_f(s['Avg_Inf_ms'])}ms")

report_lines += ["", "PER-SIZE RECALL (TEST):"]
for cat in ("very_tiny", "tiny", "small", "medium"):
    line = f"  {cat:12s}:"
    for name, s in ts.items():
        line += f"  {name.split('_')[-1]}={_f(s[f'{cat}_recall'])}"
    report_lines.append(line)

report_lines += ["", "CALIBRATION (ECE):"]
for name, ece in ece_results.items():
    report_lines.append(f"  {name:30s}: ECE={_f(ece) if ece else 'N/A'}")

report_lines += ["", "EARLY STOPPING:"]
report_lines.append(f"  NaN stop triggered : {nan_cb.triggered}")
report_lines.append(f"  F2 best            : {f2_cb.best_f2:.4f}")
report_lines.append(f"  F2 patience used   : {f2_cb.counter}/{f2_cb.patience}")

report_lines += ["", "=" * 80]

report_text = "\n".join(report_lines)
print("\n" + report_text)
with open(f"{REPORT_DIR}/MASTER_REPORT_CBAM_P2_CONTINUED.txt", "w") as f:
    f.write(report_text)

with open(f"{REPORT_DIR}/test_summaries.json", "w") as f:
    json.dump(test_summaries, f, indent=2, default=str)
with open(f"{REPORT_DIR}/val_summaries.json", "w") as f:
    json.dump(val_summaries,  f, indent=2, default=str)

print(f"\n✓ Report → {REPORT_DIR}/MASTER_REPORT_CBAM_P2_CONTINUED.txt")


# ============================================================================
# CELL 25: Package Results
# ============================================================================
import subprocess as _sp

_zip_cmd = (
    f"zip -r /kaggle/working/cbam_p2_continued_results.zip "
    f"{EXCEL_DIR} {PLOT_DIR} {REPORT_DIR} "
    f"/kaggle/working/session_last.pt "
    f"/kaggle/working/session_best.pt "
    f"/kaggle/working/session_meta.json "
    f"/kaggle/working/grad_norms.csv 2>/dev/null || true"
)
_sp.run(_zip_cmd, shell=True)
print("✓ Packaged → /kaggle/working/cbam_p2_continued_results.zip")

try:
    from IPython.display import FileLink, display
    display(FileLink("/kaggle/working/cbam_p2_continued_results.zip"))
except ImportError:
    pass

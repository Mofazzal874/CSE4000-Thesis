"""
================================================================================
DISASTER HUMAN DETECTION — SAHI + TTA EVALUATION ON ORIGINAL NO-AUG MODEL
================================================================================
PURPOSE:
  Evaluate the ORIGINAL Mamba+CBAM+P2Head model (trained WITHOUT copy-paste
  augmentation) using inference-time enhancement techniques:
    1. SAHI (Slicing Aided Hyper Inference) — tiles images, detects per tile, merges
    2. TTA  (Test-Time Augmentation) — multi-scale + flip inference
    3. SAHI + TTA combined — patches SAHI to use TTA per slice

  NO TRAINING. This is a pure evaluation script.

WHY THESE TECHNIQUES WORK ON C2A:
  C2A dataset characteristics (Nihal et al., ICPR 2024):
    - 10,215 images, 360K+ annotations, 1 class (person)
    - 47% of objects are sub-10px, 52% are 10-50px, only 1% > 50px
    - Median image width ~428px, trained at 640x640
    - 20-40 objects/image, up to 100 instances per frame

  At 640px inference, sub-10px humans are barely 2-3 feature pixels at P2 stride=4.
  SAHI re-tiles images so these tiny humans get processed at higher effective
  resolution. TTA adds multi-scale passes catching objects missed at one scale.

SAHI CONFIGURATION RATIONALE (based on Akyon et al., ICIP 2022 + VisDrone studies):
  - Slice at 0.5x training resolution (320x320) for sub-10px objects
  - Also test 256x256 (more aggressive) and 512x512 (previous run's best)
  - GREEDYNMM postprocessing (merges overlapping detections, better than NMS for
    dense aerial scenes — NMS discards valid detections in crowded regions)
  - IOS metric (Intersection over Smaller) — handles size-varying duplicates
    from adjacent slices better than IoU
  - perform_standard_pred=True — also runs full-image inference and merges,
    preserving medium/large object detection
  - Lower confidence threshold (0.15-0.20) at slice level — small objects produce
    lower confidence scores, let GREEDYNMM handle the false positives

TTA CONFIGURATION RATIONALE:
  - Ultralytics TTA uses scales [1.0, 0.83, 0.67] with [None, lr-flip, None]
  - imgsz=1280 recommended when trained at 640 (VisDrone benchmarks show
    +9 mAP@50 going from 640 to 1280 — the 0.67x scale still processes at 857px)
  - Also test imgsz=1920 for maximum small-object resolution

WHAT TO UPLOAD TO KAGGLE:
  1. C2A dataset: rgbnihal/c2a-dataset (add as input)
  2. Previous no-aug best.pt: upload as Kaggle dataset
  3. yolo11m.pt: ImageNet pretrained weights (needed for YAML model init)
  4. This script: paste as notebook code
  Set GPU to T4 x2, Internet ON (for pip installs).
  TEST_MODE=True first to smoke-test (~5 minutes), then TEST_MODE=False.

EXPECTED RUNTIME:
  TEST_MODE=True:  ~5-10 minutes
  TEST_MODE=False: ~2-3 hours (SAHI is slow — processes each image tile by tile)

================================================================================
"""

# ============================================================================
# CELL 1: Control Flags & Dependencies
# ============================================================================

TEST_MODE = False   # True = 10 images only | False = full test+val sets

import subprocess, sys, re

def pip_install(pkg, extra=""):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-U", pkg] +
        (extra.split() if extra else [])
    )

# ── CUDA / PyTorch version alignment check ────────────────────────────────
print("Checking CUDA / PyTorch version alignment ...")
try:
    _smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
except FileNotFoundError:
    _smi = ""
_driver_cuda = re.search(r"CUDA Version:\s*([\d.]+)", _smi)
_driver_cuda = _driver_cuda.group(1) if _driver_cuda else "unknown"
_torch_ver   = subprocess.run(
    [sys.executable, "-c", "import torch; print(torch.__version__)"],
    capture_output=True, text=True
).stdout.strip()
print(f"  Driver CUDA : {_driver_cuda}")
print(f"  PyTorch ver : {_torch_ver}")

_needs_upgrade = _driver_cuda.startswith("12.6") and "cu124" in _torch_ver
if _needs_upgrade:
    print("  -> Reinstalling PyTorch cu126 ...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu126"
    ])
    print("  Done.")
else:
    print(f"  OK — proceeding with {_torch_ver}")

# ── Package installs ─────────────────────────────────────────────────────
pip_install("ultralytics")
pip_install("sahi")
pip_install("timm")
pip_install("thop")
pip_install("openpyxl")
pip_install("scikit-learn")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "pandas<3.0", "matplotlib<3.10", "tqdm"])
print("All dependencies installed")


# ============================================================================
# CELL 2: Imports & Configuration
# ============================================================================
import os, sys, time, yaml, shutil, gc, math, copy, warnings, json
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

# ── GPU info ──────────────────────────────────────────────────────────────
num_gpus = torch.cuda.device_count()
gpu_name = torch.cuda.get_device_name(0) if num_gpus > 0 else "CPU"
gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3 if num_gpus > 0 else 0
DEVICE   = "cuda:0" if num_gpus > 0 else "cpu"
print(f"GPU(s): {num_gpus}  |  GPU 0: {gpu_name} ({gpu_mem:.1f} GB)")

# ── Test mode config ──────────────────────────────────────────────────────
if TEST_MODE:
    TEST_IMAGES = 10
    VAL_IMAGES  = 10
    print("TEST MODE — 10 images only per split")
else:
    TEST_IMAGES = None   # full set
    VAL_IMAGES  = None
    print("FULL MODE — all images")

# ── Output directories ───────────────────────────────────────────────────
EXCEL_DIR  = "/kaggle/working/excel_reports"
PLOT_DIR   = "/kaggle/working/plots"
REPORT_DIR = "/kaggle/working/benchmark_reports"
for d in [EXCEL_DIR, PLOT_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)


# ============================================================================
# CELL 3: Dataset Configuration
# ============================================================================
print("\nSearching for C2A dataset in /kaggle/input/ ...")
DATASET_ROOT = None
for _root, _dirs, _files in os.walk("/kaggle/input"):
    if (os.path.isdir(os.path.join(_root, "train", "images")) and
        os.path.isdir(os.path.join(_root, "val",   "images"))):
        DATASET_ROOT = _root
        print(f"  Found: {DATASET_ROOT}")
        break

if DATASET_ROOT is None:
    print("  C2A dataset NOT found. Contents of /kaggle/input/:")
    for _root, _dirs, _files in os.walk("/kaggle/input"):
        _lvl = _root.replace("/kaggle/input", "").count(os.sep)
        if _lvl < 4:
            print("  " * _lvl + f"  {os.path.basename(_root)}/")
            for _f in _files[:3]:
                print("  " * _lvl + f"    {_f}")
    raise FileNotFoundError(
        "C2A dataset not found under /kaggle/input/.\n"
        "Make sure you added the 'C2A Dataset' as input to this notebook."
    )

TEST_IMG_DIR = f"{DATASET_ROOT}/test/images"
TEST_LBL_DIR = f"{DATASET_ROOT}/test/labels"
VAL_IMG_DIR  = f"{DATASET_ROOT}/val/images"
VAL_LBL_DIR  = f"{DATASET_ROOT}/val/labels"

# Write c2a.yaml for ultralytics val()
dataset_yaml_content = f"""
train: {DATASET_ROOT}/train/images
val:   {DATASET_ROOT}/val/images
test:  {DATASET_ROOT}/test/images
nc: 1
names: ['person']
"""
with open("c2a.yaml", "w") as f:
    f.write(dataset_yaml_content)
print(f"c2a.yaml written")

# Count images
n_test = len([f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.jpg','.jpeg','.png'))])
n_val  = len([f for f in os.listdir(VAL_IMG_DIR) if f.lower().endswith(('.jpg','.jpeg','.png'))])
print(f"  Test: {n_test} images  |  Val: {n_val} images")


# ============================================================================
# CELL 4: CBAM Module (must match saved model weights exactly)
# ============================================================================
cbam_code = '''
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        r = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, r, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(r, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        a = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        m = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return x * self.sigmoid(a + m)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], 1)))

class CBAM(nn.Module):
    """CBAM with lazy initialisation — auto-detects channels at first forward."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.reduction   = 16
        self.kernel_size = 7
        if len(args) == 1 and isinstance(args[0], int) and args[0] <= 32:
            self.reduction = args[0]
        elif len(args) == 2:
            self.reduction   = args[0] if isinstance(args[0], int) else 16
            self.kernel_size = args[1] if isinstance(args[1], int) and args[1] in (3,7) else 7
        elif len(args) >= 4:
            self.reduction   = args[2] if isinstance(args[2], int) else 16
            self.kernel_size = args[3] if isinstance(args[3], int) and args[3] in (3,7) else 7
        self.reduction   = kwargs.get("reduction",   self.reduction)
        self.kernel_size = kwargs.get("kernel_size", self.kernel_size)
        if self.kernel_size not in (3, 7):
            self.kernel_size = 7
        self._initialized       = False
        self.channel_attention  = None
        self.spatial_attention  = None
    def _lazy_init(self, c, device, dtype):
        self.channel_attention = ChannelAttention(c, self.reduction).to(device=device, dtype=dtype)
        self.spatial_attention = SpatialAttention(self.kernel_size).to(device=device, dtype=dtype)
        self._initialized = True
    def forward(self, x):
        if not self._initialized:
            self._lazy_init(x.size(1), x.device, x.dtype)
        return self.spatial_attention(self.channel_attention(x))
'''
with open("/kaggle/working/cbam_module.py", "w") as f:
    f.write(cbam_code)
exec(cbam_code)
print("CBAM module OK")


# ============================================================================
# CELL 5: Mamba SSM Modules (must match saved model architecture exactly)
# ============================================================================

def _get_window_size(channels: int) -> int:
    if channels >= 512: return 4
    if channels >= 256: return 6
    return 8

class _SelectiveScan1D(nn.Module):
    def __init__(self, d_model: int, d_state: int = 4, dt_rank_ratio: int = 16):
        super().__init__()
        D, N = d_model, d_state
        dt_rank = max(D // dt_rank_ratio, 1)
        self.D, self.N, self.dt_rank = D, N, dt_rank
        self.conv1d = nn.Conv1d(D, D, kernel_size=4, padding=3, groups=D, bias=True)
        self.x_proj  = nn.Linear(D, dt_rank + 2 * N, bias=False)
        self.dt_proj = nn.Linear(dt_rank, D, bias=True)
        A = torch.arange(1, N + 1, dtype=torch.float32).unsqueeze(0).repeat(D, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D_skip = nn.Parameter(torch.ones(D))
        dt_init = torch.exp(
            torch.rand(D) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        B_win, L, D = u.shape
        in_dtype = u.dtype
        u_conv = self.conv1d(u.transpose(1, 2))[:, :, :L].transpose(1, 2)
        u_act  = F.silu(u_conv)
        xBC_dt = self.x_proj(u_act)
        dt_raw, B_param, C_param = xBC_dt.split([self.dt_rank, self.N, self.N], dim=-1)
        dt      = F.softplus(self.dt_proj(dt_raw)).float()
        B_param = B_param.float()
        C_param = C_param.float()
        u_f     = u_act.float()
        A       = -torch.exp(self.A_log.float())
        deltaA   = torch.exp(torch.einsum("bld,dn->bldn", dt, A))
        deltaB_u = torch.einsum("bld,bln,bld->bldn", dt, B_param, u_f)
        x = torch.zeros(B_win, D, self.N, device=u.device, dtype=torch.float32)
        ys = []
        for i in range(L):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y_i = (x * C_param[:, i, :].unsqueeze(1)).sum(-1)
            ys.append(y_i)
        y = torch.stack(ys, dim=1).to(in_dtype)
        y = y + u_act * self.D_skip.to(in_dtype)
        return y


class LocalWindowSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 4, window_size: int = 8):
        super().__init__()
        self.ws = window_size
        D = d_model
        self.norm    = nn.LayerNorm(D)
        self.in_proj = nn.Linear(D, D * 2, bias=False)
        self.scan_fwd = _SelectiveScan1D(D, d_state)
        self.scan_bwd = _SelectiveScan1D(D, d_state)
        self.out_proj = nn.Linear(D, D, bias=False)
        self.out_norm = nn.LayerNorm(D)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def _partition(self, x):
        B, C, H, W = x.shape
        ws = self.ws
        ph = (ws - H % ws) % ws
        pw = (ws - W % ws) % ws
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        _, _, Hp, Wp = x.shape
        nH, nW = Hp // ws, Wp // ws
        x = x.reshape(B, C, nH, ws, nW, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B * nH * nW, ws * ws, C)
        return x, (B, C, H, W, Hp, Wp, nH, nW)

    def _reverse(self, y, meta):
        B, C, H, W, Hp, Wp, nH, nW = meta
        ws = self.ws
        y = y.reshape(B, nH, nW, ws, ws, C)
        y = y.permute(0, 5, 1, 3, 2, 4).reshape(B, C, Hp, Wp)
        return y[:, :, :H, :W].contiguous()

    def forward(self, x):
        windows, meta = self._partition(x)
        residual = windows
        windows_n = self.norm(windows)
        xz = self.in_proj(windows_n)
        x_in, z = xz.chunk(2, dim=-1)
        y_fwd = self.scan_fwd(x_in)
        y_bwd = self.scan_bwd(x_in.flip(1)).flip(1)
        y     = (y_fwd + y_bwd) * F.silu(z)
        y = self.out_norm(self.out_proj(y) + residual)
        return self._reverse(y, meta)


class _MambaBottleneck(nn.Module):
    def __init__(self, c, shortcut, d_state, window_size):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.cv1 = Conv(c, c, 3, 1)
        self.ssm = LocalWindowSSM(c, d_state=d_state, window_size=window_size)
        self.cv2 = Conv(c, c, 3, 1)
        self.add = shortcut

    def forward(self, x):
        y = self.cv2(self.ssm(self.cv1(x)))
        return x + y if self.add else y


class C3K2Mamba(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, d_state=4):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        ws     = _get_window_size(self.c)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m   = nn.ModuleList(
            _MambaBottleneck(self.c, shortcut and c1 == c2, d_state, ws)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

print("Mamba modules defined")


# ============================================================================
# CELL 6: Register CBAM in Ultralytics Namespace
# ============================================================================
# CRITICAL: Must be done BEFORE loading any model with CBAM layers.
# SAHI internally calls YOLO(model_path) which needs CBAM importable.

import site as _site
import shutil

_cbam_src = "/kaggle/working/cbam_module.py"
_installed = False
for _sp in _site.getsitepackages():
    try:
        shutil.copy(_cbam_src, os.path.join(_sp, "cbam_module.py"))
        print(f"  cbam_module.py -> site-packages: {_sp}")
        _installed = True
        break
    except Exception:
        continue
if not _installed:
    print("  Could not install cbam_module to site-packages")

if "/kaggle/working" not in sys.path:
    sys.path.insert(0, "/kaggle/working")

_existing_pypath = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = "/kaggle/working" + (
    ":" + _existing_pypath if _existing_pypath else "")

# Patch ultralytics/nn/tasks.py to import CBAM
import ultralytics.nn.tasks as _ult_tasks_mod
_tasks_file = _ult_tasks_mod.__file__
_inject     = "from cbam_module import CBAM, ChannelAttention, SpatialAttention\n"

with open(_tasks_file, "r") as _f:
    _tasks_src = _f.read()
if "from cbam_module import" not in _tasks_src:
    _tasks_src = _inject + _tasks_src
    with open(_tasks_file, "w") as _f:
        _f.write(_tasks_src)
    print("  Patched ultralytics/nn/tasks.py")
else:
    print("  ultralytics/nn/tasks.py already patched")

# Patch ultralytics/nn/modules/__init__.py
import ultralytics.nn.modules as _ult_modules_mod
_modules_init = os.path.join(os.path.dirname(_ult_modules_mod.__file__), "__init__.py")
with open(_modules_init, "r") as _f:
    _modules_src = _f.read()
if "from cbam_module import" not in _modules_src:
    _modules_src = _inject + _modules_src
    with open(_modules_init, "w") as _f:
        _f.write(_modules_src)
    print("  Patched ultralytics/nn/modules/__init__.py")
else:
    print("  ultralytics/nn/modules/__init__.py already patched")

# Reload and register
import importlib
importlib.reload(_ult_tasks_mod)
importlib.reload(_ult_modules_mod)

from cbam_module import CBAM, ChannelAttention, SpatialAttention
import ultralytics.nn.modules as ult_modules
import ultralytics.nn.tasks  as ult_tasks

for name, obj in [("CBAM", CBAM),
                   ("ChannelAttention", ChannelAttention),
                   ("SpatialAttention", SpatialAttention)]:
    setattr(ult_modules, name, obj)
    setattr(ult_tasks,   name, obj)

# Also register C3K2Mamba so torch.load can deserialise the saved model
# The best.pt was saved with C3K2Mamba layers in the state dict
setattr(ult_modules, "C3K2Mamba", C3K2Mamba)
setattr(ult_tasks,   "C3K2Mamba", C3K2Mamba)
setattr(ult_modules, "LocalWindowSSM", LocalWindowSSM)
setattr(ult_tasks,   "LocalWindowSSM", LocalWindowSSM)
setattr(ult_modules, "_MambaBottleneck", _MambaBottleneck)
setattr(ult_tasks,   "_MambaBottleneck", _MambaBottleneck)
setattr(ult_modules, "_SelectiveScan1D", _SelectiveScan1D)
setattr(ult_tasks,   "_SelectiveScan1D", _SelectiveScan1D)

assert hasattr(ult_modules, "CBAM"), "CBAM registration failed"
assert hasattr(ult_modules, "C3K2Mamba"), "C3K2Mamba registration failed"
print("CBAM + C3K2Mamba registered in ultralytics namespace")


# ============================================================================
# CELL 7: Locate Model Weights
# ============================================================================
# The no-aug best.pt should be uploaded as a Kaggle dataset input.
# Auto-discover it.

print("\nSearching for no-aug best.pt in /kaggle/input/ ...")
BEST_PT = None
for _root, _dirs, _files in os.walk("/kaggle/input"):
    for _f in _files:
        if _f == "best.pt":
            candidate = os.path.join(_root, _f)
            BEST_PT = candidate
            print(f"  Found: {BEST_PT}")
            break
    if BEST_PT:
        break

if BEST_PT is None:
    raise FileNotFoundError(
        "best.pt not found in /kaggle/input/.\n"
        "Upload your no-aug Mamba+CBAM+P2 best.pt as a Kaggle dataset."
    )

# Verify we can load it
from ultralytics import YOLO
print("\nLoading model for verification ...")
_test_model = YOLO(BEST_PT)
_n_params = sum(p.numel() for p in _test_model.model.parameters())
print(f"  Model loaded: {_n_params/1e6:.2f}M parameters")

# Check that C3K2Mamba layers exist (confirms Mamba injection was saved)
_mamba_layers = [i for i, m in enumerate(_test_model.model.model)
                 if type(m).__name__ == "C3K2Mamba"]
print(f"  C3K2Mamba layers: {_mamba_layers}")
if not _mamba_layers:
    print("  WARNING: No C3K2Mamba layers found — this may be a non-Mamba model")
del _test_model; gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
print("Model verification OK")


# ============================================================================
# CELL 8: Helper Functions
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
    return (torch.tensor(boxes, dtype=torch.float32) if boxes
            else torch.empty((0, 4), dtype=torch.float32))

def match_preds_gt(pred_boxes, gt_boxes, iou_thr=0.5):
    """Returns tp, fp, fn counts."""
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    ious = box_iou(pred_boxes.float(), gt_boxes.float())
    matched_gt = set()
    tp = 0
    for pi in range(len(pred_boxes)):
        max_iou, max_j = ious[pi].max(dim=0)
        if max_iou >= iou_thr and max_j.item() not in matched_gt:
            tp += 1
            matched_gt.add(max_j.item())
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

def compute_metrics(total_tp, total_fp, total_fn, size_stats, inf_times):
    """Compute P, R, F1, F2, per-size recall from accumulators."""
    P = total_tp / (total_tp + total_fp + 1e-9)
    R = total_tp / (total_tp + total_fn + 1e-9)
    F1 = 2 * P * R / (P + R + 1e-9)
    F2 = 5 * P * R / (4 * P + R + 1e-9)
    size_recalls = {}
    for cat, s in size_stats.items():
        tot = s["tp"] + s["fn"]
        size_recalls[f"{cat}_recall"] = s["tp"] / tot if tot > 0 else 0.0
        size_recalls[f"{cat}_count"]  = s["count"]
    return {
        "Precision": round(P, 4), "Recall": round(R, 4),
        "F1": round(F1, 4), "F2": round(F2, 4),
        "Avg_Inf_ms": round(float(np.mean(inf_times)), 1) if inf_times else 0,
        **{k: round(v, 4) if "recall" in k else v for k, v in size_recalls.items()},
    }

print("Helper functions loaded")


# ============================================================================
# CELL 9: Baseline Evaluation (standard inference, no SAHI, no TTA)
# ============================================================================
print("\n" + "=" * 80)
print("BASELINE EVALUATION — Standard Inference (640px, no SAHI, no TTA)")
print("=" * 80)

from ultralytics import YOLO

# ── Standard ultralytics val() for official mAP numbers ──────────────────
baseline_model = YOLO(BEST_PT)
print("\nRunning ultralytics val() on test split ...")
res_baseline = baseline_model.val(data="c2a.yaml", split="test",
                                    imgsz=640, augment=False,
                                    batch=8, device=DEVICE, verbose=False)
baseline_official = {
    "mAP50":     round(float(res_baseline.box.map50), 4),
    "mAP50-95":  round(float(res_baseline.box.map),   4),
    "Precision":  round(float(res_baseline.box.mp),    4),
    "Recall":     round(float(res_baseline.box.mr),    4),
}
print(f"  mAP50={baseline_official['mAP50']:.4f}  "
      f"mAP50-95={baseline_official['mAP50-95']:.4f}  "
      f"P={baseline_official['Precision']:.4f}  "
      f"R={baseline_official['Recall']:.4f}")

# ── Custom per-size evaluation ────────────────────────────────────────────
print("\nRunning custom per-size evaluation on test split ...")
test_images = get_image_list(TEST_IMG_DIR)
if TEST_MODE and TEST_IMAGES:
    test_images = test_images[:TEST_IMAGES]

total_tp = total_fp = total_fn = 0
size_stats = {k: {"tp": 0, "fn": 0, "count": 0}
              for k in ("very_tiny", "tiny", "small", "medium", "large")}
inf_times = []
all_confs = []

for img_file in tqdm(test_images, desc="Baseline", ncols=80):
    img_path = f"{TEST_IMG_DIR}/{img_file}"
    lbl_path = f"{TEST_LBL_DIR}/{img_file.rsplit('.', 1)[0]}.txt"
    img = cv2.imread(img_path)
    if img is None:
        continue
    H, W = img.shape[:2]
    gt_boxes = parse_yolo_label(lbl_path, W, H)

    t0 = time.perf_counter()
    pred = baseline_model.predict(img_path, conf=0.25, verbose=False)
    inf_times.append((time.perf_counter() - t0) * 1000)

    pboxes = (pred[0].boxes.xyxy.cpu().float()
              if len(pred[0].boxes) > 0
              else torch.empty((0, 4), dtype=torch.float32))
    pconfs = (pred[0].boxes.conf.cpu().float().tolist()
              if len(pred[0].boxes) > 0 else [])
    all_confs.extend(pconfs)

    tp, fp, fn = match_preds_gt(pboxes, gt_boxes)
    total_tp += tp; total_fp += fp; total_fn += fn

    if len(gt_boxes) > 0:
        for cat, mask in categorise_boxes(gt_boxes).items():
            n_gt = int(mask.sum())
            size_stats[cat]["count"] += n_gt
            if n_gt > 0:
                stp, _, sfn = match_preds_gt(
                    pboxes, gt_boxes[mask] if mask.any() else gt_boxes[:0])
                size_stats[cat]["tp"] += stp
                size_stats[cat]["fn"] += sfn

baseline_custom = compute_metrics(total_tp, total_fp, total_fn, size_stats, inf_times)
baseline_custom["Config"] = "Baseline (640, no SAHI/TTA)"

print(f"  P={baseline_custom['Precision']:.4f}  R={baseline_custom['Recall']:.4f}  "
      f"F1={baseline_custom['F1']:.4f}  F2={baseline_custom['F2']:.4f}")
for cat in ("very_tiny", "tiny", "small", "medium"):
    print(f"  {cat:12s}: recall={baseline_custom[f'{cat}_recall']:.4f}  "
          f"(n={baseline_custom[f'{cat}_count']})")
print(f"  Latency: {baseline_custom['Avg_Inf_ms']:.1f}ms")

del baseline_model; gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ============================================================================
# CELL 10: SAHI Evaluation — Research-Optimised Configuration Sweep
# ============================================================================
#
# KEY IMPROVEMENTS over previous run:
#   1. Added 256x256 and 320x320 slices (0.4x and 0.5x training res)
#      — literature shows these catch sub-10px objects better
#   2. Using GREEDYNMM (default) instead of NMS
#      — merges overlapping detections from adjacent slices instead of
#        discarding them. Critical for dense C2A scenes (20-40 objects/image)
#   3. Using IOS metric (default) instead of IoU
#      — handles size-varying duplicates across slices
#   4. perform_standard_pred=True
#      — runs full-image inference too, preserving medium/large recall
#      — the previous run's -13.3% medium recall drop was partly because
#        it ONLY ran sliced inference, fragmenting larger objects
#   5. Lower confidence threshold (0.15)
#      — sub-10px humans produce lower confidence; let GREEDYNMM filter FPs
#   6. postprocess_match_threshold=0.5 (standard)
#
# ============================================================================

print("\n" + "=" * 80)
print("SAHI EVALUATION — Optimised Slicing Aided Hyper Inference")
print("=" * 80)

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ── Load model for SAHI ──────────────────────────────────────────────────
sahi_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=BEST_PT,
    confidence_threshold=0.15,   # lower than standard 0.25 — catches tiny objects
    device=DEVICE,
)
print(f"SAHI model loaded: {BEST_PT}")

# ── SAHI Configuration Sweep ─────────────────────────────────────────────
# Rationale for each config:
#   256x256: Most aggressive tiling. Sub-10px objects become ~25px in slice
#            context (assuming 640 training). Maximum recall for very_tiny,
#            but highest FP risk and slowest. Based on VisDrone literature.
#   320x320: 0.5x training resolution — recommended by Akyon et al. for
#            objects at 1-5% of image area. Good balance.
#   512x512: Previous run's best. Less aggressive, faster, good baseline.
#   640x640: Same as training res. Tests whether GREEDYNMM alone helps
#            (it does for dense scenes even without size benefit).

sahi_configs = [
    {"name": "slice256_ov30",  "slice_h": 256,  "slice_w": 256,
     "overlap": 0.30, "conf": 0.15},
    {"name": "slice320_ov25",  "slice_h": 320,  "slice_w": 320,
     "overlap": 0.25, "conf": 0.15},
    {"name": "slice512_ov25",  "slice_h": 512,  "slice_w": 512,
     "overlap": 0.25, "conf": 0.20},
    {"name": "slice640_ov30",  "slice_h": 640,  "slice_w": 640,
     "overlap": 0.30, "conf": 0.20},
]

sahi_results = []
test_images_sahi = get_image_list(TEST_IMG_DIR)
if TEST_MODE and TEST_IMAGES:
    test_images_sahi = test_images_sahi[:TEST_IMAGES]

for cfg in sahi_configs:
    print(f"\n{'─'*60}")
    print(f"SAHI config: {cfg['name']}  (slice={cfg['slice_h']}x{cfg['slice_w']}, "
          f"overlap={cfg['overlap']}, conf={cfg['conf']})")
    print(f"{'─'*60}")

    # Update confidence threshold for this config
    sahi_model.confidence_threshold = cfg["conf"]

    total_tp = total_fp = total_fn = 0
    size_stats = {k: {"tp": 0, "fn": 0, "count": 0}
                  for k in ("very_tiny", "tiny", "small", "medium", "large")}
    inf_times = []

    for img_file in tqdm(test_images_sahi, desc=cfg["name"], ncols=80):
        img_path = f"{TEST_IMG_DIR}/{img_file}"
        lbl_path = f"{TEST_LBL_DIR}/{img_file.rsplit('.', 1)[0]}.txt"
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        gt_boxes = parse_yolo_label(lbl_path, W, H)

        t0 = time.perf_counter()
        result = get_sliced_prediction(
            img_path,
            sahi_model,
            slice_height=cfg["slice_h"],
            slice_width=cfg["slice_w"],
            overlap_height_ratio=cfg["overlap"],
            overlap_width_ratio=cfg["overlap"],
            perform_standard_pred=True,           # KEY: also run full-image
            postprocess_type="GREEDYNMM",         # KEY: merge, don't discard
            postprocess_match_metric="IOS",        # KEY: size-aware matching
            postprocess_match_threshold=0.5,
            verbose=0,
        )
        inf_times.append((time.perf_counter() - t0) * 1000)

        # Extract SAHI predictions
        pboxes = []
        for pred_obj in result.object_prediction_list:
            bb = pred_obj.bbox
            pboxes.append([bb.minx, bb.miny, bb.maxx, bb.maxy])
        pboxes = (torch.tensor(pboxes, dtype=torch.float32)
                  if pboxes else torch.empty((0, 4), dtype=torch.float32))

        tp, fp, fn = match_preds_gt(pboxes, gt_boxes)
        total_tp += tp; total_fp += fp; total_fn += fn

        if len(gt_boxes) > 0:
            for cat, mask in categorise_boxes(gt_boxes).items():
                n_gt = int(mask.sum())
                size_stats[cat]["count"] += n_gt
                if n_gt > 0:
                    stp, _, sfn = match_preds_gt(
                        pboxes, gt_boxes[mask] if mask.any() else gt_boxes[:0])
                    size_stats[cat]["tp"] += stp
                    size_stats[cat]["fn"] += sfn

    row = compute_metrics(total_tp, total_fp, total_fn, size_stats, inf_times)
    row["Config"] = cfg["name"]
    row["Slice"]  = f"{cfg['slice_h']}x{cfg['slice_w']}"
    row["Overlap"] = cfg["overlap"]
    row["Conf_Threshold"] = cfg["conf"]
    sahi_results.append(row)

    print(f"  P={row['Precision']:.4f}  R={row['Recall']:.4f}  "
          f"F1={row['F1']:.4f}  F2={row['F2']:.4f}")
    for cat in ("very_tiny", "tiny", "small", "medium"):
        print(f"  {cat:12s}: recall={row[f'{cat}_recall']:.4f}")
    print(f"  Latency: {row['Avg_Inf_ms']:.0f}ms")

# Save SAHI sweep results
sahi_df = pd.DataFrame(sahi_results)
sahi_df.to_excel(f"{EXCEL_DIR}/sahi_sweep_noaug.xlsx", index=False)
print(f"\nSAHI sweep saved -> {EXCEL_DIR}/sahi_sweep_noaug.xlsx")

# Identify best SAHI config by very-tiny recall
best_sahi = max(sahi_results, key=lambda x: x["very_tiny_recall"])
print(f"\n{'='*70}")
print(f"BEST SAHI CONFIG: {best_sahi['Config']}")
print(f"  Very-tiny recall: {best_sahi['very_tiny_recall']:.4f}  "
      f"(baseline: {baseline_custom['very_tiny_recall']:.4f}  "
      f"delta: +{best_sahi['very_tiny_recall'] - baseline_custom['very_tiny_recall']:.4f})")
print(f"  P={best_sahi['Precision']:.4f}  R={best_sahi['Recall']:.4f}  "
      f"F1={best_sahi['F1']:.4f}  F2={best_sahi['F2']:.4f}")
print(f"  Latency: {best_sahi['Avg_Inf_ms']:.0f}ms")
print(f"{'='*70}")

# Save best config
with open(f"{REPORT_DIR}/best_sahi_config_noaug.json", "w") as f:
    json.dump(best_sahi, f, indent=2)

del sahi_model; gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ============================================================================
# CELL 11: TTA Evaluation — Multi-Scale Test-Time Augmentation
# ============================================================================
#
# Ultralytics TTA: model.val(augment=True)
# Internally runs 3 passes: [1.0x, 0.83x, 0.67x] with [none, lr-flip, none]
#
# Why imgsz=1280:
#   - VisDrone benchmarks: +9 mAP@50 going from 640 to 1280
#   - At 1280, the 0.67x TTA scale still processes at 857px (> 640 training)
#   - Sub-10px C2A objects become ~20px at 1280, much more detectable
#
# Memory: T4 has 16GB VRAM. imgsz=1280 with batch=1 uses ~6-8GB.
#         imgsz=1920 with batch=1 uses ~12-14GB — tight but feasible.
#
# ============================================================================

print("\n" + "=" * 80)
print("TTA EVALUATION — Test-Time Augmentation")
print("=" * 80)

tta_model = YOLO(BEST_PT)
tta_results = {}

# ── Config 1: Standard (no TTA, 640) — already done, but via val() for mAP ──
print("\nStandard (no TTA, 640) — reference")
tta_results["No TTA (640)"] = baseline_official.copy()
print(f"  mAP50={baseline_official['mAP50']:.4f}  "
      f"mAP50-95={baseline_official['mAP50-95']:.4f}")

# ── Config 2: TTA at 832 ─────────────────────────────────────────────────
print("\nTTA (imgsz=832, augment=True)")
res_tta_832 = tta_model.val(data="c2a.yaml", split="test",
                             imgsz=832, augment=True,
                             batch=4, device=DEVICE, verbose=False)
tta_results["TTA (832)"] = {
    "mAP50":     round(float(res_tta_832.box.map50), 4),
    "mAP50-95":  round(float(res_tta_832.box.map),   4),
    "Precision":  round(float(res_tta_832.box.mp),    4),
    "Recall":     round(float(res_tta_832.box.mr),    4),
}
print(f"  mAP50={tta_results['TTA (832)']['mAP50']:.4f}  "
      f"mAP50-95={tta_results['TTA (832)']['mAP50-95']:.4f}")

# ── Config 3: TTA at 1280 (recommended for small-object aerial) ──────────
print("\nTTA (imgsz=1280, augment=True)")
try:
    res_tta_1280 = tta_model.val(data="c2a.yaml", split="test",
                                  imgsz=1280, augment=True,
                                  batch=1, device=DEVICE, verbose=False)
    tta_results["TTA (1280)"] = {
        "mAP50":     round(float(res_tta_1280.box.map50), 4),
        "mAP50-95":  round(float(res_tta_1280.box.map),   4),
        "Precision":  round(float(res_tta_1280.box.mp),    4),
        "Recall":     round(float(res_tta_1280.box.mr),    4),
    }
    print(f"  mAP50={tta_results['TTA (1280)']['mAP50']:.4f}  "
          f"mAP50-95={tta_results['TTA (1280)']['mAP50-95']:.4f}")
except torch.cuda.OutOfMemoryError:
    print("  OOM at imgsz=1280 — skipping")
    gc.collect(); torch.cuda.empty_cache()

# ── Config 4: TTA at 1920 (aggressive, may OOM on T4) ────────────────────
print("\nTTA (imgsz=1920, augment=True)")
try:
    res_tta_1920 = tta_model.val(data="c2a.yaml", split="test",
                                  imgsz=1920, augment=True,
                                  batch=1, device=DEVICE, verbose=False)
    tta_results["TTA (1920)"] = {
        "mAP50":     round(float(res_tta_1920.box.map50), 4),
        "mAP50-95":  round(float(res_tta_1920.box.map),   4),
        "Precision":  round(float(res_tta_1920.box.mp),    4),
        "Recall":     round(float(res_tta_1920.box.mr),    4),
    }
    print(f"  mAP50={tta_results['TTA (1920)']['mAP50']:.4f}  "
          f"mAP50-95={tta_results['TTA (1920)']['mAP50-95']:.4f}")
except torch.cuda.OutOfMemoryError:
    print("  OOM at imgsz=1920 — skipping")
    gc.collect(); torch.cuda.empty_cache()

# ── TTA comparison table ─────────────────────────────────────────────────
print(f"\n{'='*70}")
print("TTA RESULTS COMPARISON")
print(f"{'='*70}")
tta_df = pd.DataFrame(tta_results).T
tta_df.index.name = "Config"
print(tta_df.to_string())
tta_df.to_excel(f"{EXCEL_DIR}/tta_comparison_noaug.xlsx")
print(f"\nTTA comparison saved -> {EXCEL_DIR}/tta_comparison_noaug.xlsx")

# Find best TTA config by mAP50-95
best_tta_name = max(tta_results, key=lambda k: tta_results[k].get("mAP50-95", 0))
best_tta = tta_results[best_tta_name]
print(f"\nBEST TTA: {best_tta_name}")
print(f"  mAP50={best_tta['mAP50']:.4f}  mAP50-95={best_tta['mAP50-95']:.4f}")

del tta_model; gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ============================================================================
# CELL 12: SAHI + TTA Combined — Best of Both Worlds
# ============================================================================
#
# SAHI handles spatial tiling (catches tiny objects in dense scenes).
# TTA handles multi-scale/flip (catches objects at varying scales).
# Combined: each SAHI tile gets TTA treatment.
#
# Implementation: SAHI internally calls model.predict(). We patch this
# to add augment=True, which activates ultralytics' built-in TTA.
#
# WARNING: This is 3x slower than SAHI alone (3 TTA passes per tile).
# For a paper, this is the "best achievable" configuration.
#
# ============================================================================

print("\n" + "=" * 80)
print("SAHI + TTA COMBINED EVALUATION")
print("=" * 80)

from sahi import AutoDetectionModel as _ADM_Combined
from sahi.predict import get_sliced_prediction as _get_sliced_combined

# ── Monkey-patch SAHI to enable TTA ──────────────────────────────────────
# SAHI's UltralyticsDetectionModel.perform_inference() does NOT pass
# augment=True to the YOLO model. We patch it to enable TTA per slice.
from sahi.models.ultralytics import UltralyticsDetectionModel

# Minimal monkey-patch: wrap the model's __call__ to inject augment=True
# into every prediction, then delegate to SAHI's ORIGINAL perform_inference.
# This preserves all original behavior (shape tracking, result parsing,
# config passing, mask handling) and just adds the TTA flag.
_original_perform_inference = UltralyticsDetectionModel.perform_inference

def _patched_perform_inference(self, image):
    """Patched: wraps model __call__ to inject augment=True for TTA."""
    _original_model_call = self.model.__call__
    def _augmented_call(*args, **kwargs):
        kwargs["augment"] = True
        return _original_model_call(*args, **kwargs)
    self.model.__call__ = _augmented_call
    try:
        _original_perform_inference(self, image)
    finally:
        self.model.__call__ = _original_model_call   # restore

# Apply patch
UltralyticsDetectionModel.perform_inference = _patched_perform_inference

# Load model
sahi_tta_model = _ADM_Combined.from_pretrained(
    model_type="ultralytics",
    model_path=BEST_PT,
    confidence_threshold=0.15,
    device=DEVICE,
)
print("SAHI+TTA model loaded (patched for augment=True)")

# Use the best SAHI config from the sweep
best_cfg_name = best_sahi["Config"]
# Parse slice size from config name
_cfg_map = {
    "slice256_ov30":  {"slice_h": 256,  "slice_w": 256,  "overlap": 0.30},
    "slice320_ov25":  {"slice_h": 320,  "slice_w": 320,  "overlap": 0.25},
    "slice512_ov25":  {"slice_h": 512,  "slice_w": 512,  "overlap": 0.25},
    "slice640_ov30":  {"slice_h": 640,  "slice_w": 640,  "overlap": 0.30},
}
sahi_tta_cfg = _cfg_map.get(best_cfg_name, {"slice_h": 320, "slice_w": 320, "overlap": 0.25})
print(f"Using SAHI config: {best_cfg_name} + TTA")

total_tp = total_fp = total_fn = 0
size_stats = {k: {"tp": 0, "fn": 0, "count": 0}
              for k in ("very_tiny", "tiny", "small", "medium", "large")}
inf_times = []

test_images_combo = get_image_list(TEST_IMG_DIR)
if TEST_MODE and TEST_IMAGES:
    test_images_combo = test_images_combo[:TEST_IMAGES]

for img_file in tqdm(test_images_combo, desc="SAHI+TTA", ncols=80):
    img_path = f"{TEST_IMG_DIR}/{img_file}"
    lbl_path = f"{TEST_LBL_DIR}/{img_file.rsplit('.', 1)[0]}.txt"
    img = cv2.imread(img_path)
    if img is None:
        continue
    H, W = img.shape[:2]
    gt_boxes = parse_yolo_label(lbl_path, W, H)

    t0 = time.perf_counter()
    result = _get_sliced_combined(
        img_path,
        sahi_tta_model,
        slice_height=sahi_tta_cfg["slice_h"],
        slice_width=sahi_tta_cfg["slice_w"],
        overlap_height_ratio=sahi_tta_cfg["overlap"],
        overlap_width_ratio=sahi_tta_cfg["overlap"],
        perform_standard_pred=True,
        postprocess_type="GREEDYNMM",
        postprocess_match_metric="IOS",
        postprocess_match_threshold=0.5,
        verbose=0,
    )
    inf_times.append((time.perf_counter() - t0) * 1000)

    pboxes = []
    for pred_obj in result.object_prediction_list:
        bb = pred_obj.bbox
        pboxes.append([bb.minx, bb.miny, bb.maxx, bb.maxy])
    pboxes = (torch.tensor(pboxes, dtype=torch.float32)
              if pboxes else torch.empty((0, 4), dtype=torch.float32))

    tp, fp, fn = match_preds_gt(pboxes, gt_boxes)
    total_tp += tp; total_fp += fp; total_fn += fn

    if len(gt_boxes) > 0:
        for cat, mask in categorise_boxes(gt_boxes).items():
            n_gt = int(mask.sum())
            size_stats[cat]["count"] += n_gt
            if n_gt > 0:
                stp, _, sfn = match_preds_gt(
                    pboxes, gt_boxes[mask] if mask.any() else gt_boxes[:0])
                size_stats[cat]["tp"] += stp
                size_stats[cat]["fn"] += sfn

sahi_tta_result = compute_metrics(total_tp, total_fp, total_fn, size_stats, inf_times)
sahi_tta_result["Config"] = f"SAHI({best_cfg_name}) + TTA"

print(f"\nSAHI + TTA Results:")
print(f"  P={sahi_tta_result['Precision']:.4f}  R={sahi_tta_result['Recall']:.4f}  "
      f"F1={sahi_tta_result['F1']:.4f}  F2={sahi_tta_result['F2']:.4f}")
for cat in ("very_tiny", "tiny", "small", "medium"):
    print(f"  {cat:12s}: recall={sahi_tta_result[f'{cat}_recall']:.4f}")
print(f"  Latency: {sahi_tta_result['Avg_Inf_ms']:.0f}ms")

# Restore original SAHI method
UltralyticsDetectionModel.perform_inference = _original_perform_inference

del sahi_tta_model; gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ============================================================================
# CELL 13: Grand Summary — All Configurations Compared
# ============================================================================

print("\n" + "=" * 80)
print("GRAND SUMMARY — All Inference Configurations")
print("=" * 80)

all_results = []

# Row 1: Baseline
all_results.append({
    "Configuration": "Baseline (640, standard)",
    "mAP50": baseline_official["mAP50"],
    "mAP50-95": baseline_official["mAP50-95"],
    **{k: v for k, v in baseline_custom.items() if k != "Config"},
    "Note": "standard ultralytics inference",
})

# Row 2-5: SAHI configs
for sr in sahi_results:
    all_results.append({
        "Configuration": f"SAHI ({sr['Config']})",
        "mAP50": "-",
        "mAP50-95": "-",
        **{k: v for k, v in sr.items() if k not in ("Config", "Slice", "Overlap", "Conf_Threshold")},
        "Note": f"SAHI custom eval, conf={sr.get('Conf_Threshold', 0.2)}",
    })

# Row 6+: TTA configs
for config_name, metrics in tta_results.items():
    if config_name == "No TTA (640)":
        continue   # already in baseline
    all_results.append({
        "Configuration": config_name,
        **metrics,
        "Note": "ultralytics val with augment=True",
    })

# Row: SAHI + TTA
all_results.append({
    "Configuration": sahi_tta_result["Config"],
    "mAP50": "-",
    "mAP50-95": "-",
    **{k: v for k, v in sahi_tta_result.items() if k != "Config"},
    "Note": "SAHI + TTA combined",
})

# Print summary table
print(f"\n{'Configuration':<35s} {'P':>6s} {'R':>6s} {'F1':>6s} {'F2':>6s} "
      f"{'VT_R':>6s} {'T_R':>6s} {'S_R':>6s} {'M_R':>6s} {'Lat':>7s}")
print("-" * 100)
for r in all_results:
    vt = r.get("very_tiny_recall", "-")
    t  = r.get("tiny_recall", "-")
    s  = r.get("small_recall", "-")
    m  = r.get("medium_recall", "-")
    lat = r.get("Avg_Inf_ms", "-")
    print(f"{r['Configuration']:<35s} "
          f"{r.get('Precision', '-'):>6} "
          f"{r.get('Recall', '-'):>6} "
          f"{r.get('F1', '-'):>6} "
          f"{r.get('F2', '-'):>6} "
          f"{vt if isinstance(vt, str) else f'{vt:.4f}':>6} "
          f"{t if isinstance(t, str) else f'{t:.4f}':>6} "
          f"{s if isinstance(s, str) else f'{s:.4f}':>6} "
          f"{m if isinstance(m, str) else f'{m:.4f}':>6} "
          f"{lat if isinstance(lat, str) else f'{lat:.0f}ms':>7}")

# Save grand summary
grand_df = pd.DataFrame(all_results)
grand_df.to_excel(f"{EXCEL_DIR}/grand_summary_sahi_tta_noaug.xlsx", index=False)
with open(f"{REPORT_DIR}/grand_summary_noaug.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nGrand summary saved -> {EXCEL_DIR}/grand_summary_sahi_tta_noaug.xlsx")


# ============================================================================
# CELL 14: Comparison Plots
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING COMPARISON PLOTS")
print("=" * 80)

# ── Plot 1: Per-size recall comparison across all configs ─────────────────
fig, ax = plt.subplots(figsize=(16, 7))
size_cats = ["very_tiny", "tiny", "small", "medium"]
size_labels = ["Very Tiny\n(<8px)", "Tiny\n(8-16px)", "Small\n(16-32px)", "Medium\n(32-96px)"]

# Select configs to plot (baseline + best SAHI + SAHI+TTA)
plot_configs = [baseline_custom]
plot_configs.append(best_sahi)
plot_configs.append(sahi_tta_result)

config_names = [
    "Baseline (640)",
    f"SAHI ({best_sahi['Config']})",
    sahi_tta_result["Config"],
]
colors = ["#1976D2", "#E64A19", "#388E3C"]

n_configs = len(plot_configs)
n_cats = len(size_cats)
bar_width = 0.8 / n_configs
x = np.arange(n_cats)

for i, (cfg, name, color) in enumerate(zip(plot_configs, config_names, colors)):
    vals = [cfg.get(f"{cat}_recall", 0) for cat in size_cats]
    bars = ax.bar(x + i * bar_width, vals, bar_width, label=name, color=color, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

# Add count annotations
for j, cat in enumerate(size_cats):
    count = baseline_custom.get(f"{cat}_count", 0)
    ax.text(x[j] + bar_width * (n_configs - 1) / 2, -0.03,
            f"n={count}", ha='center', va='top', fontsize=8, color='gray')

ax.set_xlabel("Object Size Category", fontsize=12)
ax.set_ylabel("Recall", fontsize=12)
ax.set_title("Per-Size Recall: Baseline vs SAHI vs SAHI+TTA (No-Aug Model, TEST)", fontsize=13)
ax.set_xticks(x + bar_width * (n_configs - 1) / 2)
ax.set_xticklabels(size_labels)
ax.set_ylim(0, 1.05)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/per_size_recall_sahi_tta_noaug.png", dpi=200)
plt.close()
print("  Plot saved: per_size_recall_sahi_tta_noaug.png")

# ── Plot 2: Metric improvement waterfall ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics_to_plot = [
    ("very_tiny_recall", "Very-Tiny Recall (<8px)", "#E64A19"),
    ("Recall",           "Overall Recall",          "#1976D2"),
    ("F2",               "F2 Score (recall-weighted)", "#388E3C"),
]

for ax, (metric, title, color) in zip(axes, metrics_to_plot):
    vals = []
    labels = ["Baseline"]
    vals.append(baseline_custom.get(metric, 0))

    for sr in sahi_results:
        labels.append(f"SAHI\n{sr['Config']}")
        vals.append(sr.get(metric, 0))

    labels.append(f"SAHI+TTA")
    vals.append(sahi_tta_result.get(metric, 0))

    bars = ax.bar(range(len(vals)), vals, color=[color]*len(vals), alpha=0.8)

    # Highlight best
    best_idx = np.argmax(vals)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)

    for i, (bar, v) in enumerate(zip(bars, vals)):
        delta = v - vals[0]  # delta from baseline
        label = f"{v:.4f}"
        if i > 0:
            label += f"\n({'+' if delta >= 0 else ''}{delta:.4f})"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                label, ha='center', va='bottom', fontsize=7)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, rotation=15, ha='right')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle("Inference Enhancement Impact (No-Aug Model)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/metric_improvement_waterfall_noaug.png", dpi=200, bbox_inches='tight')
plt.close()
print("  Plot saved: metric_improvement_waterfall_noaug.png")


# ============================================================================
# CELL 15: Master Report
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING MASTER REPORT")
print("=" * 80)

report_lines = [
    "=" * 80,
    "MAMBA+CBAM+P2 (NO-AUG) — SAHI + TTA EVALUATION REPORT",
    "=" * 80,
    f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}",
    f"Model     : {BEST_PT}",
    f"Dataset   : C2A ({n_test} test, {n_val} val images)",
    f"TEST MODE : {TEST_MODE}",
    "",
    "BASELINE (640px standard inference):",
    f"  mAP50={baseline_official['mAP50']:.4f}  mAP50-95={baseline_official['mAP50-95']:.4f}",
    f"  P={baseline_custom['Precision']:.4f}  R={baseline_custom['Recall']:.4f}  "
    f"F1={baseline_custom['F1']:.4f}  F2={baseline_custom['F2']:.4f}",
    f"  Very-tiny recall: {baseline_custom['very_tiny_recall']:.4f}",
    f"  Latency: {baseline_custom['Avg_Inf_ms']:.1f}ms",
    "",
    "SAHI SWEEP RESULTS:",
]

for sr in sahi_results:
    report_lines.append(
        f"  {sr['Config']:20s}: P={sr['Precision']:.4f}  R={sr['Recall']:.4f}  "
        f"F1={sr['F1']:.4f}  VT={sr['very_tiny_recall']:.4f}  "
        f"Lat={sr['Avg_Inf_ms']:.0f}ms"
    )

report_lines += [
    "",
    f"BEST SAHI: {best_sahi['Config']}",
    f"  Very-tiny recall: {best_sahi['very_tiny_recall']:.4f}  "
    f"(+{best_sahi['very_tiny_recall'] - baseline_custom['very_tiny_recall']:.4f} vs baseline)",
    f"  F1: {best_sahi['F1']:.4f}  "
    f"(+{best_sahi['F1'] - baseline_custom['F1']:.4f} vs baseline)",
    "",
    "TTA RESULTS:",
]

for name, metrics in tta_results.items():
    report_lines.append(
        f"  {name:20s}: mAP50={metrics['mAP50']:.4f}  mAP50-95={metrics['mAP50-95']:.4f}"
    )

report_lines += [
    "",
    f"BEST TTA: {best_tta_name}",
    f"  mAP50={best_tta['mAP50']:.4f}  mAP50-95={best_tta['mAP50-95']:.4f}",
    "",
    "SAHI + TTA COMBINED:",
    f"  Config: {sahi_tta_result['Config']}",
    f"  P={sahi_tta_result['Precision']:.4f}  R={sahi_tta_result['Recall']:.4f}  "
    f"F1={sahi_tta_result['F1']:.4f}  F2={sahi_tta_result['F2']:.4f}",
    f"  Very-tiny recall: {sahi_tta_result['very_tiny_recall']:.4f}",
    f"  Latency: {sahi_tta_result['Avg_Inf_ms']:.0f}ms",
    "",
    "=" * 80,
]

report_text = "\n".join(report_lines)
print(report_text)

with open(f"{REPORT_DIR}/MASTER_REPORT_SAHI_TTA_NOAUG.txt", "w") as f:
    f.write(report_text)
print(f"\nMaster report saved -> {REPORT_DIR}/MASTER_REPORT_SAHI_TTA_NOAUG.txt")


# ============================================================================
# CELL 16: Package Results for Download
# ============================================================================
import zipfile

zip_name = "/kaggle/working/sahi_tta_noaug_results.zip"
print(f"\nPackaging results -> {zip_name}")

with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    for folder in [EXCEL_DIR, PLOT_DIR, REPORT_DIR]:
        for root, dirs, files in os.walk(folder):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, "/kaggle/working")
                zf.write(filepath, arcname)

print(f"  {zip_name}  ({os.path.getsize(zip_name)/1024:.0f} KB)")
print("\n" + "=" * 80)
print("ALL DONE. Download sahi_tta_noaug_results.zip from /kaggle/working/")
print("=" * 80)

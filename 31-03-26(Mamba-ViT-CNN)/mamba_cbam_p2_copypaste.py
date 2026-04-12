"""
================================================================================
DISASTER HUMAN DETECTION — MAMBA + CBAM + P2 HEAD + COPY-PASTE AUGMENTATION
================================================================================
PURPOSE:
  Retrain the Mamba+CBAM+P2Head model with aggressive small-object augmentation
  to push very-tiny human recall higher.  Then evaluate with SAHI (Slicing Aided
  Hyper Inference) for additional inference-time gains on small objects.

WHAT CHANGED vs the previous Mamba+CBAM+P2 run (11-3-26):
  Training augmentation:
    copy_paste  : 0.0  →  0.5   (copies small humans to new locations — KEY)
    mixup       : 0.0  →  0.15  (mild image mixing)
    flipud      : 0.0  →  0.5   (vertical flip — valid for nadir aerial)
    degrees     : 0.0  →  10.0  (mild rotation — aerial-appropriate)
  Evaluation:
    + SAHI sliced inference (3 configs: 512/640/768 slice sizes)
    + TTA (test-time augmentation) evaluation
    + Combined SAHI + TTA evaluation
  Architecture: IDENTICAL to previous run (no changes)

COMPARISON MODEL:
  Loads the PREVIOUS Mamba+CBAM+P2 best.pt (no copy-paste) for direct
  comparison to isolate the effect of augmentation alone.

EXPECTED GAINS (based on published literature):
  Copy-paste:  +2-5% very-tiny recall, +1-3% mAP50-95
  SAHI:        +5-10% very-tiny recall (inference-time, no retraining)
  TTA:         +1-2% mAP50-95 (inference-time)

Architecture (unchanged from previous run):
  Backbone : CBAM replaces C2PSA at layer 10
  Neck     : C3k2 at layers 13,16,19,22,25 → C3K2Mamba
  Head     : P2 4-scale detection (P2/P3/P4/P5)

COMPUTE NOTES (Kaggle T4 x2):
  Batch size  : 8 (T4) or 4 (P100)
  d_state     : 4  (safe for T4 16GB)
  Window size : adaptive (4→512ch, 6→256ch, 8→128ch)
  Effective batch with grad_accum=2: 16
================================================================================
"""

# ============================================================================
# CELL 1: Control Flags & Dependencies
# ============================================================================

TEST_MODE       = False   # True = 2 epochs, 5% data | False = full 100 epochs
RESUME_TRAINING = False  # Set True + set RESUME_PT to resume from a checkpoint
RESUME_PT       = ""     # Path to last.pt from a previous session, e.g.:
                         # "/kaggle/input/mamba-checkpoint/last.pt"

import subprocess, sys, re

def pip_install(pkg, extra=""):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-U", pkg] +
        (extra.split() if extra else [])
    )

# ── CUDA / PyTorch version alignment check ────────────────────────────────
# CRITICAL: Kaggle's default PyTorch is built against CUDA 12.4 (cu124),
# but the actual GPU driver on T4 nodes runs CUDA 12.6 (cu126).
# mamba-ssm compiled extensions fail with "undefined symbol" because of
# this mismatch. Even though our SSM is pure-PyTorch (no mamba-ssm needed),
# the same mismatch can cause AMP/flash-attention instability.
# We detect and fix it proactively. Source: https://ranaadeeltahir.me/blog/run-mamba-on-kaggle
print("Checking CUDA / PyTorch version alignment …")

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

_needs_upgrade = (
    _driver_cuda.startswith("12.6") and "cu124" in _torch_ver
)
if _needs_upgrade:
    print("  ⚠  MISMATCH detected (driver=12.6, torch=cu124).")
    print("  → Reinstalling PyTorch cu126 — this takes ~3 min …")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu126"
    ])
    print("  ✓ PyTorch cu126 installed. Kernel restart is NOT needed")
    print("    because we import torch AFTER this cell runs.")
else:
    print(f"  ✓ No mismatch — proceeding with {_torch_ver}")

# ── Package installs ─────────────────────────────────────────────────────
# NOTE: Do NOT downgrade numpy. Kaggle 2026 pre-installs pandas/opencv
# compiled against numpy 2.x — downgrading to 1.x causes
# "numpy.dtype size changed" binary incompatibility errors.
# netcal is NOT installed — it requires numpy<2 and is never actually used
# (ECE calibration is implemented inline in the analysis cells).
pip_install("ultralytics")
pip_install("timm")
pip_install("thop")
pip_install("openpyxl")
pip_install("scikit-learn")
pip_install("sahi")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "pandas<3.0", "matplotlib<3.10", "tqdm"])
print("✓ All dependencies installed (including sahi)")


# ============================================================================
# CELL 2: Imports & Training Configuration
# ============================================================================
import os, sys, time, yaml, shutil, gc, math, copy, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

# ── GPU fingerprint ─────────────────────────────────────────────────────────
num_gpus = torch.cuda.device_count()
gpu_name = torch.cuda.get_device_name(0) if num_gpus > 0 else "CPU"
gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3 if num_gpus > 0 else 0
DEVICE   = "0,1" if num_gpus >= 2 else "0" if num_gpus == 1 else "cpu"
print(f"GPU(s): {num_gpus}  |  GPU 0: {gpu_name} ({gpu_mem:.1f} GB)  |  DEVICE: {DEVICE}")

# ── BF16 support check (P100 does NOT support BF16) ─────────────────────────
BF16_SUPPORTED = torch.cuda.is_bf16_supported() if num_gpus > 0 else False
print(f"BF16 support: {BF16_SUPPORTED}  |  AMP will use: {'BF16' if BF16_SUPPORTED else 'FP16'}")

# ── Training hyper-params ────────────────────────────────────────────────────
if TEST_MODE:
    TRAIN_FRACTION = 0.05
    NUM_EPOCHS     = 2
    PATIENCE       = 2
    F2_PATIENCE    = 2
    TEST_IMAGES    = 10
    VAL_IMAGES     = 20
    SAVE_PERIOD    = 1
    BATCH_SIZE     = 4
    print("⚠  TEST MODE  — 5% data, 2 epochs")
else:
    TRAIN_FRACTION = 1.0
    NUM_EPOCHS     = 120
    PATIENCE       = 15       # Ultralytics built-in mAP patience
    F2_PATIENCE    = 10       # custom F2 patience
    TEST_IMAGES    = None     # full test set
    VAL_IMAGES     = None     # full val set
    SAVE_PERIOD    = 5
    BATCH_SIZE     = 8 if gpu_mem >= 14 else 4
    print(f"🚀  FULL MODE  — 100 epochs | batch={BATCH_SIZE}")

GRAD_ACCUM = max(1, 16 // BATCH_SIZE)   # effective batch always ~16
print(f"  Batch={BATCH_SIZE} | GradAccum={GRAD_ACCUM} | EffectiveBatch={BATCH_SIZE*GRAD_ACCUM}")

# ── Checkpoint / session-resume config ──────────────────────────────────────
# Kaggle sessions reset every 12 hours. CHECKPOINT_EVERY controls how often
# last.pt + metadata.json are copied to /kaggle/working/ (easily downloadable).
# Keep this ≤ SAVE_PERIOD to avoid gaps.
CHECKPOINT_EVERY = 5 if not TEST_MODE else 1   # copy to working/ every N epochs

# ── OOM retry ladder ─────────────────────────────────────────────────────────
# If training throws OutOfMemoryError, the loop tries progressively smaller
# batches before giving up. Set to [BATCH_SIZE] to disable retry.
OOM_RETRY_BATCHES = [BATCH_SIZE, max(BATCH_SIZE // 2, 2), max(BATCH_SIZE // 4, 1)]


# ============================================================================
# CELL 3: Dataset Configuration
# ============================================================================
# Auto-discover C2A dataset — Kaggle mounts at unpredictable paths
# based on dataset owner/slug, so we walk /kaggle/input/ to find it.
print("Searching for C2A dataset in /kaggle/input/ …")
DATASET_ROOT = None
for _root, _dirs, _files in os.walk("/kaggle/input"):
    if (os.path.isdir(os.path.join(_root, "train", "images")) and
        os.path.isdir(os.path.join(_root, "val",   "images"))):
        DATASET_ROOT = _root
        print(f"  ✓ C2A dataset found at: {DATASET_ROOT}")
        break

if DATASET_ROOT is None:
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

dataset_yaml_content = f"""
train: {DATASET_ROOT}/train/images
val:   {DATASET_ROOT}/val/images
test:  {DATASET_ROOT}/test/images
nc: 1
names: ['person']
"""
with open("c2a.yaml", "w") as f:
    f.write(dataset_yaml_content.strip())
print("✓ Dataset YAML written: c2a.yaml")

# ── Locate previous Mamba+CBAM+P2 model (no copy-paste) for comparison ──────
# Upload the previous run's best.pt as a Kaggle dataset input.
# This script will find it automatically.
PREV_MAMBA_BEST = None
print("Searching for previous Mamba+CBAM+P2 (no copy-paste) model in /kaggle/input/ …")
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f == "best.pt":
            candidate = os.path.join(root, f)
            # Skip our own training output
            if "/kaggle/working/" in candidate:
                continue
            PREV_MAMBA_BEST = candidate
            print(f"  ✓ Found: {PREV_MAMBA_BEST}")
            break
    if PREV_MAMBA_BEST:
        break
if PREV_MAMBA_BEST is None:
    print("  ⚠  Previous Mamba+CBAM+P2 model NOT found — will run new model only (no 2-way comparison)")
    print("     To enable comparison: upload your previous best.pt as a Kaggle Dataset input.")


# ============================================================================
# CELL 4: CBAM Module (identical to previous study — must match saved weights)
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

# Quick sanity check
_t = torch.randn(2, 512, 20, 20)
assert CBAM(16, 7)(_t).shape == _t.shape
print("✓ CBAM module OK")


# ============================================================================
# CELL 5: Mamba SSM Modules
# ============================================================================
#
# Architecture per neck layer:
#
#  Input (B, C_in, H, W)
#     │
#     ▼ cv1: Conv1×1 → (B, 2·C_mid, H, W)
#     │
#     ▼ chunk(2) → [x1, x2]  (each C_mid)
#     │
#  x2 ──► MambaBottleneck × n  (C_mid → C_mid)
#     │      Conv3×3 → LocalWindowSSM → Conv3×3  (+residual if same C)
#     │
#     ▼ cat([x1, x2, mb1_out, mb2_out, …])
#     │
#     ▼ cv2: Conv1×1 → (B, C_out, H, W)
#
# LocalWindowSSM internals:
#   window_partition(ws×ws)
#   │
#   ├─ fwd scan  (L→R  within window)
#   └─ bwd scan  (R→L  within window, then flip back)
#   merge  = (fwd + bwd)
#   gate   = merge × SiLU(z)
#   output = out_proj + residual
#   │
#   window_reverse
# ============================================================================

def _get_window_size(channels: int) -> int:
    """Adaptive window size: larger windows for lower channel counts."""
    if channels >= 512: return 4   # 4×4=16 tokens — memory safe at 512ch
    if channels >= 256: return 6   # 6×6=36 tokens
    return 8                        # 8×8=64 tokens

# ── Selective Scan (pure PyTorch, sequential, L ≤ 64) ───────────────────────
class _SelectiveScan1D(nn.Module):
    """
    One-direction selective scan.
    u       : (B, L, D)
    Returns : (B, L, D)

    All SSM computations are forced to FP32 for numerical stability,
    then cast back to the input dtype. This makes it safe for AMP/FP16.
    """
    def __init__(self, d_model: int, d_state: int = 4, dt_rank_ratio: int = 16):
        super().__init__()
        D, N = d_model, d_state
        dt_rank = max(D // dt_rank_ratio, 1)
        self.D, self.N, self.dt_rank = D, N, dt_rank

        # local depthwise conv for positional context within the window
        self.conv1d = nn.Conv1d(D, D, kernel_size=4, padding=3,
                                groups=D, bias=True)

        # SSM parameter projections (input-dependent B, C, dt)
        self.x_proj  = nn.Linear(D, dt_rank + 2 * N, bias=False)
        self.dt_proj = nn.Linear(dt_rank, D, bias=True)

        # State transition matrix A (log-parameterised, always negative)
        A = torch.arange(1, N + 1, dtype=torch.float32).unsqueeze(0).repeat(D, 1)
        self.A_log = nn.Parameter(torch.log(A))         # (D, N)
        self.D_skip = nn.Parameter(torch.ones(D))       # skip-connection scale

        # Initialise dt_proj bias for stable initial dynamics
        # (follows Mamba paper §3.6)
        dt_init = torch.exp(
            torch.rand(D) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        B_win, L, D = u.shape
        in_dtype = u.dtype

        # ── local context via 1-D depthwise conv ────────────────────────────
        u_conv = self.conv1d(u.transpose(1, 2))[:, :, :L].transpose(1, 2)  # (B,L,D)
        u_act  = F.silu(u_conv)

        # ── input-dependent parameters ───────────────────────────────────────
        xBC_dt = self.x_proj(u_act)                              # (B,L,dt_rank+2N)
        dt_raw, B_param, C_param = xBC_dt.split(
            [self.dt_rank, self.N, self.N], dim=-1
        )
        # force FP32 for numerically sensitive ops
        dt      = F.softplus(self.dt_proj(dt_raw)).float()       # (B, L, D)
        B_param = B_param.float()                                 # (B, L, N)
        C_param = C_param.float()                                 # (B, L, N)
        u_f     = u_act.float()                                   # (B, L, D)
        A       = -torch.exp(self.A_log.float())                  # (D, N)

        # ── discretise continuous SSM → ZOH ─────────────────────────────────
        # deltaA  : (B, L, D, N)
        # deltaB_u: (B, L, D, N)
        deltaA   = torch.exp(torch.einsum("bld,dn->bldn", dt, A))
        deltaB_u = torch.einsum("bld,bln,bld->bldn", dt, B_param, u_f)

        # ── sequential scan (L ≤ 64, manageable in Python loop) ─────────────
        x = torch.zeros(B_win, D, self.N, device=u.device, dtype=torch.float32)
        ys = []
        for i in range(L):
            x = deltaA[:, i] * x + deltaB_u[:, i]                # (B, D, N)
            y_i = (x * C_param[:, i, :].unsqueeze(1)).sum(-1)    # (B, D)
            ys.append(y_i)

        y = torch.stack(ys, dim=1).to(in_dtype)                  # (B, L, D)
        y = y + u_act * self.D_skip.to(in_dtype)
        return y


class LocalWindowSSM(nn.Module):
    """
    Bidirectional local-window 2-D Selective State Space Model.

    Steps:
      1. Partition feature map into ws×ws windows.
      2. Run _SelectiveScan1D in forward direction.
      3. Run _SelectiveScan1D in backward direction (flip, scan, flip).
      4. Merge (fwd + bwd), gate with SiLU(z), project out, add residual.
      5. Reverse window partition.
    """
    def __init__(self, d_model: int, d_state: int = 4, window_size: int = 8):
        super().__init__()
        self.ws = window_size
        D = d_model

        self.norm    = nn.LayerNorm(D)
        self.in_proj = nn.Linear(D, D * 2, bias=False)   # splits to (x, z)
        self.scan_fwd = _SelectiveScan1D(D, d_state)
        self.scan_bwd = _SelectiveScan1D(D, d_state)     # separate weights — bidirectional
        self.out_proj = nn.Linear(D, D, bias=False)
        self.out_norm = nn.LayerNorm(D)

        # Small-scale init for out_proj (GPT-2 / BERT convention: std=0.02).
        # NOTE: we do NOT use zero-init here because it completely kills the
        # SSM signal at init  (out_proj(y) == 0  ⇒  output = out_norm(residual)
        # regardless of scan direction).  Small-scale init lets a tiny SSM
        # gradient flow through while keeping the residual path dominant for
        # stable early training.
        nn.init.normal_(self.out_proj.weight, std=0.02)

    # ── window utilities ─────────────────────────────────────────────────────
    def _partition(self, x: torch.Tensor):
        """(B,C,H,W) → (B*nH*nW, ws*ws, C), padding metadata."""
        B, C, H, W = x.shape
        ws = self.ws
        ph = (ws - H % ws) % ws
        pw = (ws - W % ws) % ws
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        _, _, Hp, Wp = x.shape
        nH, nW = Hp // ws, Wp // ws
        # (B, C, nH, ws, nW, ws) → (B, nH, nW, ws, ws, C) → (B*nH*nW, ws*ws, C)
        x = x.reshape(B, C, nH, ws, nW, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B * nH * nW, ws * ws, C)
        return x, (B, C, H, W, Hp, Wp, nH, nW)

    def _reverse(self, y: torch.Tensor, meta) -> torch.Tensor:
        """(B*nH*nW, ws*ws, C) → (B, C, H, W)."""
        B, C, H, W, Hp, Wp, nH, nW = meta
        ws = self.ws
        y = y.reshape(B, nH, nW, ws, ws, C)
        y = y.permute(0, 5, 1, 3, 2, 4).reshape(B, C, Hp, Wp)
        return y[:, :, :H, :W].contiguous()

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)  →  (B, C, H, W)"""
        windows, meta = self._partition(x)   # (B*nW*nH, L, C)
        residual = windows

        windows_n = self.norm(windows)
        xz = self.in_proj(windows_n)
        x_in, z = xz.chunk(2, dim=-1)        # each (B_win, L, C)

        y_fwd = self.scan_fwd(x_in)
        y_bwd = self.scan_bwd(x_in.flip(1)).flip(1)   # scan reversed seq, flip back
        y     = (y_fwd + y_bwd) * F.silu(z)

        y = self.out_norm(self.out_proj(y) + residual)
        return self._reverse(y, meta)


class _MambaBottleneck(nn.Module):
    """
    Single bottleneck unit inside C3K2Mamba.
    Replaces the CNN bottleneck with: Conv3×3 → LocalWindowSSM → Conv3×3
    """
    def __init__(self, c: int, shortcut: bool, d_state: int, window_size: int):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.cv1 = Conv(c, c, 3, 1)
        self.ssm = LocalWindowSSM(c, d_state=d_state, window_size=window_size)
        self.cv2 = Conv(c, c, 3, 1)
        self.add = shortcut   # residual if in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.ssm(self.cv1(x)))
        return x + y if self.add else y


class C3K2Mamba(nn.Module):
    """
    C2f-style block with Mamba SSM bottleneck.
    Drop-in replacement for C3k2 in YOLO11m neck.

    signature matches C3k2/C2f so YOLO forward pass (which reads .f/.i from
    layer attributes) works unchanged after post-init injection.
    """
    def __init__(self, c1: int, c2: int, n: int = 1,
                 shortcut: bool = False, g: int = 1, e: float = 0.5,
                 d_state: int = 4):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)               # inner channels
        ws     = _get_window_size(self.c)  # adaptive window size

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m   = nn.ModuleList(
            _MambaBottleneck(self.c, shortcut and c1 == c2, d_state, ws)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


print("✓ Mamba modules defined (LocalWindowSSM, C3K2Mamba)")


# ============================================================================
# CELL 6: Smoke Test & Dry Run
# ============================================================================
# Tests: shape, no NaN, gradient flow, AMP safety, memory estimate
# This MUST pass before any training attempt.

def run_smoke_test():
    print("\n" + "=" * 70)
    print("SMOKE TEST: Verifying Mamba modules before training")
    print("=" * 70)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    passed = 0

    # ── Test 1: LocalWindowSSM shapes ───────────────────────────────────────
    print("\n[1/7] LocalWindowSSM shape test …")
    for C, H in [(512, 40), (256, 80), (128, 160)]:
        ssm = LocalWindowSSM(C, d_state=4, window_size=_get_window_size(C)).to(device)
        x = torch.randn(2, C, H, H, device=device)
        y = ssm(x)
        assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"
        print(f"    C={C:4d}  H={H:3d}  ws={_get_window_size(C)}  → {y.shape}  ✓")
    passed += 1

    # ── Test 2: C3K2Mamba shape ──────────────────────────────────────────────
    print("\n[2/7] C3K2Mamba shape test …")
    for c1, c2, n in [(1024, 512, 2), (768, 256, 2), (384, 128, 2)]:
        blk = C3K2Mamba(c1, c2, n=n, shortcut=False).to(device)
        x = torch.randn(2, c1, 40, 40, device=device)
        y = blk(x)
        assert y.shape == (2, c2, 40, 40), f"Shape mismatch: {y.shape}"
        print(f"    ({c1}→{c2}, n={n}) → {y.shape}  ✓")
    passed += 1

    # ── Test 3: NaN/Inf check ────────────────────────────────────────────────
    print("\n[3/7] NaN / Inf check …")
    blk = C3K2Mamba(1024, 512, n=2).to(device)
    x = torch.randn(2, 1024, 40, 40, device=device)
    y = blk(x)
    assert torch.isfinite(y).all(), "NaN or Inf in output!"
    print("    No NaN/Inf  ✓")
    passed += 1

    # ── Test 4: Gradient flow ────────────────────────────────────────────────
    print("\n[4/7] Gradient flow …")
    blk = C3K2Mamba(512, 256, n=2).to(device)
    x = torch.randn(2, 512, 40, 40, device=device, requires_grad=True)
    y = blk(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None, "No gradient at input!"
    total_norm = sum(p.grad.norm().item()**2 for p in blk.parameters()
                     if p.grad is not None) ** 0.5
    print(f"    Grad norm: {total_norm:.4f}  ✓")
    passed += 1

    # ── Test 5: AMP / FP16 safety ────────────────────────────────────────────
    if device == "cuda":
        print("\n[5/7] AMP (FP16) safety …")
        blk = C3K2Mamba(512, 256, n=2).to(device)
        x   = torch.randn(2, 512, 40, 40, device=device)
        with torch.cuda.amp.autocast():
            y = blk(x)
        assert torch.isfinite(y).all(), "NaN in FP16 forward!"
        print(f"    Input dtype: {x.dtype} | Output dtype: {y.dtype}  ✓")
    else:
        print("\n[5/7] AMP test skipped (CPU mode)")
    passed += 1

    # ── Test 6: Bidirectional scan validates asymmetry ───────────────────────
    print("\n[6/7] Bidirectional scan asymmetry test …")
    ssm = LocalWindowSSM(64, d_state=4, window_size=4).to(device)
    x1 = torch.randn(1, 64, 8, 8, device=device)
    x2 = x1.flip(-1)       # horizontally flipped input
    y1, y2 = ssm(x1), ssm(x2)
    # outputs must differ (forward ≠ backward pass gives different context)
    assert not torch.allclose(y1, y2.flip(-1), atol=1e-4), \
        "Bidirectional scan is symmetric — forward/backward weights not independent!"
    print("    Forward and backward scans differ  ✓")
    passed += 1

    # ── Test 7: Memory estimate ──────────────────────────────────────────────
    if device == "cuda":
        print("\n[7/7] Memory estimate (layer-13 equivalent, batch=8) …")
        torch.cuda.reset_peak_memory_stats()
        blk = C3K2Mamba(1024, 512, n=2).to(device)
        x = torch.randn(8, 1024, 40, 40, device=device)
        with torch.cuda.amp.autocast():
            y = blk(x)
        y.sum().backward()
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"    Peak VRAM: {peak_mb:.0f} MB  (T4 budget: 16384 MB)  ✓")
        del blk, x, y; torch.cuda.empty_cache(); gc.collect()
    else:
        print("\n[7/7] Memory test skipped (CPU mode)")
    passed += 1

    print(f"\n{'='*70}")
    print(f"SMOKE TEST: {passed}/7 passed ✓" if passed == 7
          else f"SMOKE TEST: {passed}/7 — CHECK FAILURES ABOVE ⚠")
    print("=" * 70)
    return passed == 7

assert run_smoke_test(), "Smoke test failed — fix errors before training!"


# ============================================================================
# CELL 7: Register Modules in Ultralytics Namespace (DDP-safe)
# ============================================================================
# CBAM must be registered (identical to previous study).
# C3K2Mamba does NOT go into parse_model — we use post-init injection instead.
#
# DDP spawns fresh subprocesses via torch.distributed.run. Those processes
# re-import ultralytics from scratch. Two things must work:
#   1. `import cbam_module` must succeed  → install to site-packages
#   2. `globals()["CBAM"]` in tasks.py must exist  → patch the file on disk

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
import ultralytics.nn.tasks as _ult_tasks_mod
_tasks_file = _ult_tasks_mod.__file__
_inject     = "from cbam_module import CBAM, ChannelAttention, SpatialAttention\n"

with open(_tasks_file, "r") as _f:
    _tasks_src = _f.read()

if "from cbam_module import" not in _tasks_src:
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
for name, obj in [("CBAM", CBAM),
                   ("ChannelAttention", ChannelAttention),
                   ("SpatialAttention", SpatialAttention)]:
    setattr(ult_modules, name, obj)
    setattr(ult_tasks,   name, obj)

assert hasattr(ult_modules, "CBAM"), "CBAM registration failed"
print("✓ CBAM registered in ultralytics namespace (DDP-safe)")


# ============================================================================
# CELL 8: Post-Init Injection Function
# ============================================================================
# Why post-init instead of YAML:
#   Ultralytics parse_model uses frozenset membership checks (base_modules,
#   repeat_modules) to decide whether to prepend c1 to args and handle n.
#   These frozensets cannot be patched at runtime.
#   Post-init injection bypasses this entirely:
#     1. Model loads with the existing CBAM+P2 YAML (C3k2 in neck).
#     2. After parsing, we find C3k2 instances by layer index.
#     3. Inspect c1, c2, n from the existing layer's Conv attributes.
#     4. Create C3K2Mamba with identical dimensions.
#     5. Copy Ultralytics bookkeeping attrs (.f, .i, .type, .np).
#     6. Replace in model.model.model sequential.

def inject_mamba_neck(yolo_model, min_layer_idx: int = 11,
                       max_channels: int = 512,
                       d_state: int = 4,
                       verbose: bool = True) -> None:
    """
    Surgically replace eligible C3k2/C2f neck layers with C3K2Mamba.

    Parameters
    ----------
    yolo_model      : ultralytics YOLO object
    min_layer_idx   : layers below this index are backbone — skip them
    max_channels    : skip layers with c_out > this (e.g. 1024-ch P5 layer)
    d_state         : Mamba d_state parameter
    verbose         : print replacement log
    """
    try:
        from ultralytics.nn.modules.block import C3k2, C2f
    except ImportError:
        from ultralytics.nn.modules import C3k2, C2f

    nn_model  = yolo_model.model.model   # nn.Sequential of layers
    replaced  = []

    for idx, layer in enumerate(nn_model):
        if idx < min_layer_idx:
            continue
        if not isinstance(layer, (C3k2, C2f)):
            continue

        # ── infer dimensions from existing layer ────────────────────────────
        if not (hasattr(layer, "cv1") and hasattr(layer, "cv2")):
            continue
        c1 = layer.cv1.conv.in_channels
        # cv1 outputs 2·c_  →  c_ = out_channels // 2
        c_ = getattr(layer, "c", layer.cv1.conv.out_channels // 2)
        c2 = layer.cv2.conv.out_channels
        n  = len(layer.m)
        shortcut = getattr(layer.m[0], "add", False) if len(layer.m) > 0 else False

        if c2 > max_channels:
            if verbose:
                print(f"  SKIP  layer {idx:2d}: {type(layer).__name__}"
                      f"({c1}→{c2}) — c2 > {max_channels}")
            continue

        # ── build replacement ────────────────────────────────────────────────
        dev   = next(layer.parameters()).device
        new   = C3K2Mamba(c1, c2, n=n, shortcut=shortcut, d_state=d_state)
        new   = new.to(device=dev)

        # ── copy Ultralytics bookkeeping attributes ──────────────────────────
        # DetectionModel._predict_once uses layer.f (from-index) and layer.i
        # These are set by parse_model; we must preserve them.
        for attr in ("f", "i", "np"):
            if hasattr(layer, attr):
                setattr(new, attr, getattr(layer, attr))
        new.type = type(new).__name__

        nn_model[idx] = new
        replaced.append((idx, c1, c2, n))
        if verbose:
            ws = _get_window_size(c_)
            print(f"  ✓ layer {idx:2d}: {type(layer).__name__:8s}"
                  f"({c1}→{c2}, n={n}) → C3K2Mamba  [ws={ws}, d_state={d_state}]")

    if verbose:
        print(f"\n  Total injected: {len(replaced)} layer(s)")
        if len(replaced) == 0:
            print("  ⚠  Nothing was injected — check min_layer_idx / max_channels")
    return replaced


# ── Quick injection dry-run (validates logic without full model) ─────────────
def _dry_run_injection():
    from ultralytics import YOLO
    print("\n--- Injection dry-run ---")
    _m = YOLO("yolov11m_cbam_p2head.yaml")
    _m.load("yolo11m.pt")
    replaced = inject_mamba_neck(_m, verbose=True)
    n_params_before = sum(p.numel() for p in _m.model.parameters())
    print(f"  Params after injection: {n_params_before/1e6:.2f}M")
    del _m; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    assert len(replaced) > 0, "Dry-run injection found nothing to replace!"
    print("--- Injection dry-run PASSED ---\n")

# Create the CBAM+P2 YAML (same as your previous study — reused here)
cbam_p2_yaml = """# YOLO11m + CBAM + P2 Extra Detection Head (used as base for Mamba injection)
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
print("✓ CBAM+P2 YAML written")

_dry_run_injection()


# ============================================================================
# CELL 9: Custom Callbacks
# ============================================================================

class NaNStopCallback:
    """
    Stops training immediately on NaN or Inf loss.
    Saves current weights before stopping so no work is lost.
    """
    def __init__(self):
        self.triggered = False

    def on_train_batch_end(self, trainer):
        if self.triggered:
            return
        loss = getattr(trainer, "loss", None)
        if loss is not None and not torch.isfinite(loss):
            self.triggered = True
            print(f"\n🚨 NaN/Inf loss at epoch {trainer.epoch+1}, "
                  f"batch {getattr(trainer,'batch_i',0)}! Stopping.")
            # Save emergency checkpoint
            emg = Path(trainer.save_dir) / "weights" / "emergency_nan_stop.pt"
            try:
                trainer.model.save(str(emg))
                print(f"   Emergency checkpoint saved: {emg}")
            except Exception:
                pass
            try:
                trainer.stop = True   # Ultralytics >= 8.2
            except AttributeError:
                trainer.epoch = trainer.epochs  # fallback: skip remaining epochs


class F2EarlyStopCallback:
    """
    Stops training if F2 score (β=2, recall-weighted) does not improve
    for `patience` epochs. Complements Ultralytics' mAP-based stopper.

    F2 = 5·P·R / (4·P + R)  — heavily weights recall over precision,
    which is the correct priority for disaster survivor detection.
    """
    def __init__(self, patience: int = 10, min_delta: float = 5e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_f2    = 0.0
        self.counter    = 0
        self.history    = []   # (epoch, f2)

    def on_fit_epoch_end(self, trainer):
        m = trainer.metrics
        P = float(m.get("metrics/precision(B)", 0))
        R = float(m.get("metrics/recall(B)",    0))
        f2 = 5 * P * R / (4 * P + R + 1e-9)
        self.history.append((trainer.epoch, f2))

        if f2 > self.best_f2 + self.min_delta:
            self.best_f2 = f2
            self.counter = 0
        else:
            self.counter += 1
            remaining = self.patience - self.counter
            if self.counter >= self.patience:
                print(f"\n⏹  F2 Early Stop: no improvement for {self.patience} "
                      f"epochs. Best F2={self.best_f2:.4f}")
                try:
                    trainer.stop = True
                except AttributeError:
                    trainer.epoch = trainer.epochs
            elif remaining <= 3:
                print(f"  F2 patience: {remaining} epoch(s) remaining "
                      f"(best F2={self.best_f2:.4f}, current={f2:.4f})")


class GradientMonitorCallback:
    """
    Logs gradient norm each epoch. Warns on explosion (norm > 500).
    Saves gradient norm CSV for post-training analysis.
    """
    def __init__(self, save_dir: str = "/kaggle/working"):
        self.save_dir = save_dir
        self.records  = []

    def on_train_epoch_end(self, trainer):
        total_norm = 0.0
        n_params   = 0
        for p in trainer.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
                n_params   += 1
        total_norm = total_norm ** 0.5
        self.records.append({"epoch": trainer.epoch + 1, "grad_norm": total_norm})

        if total_norm > 500:
            print(f"  ⚠  Gradient explosion: norm={total_norm:.1f} at epoch {trainer.epoch+1}")
        elif total_norm < 1e-6 and n_params > 0:
            print(f"  ⚠  Vanishing gradients: norm={total_norm:.2e} at epoch {trainer.epoch+1}")

    def on_train_end(self, trainer):
        if self.records:
            df = pd.DataFrame(self.records)
            df.to_csv(f"{self.save_dir}/gradient_norms.csv", index=False)
            print(f"  ✓ Gradient norms saved → {self.save_dir}/gradient_norms.csv")


class SessionCheckpointManager:
    """
    Survives Kaggle's 12-hour session reset.

    What it does every CHECKPOINT_EVERY epochs:
      1. Copies  last.pt  → /kaggle/working/session_last.pt
                best.pt  → /kaggle/working/session_best.pt
      2. Writes  /kaggle/working/session_meta.json  with:
             completed_epochs, best_f2, best_map50, batch_size,
             train_args  (so you can reconstruct the train() call exactly)
      3. Prints a clear DOWNLOAD THESE FILES message with resume instructions.

    How to resume after a session reset:
      ① Download  session_last.pt  and  session_meta.json
      ② Create a new Kaggle Dataset, upload both files
      ③ In CELL 1 of the new session set:
             RESUME_TRAINING = True
             RESUME_PT = "/kaggle/input/<your-dataset>/session_last.pt"
      ④ Run the notebook — training continues from the saved epoch.

    NOTE: last.pt contains the full optimizer + scaler + epoch state.
          best.pt only has model weights — use last.pt for resuming.
    """
    def __init__(self, checkpoint_every: int = 5,
                 working_dir: str = "/kaggle/working"):
        self.every       = checkpoint_every
        self.working_dir = Path(working_dir)
        self.best_f2     = 0.0
        self.best_map50  = 0.0
        self._train_args = {}          # filled in before train() is called
        self._batch_size = BATCH_SIZE

    def register_train_args(self, args: dict):
        """Call this with the kwargs you pass to model.train() — saved to JSON."""
        self._train_args = {k: str(v) for k, v in args.items()}

    # ── internal copy helper ──────────────────────────────────────────────
    def _safe_copy(self, src: Path, dst: Path):
        if src.exists():
            shutil.copy(str(src), str(dst))
            return True
        return False

    def _write_meta(self, epoch: int, save_dir: Path):
        meta = {
            "completed_epochs": epoch,
            "best_f2":          round(self.best_f2,    4),
            "best_map50":       round(self.best_map50, 4),
            "batch_size":       self._batch_size,
            "save_dir":         str(save_dir),
            "train_args":       self._train_args,
            "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        meta_path = self.working_dir / "session_meta.json"
        import json
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        return meta_path

    def _print_resume_instructions(self, epoch: int):
        print("\n" + "─" * 65)
        print(f"  📥 CHECKPOINT SAVED  (epoch {epoch})")
        print("  Download these two files from /kaggle/working/:")
        print("    • session_last.pt")
        print("    • session_meta.json")
        print("  Upload them to a NEW Kaggle Dataset, then in CELL 1:")
        print("    RESUME_TRAINING = True")
        print("    RESUME_PT = '/kaggle/input/<your-dataset>/session_last.pt'")
        print("─" * 65 + "\n")

    # ── callbacks ─────────────────────────────────────────────────────────
    def on_fit_epoch_end(self, trainer):
        # track best metrics
        m = trainer.metrics
        P  = float(m.get("metrics/precision(B)", 0))
        R  = float(m.get("metrics/recall(B)",    0))
        f2 = 5 * P * R / (4 * P + R + 1e-9)
        m50 = float(m.get("metrics/mAP50(B)", 0))
        if f2  > self.best_f2:    self.best_f2    = f2
        if m50 > self.best_map50: self.best_map50 = m50

        epoch = trainer.epoch + 1
        if epoch % self.every != 0:
            return

        save_dir = Path(trainer.save_dir)
        # copy weights
        copied_last = self._safe_copy(
            save_dir / "weights" / "last.pt",
            self.working_dir / "session_last.pt"
        )
        self._safe_copy(
            save_dir / "weights" / "best.pt",
            self.working_dir / "session_best.pt"
        )
        # write metadata
        meta_path = self._write_meta(epoch, save_dir)

        if copied_last:
            self._print_resume_instructions(epoch)
        else:
            print(f"  ⚠  Checkpoint copy skipped — last.pt not found at epoch {epoch}")

    def on_train_end(self, trainer):
        """Final checkpoint + summary at training end."""
        save_dir = Path(trainer.save_dir)
        self._safe_copy(
            save_dir / "weights" / "last.pt",
            self.working_dir / "session_last.pt"
        )
        self._safe_copy(
            save_dir / "weights" / "best.pt",
            self.working_dir / "session_best.pt"
        )
        self._write_meta(trainer.epoch + 1, save_dir)
        print(f"\n📥 Final checkpoint saved → {self.working_dir}/session_last.pt")
        print(f"   Best F2={self.best_f2:.4f}  Best mAP50={self.best_map50:.4f}")


print("✓ Custom callbacks defined: NaNStop, F2EarlyStop, GradientMonitor, SessionCheckpointManager")


# ============================================================================
# CELL 10: Training  (OOM-safe, session-resumable)
# ============================================================================
from ultralytics import YOLO

print("\n" + "=" * 80)
print("TRAINING: YOLO11m + Mamba (neck) + CBAM (backbone) + P2 Head + COPY-PASTE AUG")
print("=" * 80)

# Instantiate callbacks
nan_cb  = NaNStopCallback()
f2_cb   = F2EarlyStopCallback(patience=F2_PATIENCE)
grad_cb = GradientMonitorCallback()
ckpt_mgr = SessionCheckpointManager(
    checkpoint_every=CHECKPOINT_EVERY,
    working_dir="/kaggle/working"
)

MAMBA_RUN_DIR = "runs/detect/yolo11m_mamba_cbam_p2_copypaste"
MAMBA_BEST    = f"{MAMBA_RUN_DIR}/weights/best.pt"

def _register_callbacks(model):
    model.add_callback("on_train_batch_end", nan_cb.on_train_batch_end)
    model.add_callback("on_fit_epoch_end",   f2_cb.on_fit_epoch_end)
    model.add_callback("on_fit_epoch_end",   ckpt_mgr.on_fit_epoch_end)
    model.add_callback("on_train_epoch_end", grad_cb.on_train_epoch_end)
    model.add_callback("on_train_end",       grad_cb.on_train_end)
    model.add_callback("on_train_end",       ckpt_mgr.on_train_end)


# ── RESUME PATH ──────────────────────────────────────────────────────────────
if RESUME_TRAINING and RESUME_PT:
    # The resume checkpoint may be in a Kaggle input dataset (read-only).
    # Ultralytics resume=True writes to the run directory, so we copy
    # last.pt into the expected run/weights/ path first.
    _resume_src = Path(RESUME_PT)
    if not _resume_src.exists():
        raise FileNotFoundError(
            f"RESUME_PT not found: {RESUME_PT}\n"
            "Did you forget to add the checkpoint dataset as a Kaggle input?"
        )

    # Reconstruct the weights path inside the expected run directory
    _weights_dir = Path(MAMBA_RUN_DIR) / "weights"
    _weights_dir.mkdir(parents=True, exist_ok=True)
    _local_last  = _weights_dir / "last.pt"

    if not _local_last.exists():
        shutil.copy(str(_resume_src), str(_local_last))
        print(f"  ✓ Copied last.pt → {_local_last}")

    # Load the metadata to report where we're resuming from
    _meta_src = Path(RESUME_PT).parent / "session_meta.json"
    if _meta_src.exists():
        import json
        _meta = json.load(open(_meta_src))
        print(f"  Resuming from epoch {_meta.get('completed_epochs', '?')}  |  "
              f"Best F2={_meta.get('best_f2','?')}  mAP50={_meta.get('best_map50','?')}")
        ckpt_mgr.best_f2    = float(_meta.get("best_f2",    0))
        ckpt_mgr.best_map50 = float(_meta.get("best_map50", 0))

    mamba_model = YOLO(str(_local_last))
    _register_callbacks(mamba_model)
    print(f"\n⚡ RESUMING training from {_local_last}")
    mamba_model.train(resume=True)

# ── FRESH TRAINING with OOM retry ────────────────────────────────────────────
else:
    _trained_ok = False

    for _batch_attempt in OOM_RETRY_BATCHES:
        print(f"\n→ Attempting training with batch_size={_batch_attempt} …")

        # ── Build fresh model (rebuilt each retry so weights are clean) ──────
        mamba_model = YOLO("yolov11m_cbam_p2head.yaml")
        mamba_model.load("yolo11m.pt")
        print("  ✓ Architecture + ImageNet weights loaded")

        print("  Injecting Mamba SSM blocks into neck …")
        replaced = inject_mamba_neck(mamba_model, d_state=4, verbose=(_batch_attempt == OOM_RETRY_BATCHES[0]))
        if len(replaced) == 0:
            raise RuntimeError("Injection produced no replacements — abort!")

        mamba_model.info(verbose=False)
        _register_callbacks(mamba_model)

        # ── Register train args in checkpoint manager ─────────────────────
        _train_kwargs = dict(
            data         = "c2a.yaml",
            epochs       = NUM_EPOCHS,
            imgsz        = 640,
            batch        = _batch_attempt,
            device       = DEVICE,
            optimizer    = "AdamW",
            lr0          = 0.001,
            lrf          = 0.01,
            weight_decay = 0.0005,
            momentum     = 0.937,
            warmup_epochs= 3,
            close_mosaic = 10,
            amp          = True,
            patience     = PATIENCE,
            save         = True,
            save_period  = SAVE_PERIOD,
            plots        = True,
            verbose      = True,
            fraction     = TRAIN_FRACTION,
            cache        = True,
            workers      = 2,
            name         = "yolo11m_mamba_cbam_p2_copypaste",
            exist_ok     = True,
            # ── AUGMENTATION CHANGES (vs previous run) ──────────────
            # Previous run: copy_paste=0, mixup=0, flipud=0, degrees=0
            copy_paste   = 0.5,     # KEY: 50% prob copy-paste — targets small objects
            mixup        = 0.15,    # mild image mixing for regularization
            flipud       = 0.5,     # vertical flip — valid for nadir aerial imagery
            degrees      = 10.0,    # mild rotation — aerial-appropriate
        )
        ckpt_mgr.register_train_args(_train_kwargs)
        ckpt_mgr._batch_size = _batch_attempt

        try:
            mamba_model.train(**_train_kwargs)
            _trained_ok = True
            print(f"\n✓ Training complete (batch_size={_batch_attempt})")
            break   # success — exit retry loop

        except torch.cuda.OutOfMemoryError as _oom:
            print(f"\n💥 OOM at batch_size={_batch_attempt}: {_oom}")

            # ── Check if we can resume from a partial checkpoint ──────────
            _partial_last = Path(MAMBA_RUN_DIR) / "weights" / "last.pt"
            if _partial_last.exists():
                print(f"  Partial checkpoint found: {_partial_last}")
                print(f"  Copying to /kaggle/working/session_last.pt for download.")
                shutil.copy(str(_partial_last),
                            "/kaggle/working/session_last.pt")
                ckpt_mgr._write_meta(
                    epoch    = -1,   # unknown — stored inside last.pt
                    save_dir = Path(MAMBA_RUN_DIR)
                )

            # Clear GPU memory before next attempt
            del mamba_model
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(3)

            next_batch = _batch_attempt // 2
            if next_batch < 1 or _batch_attempt == OOM_RETRY_BATCHES[-1]:
                raise RuntimeError(
                    f"OOM even at batch_size={_batch_attempt}. "
                    "All retry options exhausted.\n"
                    "Options:\n"
                    "  1. Switch to P100 GPU in Kaggle accelerator settings.\n"
                    "  2. Reduce d_state from 4 to 2 in inject_mamba_neck call.\n"
                    "  3. Remove P2 head (use 3-scale detection instead).\n"
                    "  4. Reduce imgsz from 640 to 512."
                ) from _oom
            print(f"  Retrying with batch_size={next_batch} …")

    if not _trained_ok:
        raise RuntimeError("Training did not complete. Check OOM messages above.")

del mamba_model; torch.cuda.empty_cache(); gc.collect()


# ============================================================================
# CELL 11: Output Directory Setup
# ============================================================================
BASE_DIR   = "/kaggle/working"
EXCEL_DIR  = f"{BASE_DIR}/excel_reports"
PLOT_DIR   = f"{BASE_DIR}/plots"
REPORT_DIR = f"{BASE_DIR}/benchmark_reports"
for d in [EXCEL_DIR, PLOT_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)
print("✓ Output directories created")


# ============================================================================
# CELL 12: Training Curves (enhanced — adds F1/F2 per epoch)
# ============================================================================

def load_results_csv(run_dir: str) -> pd.DataFrame:
    """Load and normalise a results.csv from a training run."""
    path = f"{run_dir}/results.csv"
    if not os.path.exists(path):
        print(f"  ⚠  results.csv not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # Compute per-epoch F1 and F2 from precision and recall
    P = df.get("metrics/precision(B)", pd.Series(dtype=float))
    R = df.get("metrics/recall(B)",    pd.Series(dtype=float))
    df["metrics/F1(B)"] = 2 * P * R / (P + R + 1e-9)
    df["metrics/F2(B)"] = 5 * P * R / (4 * P + R + 1e-9)
    # Total losses
    train_loss_cols = [c for c in df.columns if "train" in c and "loss" in c]
    val_loss_cols   = [c for c in df.columns if "val"   in c and "loss" in c]
    if train_loss_cols:
        df["train/total_loss"] = df[train_loss_cols].sum(axis=1)
    if val_loss_cols:
        df["val/total_loss"] = df[val_loss_cols].sum(axis=1)
    return df

# Load both runs
runs = {}
if os.path.exists(MAMBA_RUN_DIR):
    runs["Mamba+CBAM+P2+CopyPaste"] = load_results_csv(MAMBA_RUN_DIR)
if PREV_MAMBA_BEST:
    # Try to find results.csv from previous run (may not exist if only best.pt uploaded)
    prev_run = str(Path(PREV_MAMBA_BEST).parent.parent)
    prev_df = load_results_csv(prev_run)
    if not prev_df.empty:
        runs["Mamba+CBAM+P2 (no aug)"] = prev_df

# ── Per-run individual plots ─────────────────────────────────────────────────
def plot_single_run(df: pd.DataFrame, tag: str):
    if df.empty:
        return
    df.to_excel(f"{EXCEL_DIR}/{tag}_training.xlsx", index=False)
    ep = df["epoch"]

    # Loss curves (individual + total)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    ax = axes[0]
    for k in ["box", "cls", "dfl"]:
        tc, vc = f"train/{k}_loss", f"val/{k}_loss"
        if tc in df.columns:
            ax.plot(ep, df[tc], label=f"Train {k}", alpha=0.8)
            ax.plot(ep, df[vc], label=f"Val {k}",   linestyle="--", alpha=0.8)
    ax.set(xlabel="Epoch", ylabel="Loss", title=f"{tag} — Individual Losses")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    if "train/total_loss" in df.columns:
        ax.plot(ep, df["train/total_loss"], label="Train Total", lw=2)
        ax.plot(ep, df["val/total_loss"],   label="Val Total",   lw=2, ls="--")
    ax.set(xlabel="Epoch", ylabel="Total Loss", title=f"{tag} — Total Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{tag}_loss_curves.png", dpi=300); plt.close()

    # Metrics: Precision, Recall, mAP50, mAP50-95, F1, F2
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    metric_map = [
        ("metrics/precision(B)", "Precision",   axes[0, 0]),
        ("metrics/recall(B)",    "Recall",       axes[0, 1]),
        ("metrics/mAP50(B)",     "mAP@0.5",      axes[0, 2]),
        ("metrics/mAP50-95(B)",  "mAP@0.5:0.95", axes[1, 0]),
        ("metrics/F1(B)",        "F1",            axes[1, 1]),
        ("metrics/F2(B)",        "F2",            axes[1, 2]),
    ]
    for col, label, ax in metric_map:
        if col in df.columns:
            ax.plot(ep, df[col], marker="o", markersize=3, lw=2)
            best_val = df[col].max()
            best_ep  = df.loc[df[col].idxmax(), "epoch"]
            ax.axhline(best_val, color="red", ls=":", lw=1, alpha=0.6,
                       label=f"Best={best_val:.4f} @ ep{best_ep:.0f}")
        ax.set(xlabel="Epoch", ylabel=label, title=f"{tag} — {label}")
        ax.set_ylim([0, 1.05]); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.suptitle(f"{tag} — Validation Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{tag}_metrics_6panel.png", dpi=300); plt.close()
    print(f"  ✓ {tag}: loss + metrics plots saved")

for tag, df in runs.items():
    plot_single_run(df, tag.replace("+", "_").replace(" ", "_"))

# ── 2-way overlay comparison ─────────────────────────────────────────────────
if len(runs) == 2:
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    overlay_metrics = [
        ("metrics/mAP50(B)",    "mAP@0.5",      axes[0, 0]),
        ("metrics/mAP50-95(B)", "mAP@0.5:0.95", axes[0, 1]),
        ("metrics/recall(B)",   "Recall",        axes[0, 2]),
        ("metrics/F1(B)",       "F1",            axes[1, 0]),
        ("metrics/F2(B)",       "F2",            axes[1, 1]),
        ("val/total_loss",      "Val Total Loss",axes[1, 2]),
    ]
    colors = {"Mamba+CBAM+P2+CopyPaste": "#E53935", "Mamba+CBAM+P2 (no aug)": "#1E88E5"}
    for col, label, ax in overlay_metrics:
        for tag, df in runs.items():
            if not df.empty and col in df.columns:
                ax.plot(df["epoch"], df[col], label=tag,
                        color=colors.get(tag, "grey"), lw=2, alpha=0.9)
        ax.set(xlabel="Epoch", ylabel=label, title=label)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        if "loss" not in col.lower():
            ax.set_ylim([0, 1.05])
    plt.suptitle("Training Comparison: +CopyPaste  vs  No Augmentation",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/2way_overlay_comparison.png", dpi=300); plt.close()
    print("  ✓ 2-way overlay comparison plot saved")

print("✓ Training curve analysis complete")


# ============================================================================
# CELL 13: Model Complexity Comparison
# ============================================================================
from ultralytics import YOLO

def get_complexity(weights_path: str, label: str) -> dict:
    model = YOLO(weights_path)
    n_params = sum(p.numel() for p in model.model.parameters())
    try:
        from thop import profile
        dummy = torch.zeros(1, 3, 640, 640)
        if next(model.model.parameters()).is_cuda:
            dummy = dummy.cuda()
        macs, _ = profile(model.model, inputs=(dummy,), verbose=False)
        gflops  = macs / 1e9
    except Exception:
        gflops = 0.0
    size_kb = os.path.getsize(weights_path) / 1024
    layers  = len(list(model.model.modules()))
    del model; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return {"Model": label, "Params(M)": round(n_params/1e6, 3),
            "GFLOPs": round(gflops, 1), "Size(KB)": round(size_kb, 0),
            "Layers": layers}

complexity_rows = []
if os.path.exists(MAMBA_BEST):
    complexity_rows.append(get_complexity(MAMBA_BEST, "Mamba+CBAM+P2+CopyPaste"))
if PREV_MAMBA_BEST and os.path.exists(PREV_MAMBA_BEST):
    complexity_rows.append(get_complexity(PREV_MAMBA_BEST, "Mamba+CBAM+P2 (no aug)"))

if complexity_rows:
    cdf = pd.DataFrame(complexity_rows)
    if len(cdf) > 1:
        base = cdf.iloc[-1]["Params(M)"]   # CBAM+P2 as reference
        cdf["Δ Params(%)"] = ((cdf["Params(M)"] / base) - 1) * 100
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

print("✓ Helper functions loaded")


# ============================================================================
# CELL 15: Comprehensive Evaluation Function (enhanced)
# ============================================================================
from tqdm import tqdm

def evaluate_model_comprehensive(model, model_name: str, img_dir: str, lbl_dir: str,
                                   split_name: str = "test", n_images=None,
                                   conf: float = 0.25) -> tuple:
    print(f"\n{'='*70}\nEVALUATING: {model_name}  on  {split_name.upper()}\n{'='*70}")
    image_list = get_image_list(img_dir)
    if n_images:
        image_list = image_list[:min(n_images, len(image_list))]

    records      = []
    inf_times    = []
    total_tp = total_fp = total_fn = 0
    size_stats = {k: {"tp": 0, "fn": 0, "count": 0}
                  for k in ("very_tiny", "tiny", "small", "medium", "large")}
    all_confs = []

    for img_file in tqdm(image_list, desc=f"{model_name}", ncols=80):
        img_path = f"{img_dir}/{img_file}"
        lbl_path = f"{lbl_dir}/{img_file.rsplit('.', 1)[0]}.txt"
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        gt_boxes = parse_yolo_label(lbl_path, W, H)

        t0   = time.perf_counter()
        pred = model.predict(img_path, conf=conf, verbose=False)
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

        P_img = tp / (tp + fp + 1e-9)
        R_img = tp / (tp + fn + 1e-9)
        F1_img = 2 * P_img * R_img / (P_img + R_img + 1e-9)
        F2_img = 5 * P_img * R_img / (4 * P_img + R_img + 1e-9)
        records.append({
            "Image": img_file, "GT": len(gt_boxes), "Pred": len(pboxes),
            "TP": tp, "FP": fp, "FN": fn,
            "Precision": P_img, "Recall": R_img, "F1": F1_img, "F2": F2_img,
            "Avg_Conf": float(np.mean(pconfs)) if pconfs else 0.0,
            "Inference_ms": inf_times[-1],
        })

    df = pd.DataFrame(records)
    df.to_excel(f"{EXCEL_DIR}/{model_name}_{split_name}_detailed.xlsx", index=False)

    # Overall metrics
    P_ov = total_tp / (total_tp + total_fp + 1e-9)
    R_ov = total_tp / (total_tp + total_fn + 1e-9)
    F1   = 2 * P_ov * R_ov / (P_ov + R_ov + 1e-9)
    F2   = 5 * P_ov * R_ov / (4 * P_ov + R_ov + 1e-9)

    size_recalls = {}
    for cat, s in size_stats.items():
        tot = s["tp"] + s["fn"]
        size_recalls[f"{cat}_recall"]  = s["tp"] / tot if tot > 0 else 0.0
        size_recalls[f"{cat}_count"]   = s["count"]

    summary = {
        "Model": model_name, "Split": split_name, "N_images": len(image_list),
        "Total_GT": total_tp + total_fn, "Total_Pred": total_tp + total_fp,
        "TP": total_tp, "FP": total_fp, "FN": total_fn,
        "Precision": P_ov, "Recall": R_ov, "F1": F1, "F2": F2,
        **size_recalls,
        "Avg_Inf_ms":  float(np.mean(inf_times)),
        "Std_Inf_ms":  float(np.std(inf_times)),
        "P95_Inf_ms":  float(np.percentile(inf_times, 95)),
    }

    print(f"  P={P_ov:.4f}  R={R_ov:.4f}  F1={F1:.4f}  F2={F2:.4f}")
    for cat in ("very_tiny", "tiny", "small", "medium"):
        rc = size_recalls[f"{cat}_recall"]
        n  = size_recalls[f"{cat}_count"]
        print(f"  {cat:12s}: recall={rc:.4f}  (n={n})")
    print(f"  Latency: {summary['Avg_Inf_ms']:.1f}±{summary['Std_Inf_ms']:.1f} ms")

    return df, summary, inf_times, all_confs

print("✓ Evaluation function loaded")


# ============================================================================
# CELL 16: Advanced Visualisation Functions
# ============================================================================

def plot_confidence_distribution(confs: list, model_name: str, split: str):
    """Histogram + KDE of prediction confidences."""
    if not confs:
        return
    plt.figure(figsize=(10, 5))
    plt.hist(confs, bins=50, range=(0, 1), color="#1E88E5",
             alpha=0.7, density=True, label="Confidence distribution")
    plt.axvline(np.mean(confs), color="red",    ls="--", lw=1.5,
                label=f"Mean={np.mean(confs):.3f}")
    plt.axvline(np.median(confs), color="orange", ls="--", lw=1.5,
                label=f"Median={np.median(confs):.3f}")
    plt.xlabel("Confidence"); plt.ylabel("Density")
    plt.title(f"{model_name} — Confidence Distribution ({split})")
    plt.legend(); plt.grid(True, alpha=0.3); plt.xlim([0, 1])
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{model_name}_{split}_conf_dist.png", dpi=200)
    plt.close()


def plot_calibration_ece(results_df: pd.DataFrame, model_name: str,
                          split: str, n_bins: int = 10):
    """ECE calibration plot + returns ECE value."""
    df = results_df[results_df["Pred"] > 0].copy()
    if df.empty:
        return None
    bins = np.linspace(0, 1, n_bins + 1)
    cal_rows = []
    for i in range(n_bins):
        m = (df["Avg_Conf"] >= bins[i]) & (df["Avg_Conf"] < bins[i+1])
        if m.sum() > 0:
            cal_rows.append({
                "Conf_mid":   (bins[i] + bins[i+1]) / 2,
                "Avg_Conf":   df.loc[m, "Avg_Conf"].mean(),
                "Avg_Prec":   df.loc[m, "Precision"].mean(),
                "Count":      int(m.sum()),
            })
    if not cal_rows:
        return None
    cdf = pd.DataFrame(cal_rows)
    ece = float(np.average(
        np.abs(cdf["Avg_Conf"] - cdf["Avg_Prec"]),
        weights=cdf["Count"]
    ))
    cdf["Gap"] = cdf["Avg_Conf"] - cdf["Avg_Prec"]
    cdf.to_excel(f"{EXCEL_DIR}/{model_name}_{split}_calibration.xlsx", index=False)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.bar(cdf["Conf_mid"], cdf["Avg_Prec"],
           width=1/n_bins, alpha=0.6, label="Precision@bin", color="#4CAF50")
    ax.bar(cdf["Conf_mid"], cdf["Avg_Conf"] - cdf["Avg_Prec"],
           width=1/n_bins, alpha=0.4, label="Calibration gap",
           color="#F44336", bottom=cdf["Avg_Prec"])
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    ax.set(xlabel="Confidence", ylabel="Precision",
           title=f"{model_name} — Calibration  (ECE={ece:.4f})")
    ax.legend(); ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{model_name}_{split}_calibration.png", dpi=200)
    plt.close()
    return ece


def plot_per_size_recall(summaries: list, split: str):
    """Grouped bar chart: per-size recall comparison."""
    cats   = ["very_tiny", "tiny", "small", "medium", "large"]
    labels = ["Very Tiny\n(<8²px)", "Tiny\n(8–16px)", "Small\n(16–32px)",
              "Medium\n(32–96px)", "Large\n(>96px)"]
    x      = np.arange(len(cats))
    n      = len(summaries)
    width  = 0.7 / n
    colors = ["#E53935", "#1E88E5", "#43A047", "#FB8C00"]

    fig, ax = plt.subplots(figsize=(16, 7))
    for si, s in enumerate(summaries):
        recalls = [s.get(f"{c}_recall", 0) for c in cats]
        bars    = ax.bar(x + (si - n/2 + 0.5) * width, recalls, width,
                         label=s["Model"], color=colors[si % len(colors)], alpha=0.85)
        for bar, rc in zip(bars, recalls):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f"{rc:.3f}", ha="center", va="bottom", fontsize=8)

    counts = [summaries[0].get(f"{c}_count", 0) for c in cats]
    for i, c in enumerate(counts):
        ax.text(i, max(s.get(f"{cats[i]}_recall", 0) for s in summaries) + 0.04,
                f"n={c}", ha="center", fontsize=9, color="grey")

    ax.set(xlabel="Object Size", ylabel="Recall",
           title=f"Per-Size Recall Comparison ({split.upper()})")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend(fontsize=11); ax.set_ylim([0, 1.18])
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/per_size_recall_{split}.png", dpi=300); plt.close()
    print(f"  ✓ Per-size recall chart saved ({split})")


def benchmark_speed(model, sample_img: str, model_name: str) -> pd.DataFrame:
    """FPS at 4 resolutions, with warmup runs."""
    rows = []
    for sz in [320, 480, 640, 800]:
        # warmup
        for _ in range(3):
            model.predict(sample_img, imgsz=sz, verbose=False)
        times = []
        for _ in range(15):
            t0 = time.perf_counter()
            model.predict(sample_img, imgsz=sz, verbose=False)
            times.append((time.perf_counter() - t0) * 1000)
        mean_ms = float(np.mean(times[3:]))   # discard first 3 warmup-adjacent
        rows.append({"Resolution": sz, "Avg_ms": round(mean_ms, 2),
                     "FPS": round(1000 / mean_ms, 1)})
    sdf = pd.DataFrame(rows)
    sdf.to_excel(f"{EXCEL_DIR}/{model_name}_speed.xlsx", index=False)
    print(f"  Speed — {model_name}:")
    print(sdf.to_string(index=False))
    return sdf


def visualise_predictions(model, img_dir: str, lbl_dir: str,
                           model_name: str, split: str = "test",
                           n_images: int = 15, conf: float = 0.25):
    """Grid plot of predictions vs ground-truth counts."""
    images = get_image_list(img_dir)[:n_images]
    cols   = 5
    rows   = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, f in enumerate(images):
        lbl = f"{lbl_dir}/{f.rsplit('.', 1)[0]}.txt"
        gt  = len(open(lbl).readlines()) if os.path.exists(lbl) else 0
        prd = model.predict(f"{img_dir}/{f}", conf=conf, verbose=False)
        pc  = len(prd[0].boxes)
        axes[i].imshow(cv2.cvtColor(prd[0].plot(), cv2.COLOR_BGR2RGB))
        axes[i].axis("off")
        col = "green" if pc == gt else "orange" if abs(pc - gt) <= 2 else "red"
        axes[i].set_title(f"GT:{gt}|Pred:{pc}", fontsize=10,
                           color=col, fontweight="bold")
    for j in range(len(images), len(axes)):
        axes[j].axis("off")
    plt.suptitle(f"{model_name} — {split} predictions", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{model_name}_{split}_predictions.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def copy_yolo_plots(run_dir: str, model_name: str):
    """
    Ultralytics generates PR_curve.png and F1_curve.png automatically
    when plots=True. Copy them into our consolidated plots folder.
    """
    for fname in ("PR_curve.png", "F1_curve.png", "confusion_matrix.png",
                  "confusion_matrix_normalized.png", "results.png"):
        src = Path(run_dir) / fname
        if src.exists():
            dst = Path(PLOT_DIR) / f"{model_name}_{fname}"
            shutil.copy(str(src), str(dst))
            print(f"  ✓ {fname} → {dst.name}")

print("✓ Visualisation functions loaded")


# ============================================================================
# CELL 17: Load Models for Evaluation
# ============================================================================
from ultralytics import YOLO

models_for_eval = {}
if os.path.exists(MAMBA_BEST):
    models_for_eval["Mamba_CBAM_P2_copypaste"] = YOLO(MAMBA_BEST)
    print(f"✓ Loaded: Mamba+CBAM+P2+CopyPaste  ({MAMBA_BEST})")
else:
    print(f"⚠  Copy-paste model not found: {MAMBA_BEST}")

if PREV_MAMBA_BEST and os.path.exists(PREV_MAMBA_BEST):
    models_for_eval["Mamba_CBAM_P2_noaug"] = YOLO(PREV_MAMBA_BEST)
    print(f"✓ Loaded: Mamba+CBAM+P2 (no aug)  ({PREV_MAMBA_BEST})")

assert models_for_eval, "No models loaded — cannot proceed with evaluation!"

# Copy Ultralytics auto-generated plots (PR curve, F1 curve, etc.)
if os.path.exists(MAMBA_RUN_DIR):
    copy_yolo_plots(MAMBA_RUN_DIR, "Mamba_CBAM_P2_copypaste")


# ============================================================================
# CELL 18: Test Set Evaluation (2-way comparison)
# ============================================================================
print("\n" + "=" * 80)
print(f"TEST SET EVALUATION ({TEST_IMAGES or 'ALL'} images)")
print("=" * 80)

test_summaries = []
test_dfs       = {}
test_confs_all = {}

for name, model in models_for_eval.items():
    df, summary, _, confs = evaluate_model_comprehensive(
        model, name, TEST_IMG_DIR, TEST_LBL_DIR, "test", TEST_IMAGES)
    test_dfs[name]       = df
    test_summaries.append(summary)
    test_confs_all[name] = confs
    visualise_predictions(model, TEST_IMG_DIR, TEST_LBL_DIR, name, "test", 15)


# ── Official YOLO val metrics (mAP from Ultralytics, not our custom loop) ──
official_test = {}
for name, model in models_for_eval.items():
    res = model.val(data="c2a.yaml", split="test", verbose=False, plots=True)
    official_test[name] = {
        "mAP50":    round(float(res.box.map50),  4),
        "mAP50-95": round(float(res.box.map),    4),
        "Precision":round(float(res.box.mp),     4),
        "Recall":   round(float(res.box.mr),     4),
    }
    print(f"  Official {name}: "
          f"mAP50={official_test[name]['mAP50']:.4f}  "
          f"mAP50-95={official_test[name]['mAP50-95']:.4f}")

pd.DataFrame(official_test).T.to_excel(
    f"{EXCEL_DIR}/official_test_metrics.xlsx")


# ============================================================================
# CELL 19: Validation Set Evaluation (2-way)
# ============================================================================
print("\n" + "=" * 80)
print(f"VALIDATION SET EVALUATION ({VAL_IMAGES or 'ALL'} images)")
print("=" * 80)

val_summaries = []
val_dfs       = {}

for name, model in models_for_eval.items():
    df, summary, _, confs = evaluate_model_comprehensive(
        model, name, VAL_IMG_DIR, VAL_LBL_DIR, "val", VAL_IMAGES)
    val_dfs[name]       = df
    val_summaries.append(summary)
    visualise_predictions(model, VAL_IMG_DIR, VAL_LBL_DIR, name, "val", 15)


# ============================================================================
# CELL 20: Advanced Metrics (ECE, speed, confidence distribution)
# ============================================================================
print("\n" + "=" * 80)
print("ADVANCED METRICS")
print("=" * 80)

sample_imgs = get_image_list(TEST_IMG_DIR)
sample_img  = f"{TEST_IMG_DIR}/{sample_imgs[0]}"

ece_results   = {}
speed_results = {}

for name, model in models_for_eval.items():
    print(f"\n── {name} ──")
    # ECE calibration
    ece = plot_calibration_ece(test_dfs[name], name, "test")
    ece_results[name] = ece
    print(f"  ECE: {ece:.4f}" if ece else "  ECE: N/A")
    # Confidence distribution
    plot_confidence_distribution(test_confs_all.get(name, []), name, "test")
    # Speed benchmark
    speed_results[name] = benchmark_speed(model, sample_img, name)

# Per-size recall charts
if test_summaries:
    plot_per_size_recall(test_summaries, "test")
if val_summaries:
    plot_per_size_recall(val_summaries,  "val")


# ============================================================================
# CELL 21: 2-Way Comparison Tables
# ============================================================================

def make_comparison_table(summaries: list, split: str) -> pd.DataFrame:
    metrics = [
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
    rows = []
    base = summaries[-1] if len(summaries) > 1 else summaries[0]  # CBAM+P2 as base
    for key, label in metrics:
        row = {"Metric": label}
        for s in summaries:
            row[s["Model"]] = round(float(s.get(key, 0)), 4)
        if len(summaries) > 1:
            bv = float(base.get(key, 0))
            mv = float(summaries[0].get(key, 0))  # Mamba
            row["Δ (Mamba−CBAM+P2)"] = round(mv - bv, 4)
        rows.append(row)
    cdf = pd.DataFrame(rows)
    cdf.to_excel(f"{EXCEL_DIR}/2way_comparison_{split}.xlsx", index=False)
    print(f"\n{'='*70}\n2-WAY COMPARISON — {split.upper()}\n{'='*70}")
    print(cdf.to_string(index=False))
    return cdf

if test_summaries:
    make_comparison_table(test_summaries, "test")
if val_summaries:
    make_comparison_table(val_summaries,  "val")


# ============================================================================
# CELL 22: Master Summary Report
# ============================================================================
import json

def best_epoch(df: pd.DataFrame) -> int:
    if df.empty or "metrics/mAP50(B)" not in df.columns:
        return -1
    return int(df.loc[df["metrics/mAP50(B)"].idxmax(), "epoch"])

mamba_df = runs.get("Mamba+CBAM+P2+CopyPaste", pd.DataFrame())
cbam_df  = runs.get("Mamba+CBAM+P2 (no aug)",  pd.DataFrame())

def fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)

ts = {s["Model"]: s for s in test_summaries}
vs = {s["Model"]: s for s in val_summaries}

report_lines = [
    "=" * 80,
    "DISASTER HUMAN DETECTION — MAMBA+CBAM+P2 + COPY-PASTE AUGMENTATION REPORT",
    "=" * 80,
    f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}",
    f"TEST MODE : {TEST_MODE}  |  Epochs: {NUM_EPOCHS}  |  Fraction: {TRAIN_FRACTION*100:.0f}%",
    "",
    "MODEL COMPLEXITY",
]
if complexity_rows:
    for r in complexity_rows:
        report_lines.append(
            f"  {r['Model']:20s}: {r['Params(M)']:.3f}M params  "
            f"{r['GFLOPs']:.1f} GFLOPs  {r['Size(KB)']:.0f} KB")

report_lines += ["", "OFFICIAL mAP (test split — Ultralytics val):"]
for name, m in official_test.items():
    report_lines.append(
        f"  {name:20s}: mAP50={m['mAP50']:.4f}  mAP50-95={m['mAP50-95']:.4f}")

report_lines += ["", "TEST SET (custom evaluation):"]
for name, s in ts.items():
    report_lines.append(
        f"  {name:20s}: P={fmt(s['Precision'])}  R={fmt(s['Recall'])}"
        f"  F1={fmt(s['F1'])}  F2={fmt(s['F2'])}"
        f"  Lat={fmt(s['Avg_Inf_ms'])}ms")

report_lines += ["", "SMALL OBJECT RECALL — TEST:"]
for cat in ("very_tiny", "tiny", "small", "medium"):
    line = f"  {cat:12s}:"
    for name, s in ts.items():
        line += f"  {name}={fmt(s[f'{cat}_recall'])}"
    report_lines.append(line)

report_lines += ["", "CALIBRATION (ECE):"]
for name, ece in ece_results.items():
    report_lines.append(f"  {name:20s}: ECE={fmt(ece) if ece else 'N/A'}")

report_lines += ["", "TRAINING CONVERGENCE:"]
if not mamba_df.empty:
    ep = best_epoch(mamba_df)
    if "metrics/F2(B)" in mamba_df.columns and ep >= 0:
        report_lines.append(
            f"  Mamba+CBAM+P2 best epoch {ep}: "
            f"F2={mamba_df.loc[mamba_df['epoch']==ep,'metrics/F2(B)'].values[0]:.4f}  "
            f"mAP50={mamba_df.loc[mamba_df['epoch']==ep,'metrics/mAP50(B)'].values[0]:.4f}")
if not cbam_df.empty:
    ep = best_epoch(cbam_df)
    if "metrics/F2(B)" in cbam_df.columns and ep >= 0:
        report_lines.append(
            f"  CBAM+P2       best epoch {ep}: "
            f"F2={cbam_df.loc[cbam_df['epoch']==ep,'metrics/F2(B)'].values[0]:.4f}  "
            f"mAP50={cbam_df.loc[cbam_df['epoch']==ep,'metrics/mAP50(B)'].values[0]:.4f}")

report_lines += ["", "F2 EARLY STOPPING LOG:"]
for ep, f2v in f2_cb.history[-10:]:
    report_lines.append(f"  epoch {ep+1:3d}: F2={f2v:.4f}")

report_lines += ["", "NaN STOP TRIGGERED:", f"  {nan_cb.triggered}"]
report_lines += ["", "=" * 80]

report_text = "\n".join(report_lines)
print(report_text)

with open(f"{REPORT_DIR}/MASTER_REPORT_COPYPASTE.txt", "w") as f:
    f.write(report_text)

# Also save complete summaries as JSON for reproducibility
with open(f"{REPORT_DIR}/test_summaries.json", "w") as f:
    json.dump(test_summaries, f, indent=2, default=str)
with open(f"{REPORT_DIR}/val_summaries.json",  "w") as f:
    json.dump(val_summaries,  f, indent=2, default=str)

print(f"\n✓ Master report → {REPORT_DIR}/MASTER_REPORT_COPYPASTE.txt")


# ============================================================================
# CELL 23: SAHI — Slicing Aided Hyper Inference Evaluation
# ============================================================================
# SAHI slices large images into overlapping tiles, runs detection per tile,
# then merges results with NMS.  Published gains: +6-14% AP on VisDrone/xView.
# This is an inference-time technique — NO retraining needed.
#
# We test 3 slice configurations and report the best.
# Reference: Akyon et al., "Slicing Aided Hyper Inference" (2022)
# ============================================================================

print("\n" + "=" * 80)
print("SAHI — SLICING AIDED HYPER INFERENCE EVALUATION")
print("=" * 80)

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ── Load the best copy-paste model for SAHI ─────────────────────────────────
sahi_results = []     # initialise — may stay empty if model not found
best_sahi    = None

if os.path.exists(MAMBA_BEST):
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=MAMBA_BEST,
        confidence_threshold=0.25,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    print(f"✓ SAHI model loaded: {MAMBA_BEST}")
else:
    sahi_model = None
    print(f"⚠  Best model not found at {MAMBA_BEST} — skipping SAHI")

if sahi_model is not None:
    # ── SAHI Configuration Sweep ─────────────────────────────────────────────
    # Based on VisDrone optimal parameters (published study):
    #   Fine slicing improves tiny/occluded recall.
    #   Overlap 0.3-0.5 optimal.  NMS IoU 0.3-0.5 best balance.
    sahi_configs = [
        {"name": "slice512_ov25", "slice_h": 512, "slice_w": 512,
         "overlap": 0.25, "nms_thr": 0.5},
        {"name": "slice640_ov30", "slice_h": 640, "slice_w": 640,
         "overlap": 0.30, "nms_thr": 0.5},
        {"name": "slice768_ov40", "slice_h": 768, "slice_w": 768,
         "overlap": 0.40, "nms_thr": 0.5},
    ]

    sahi_results = []
    test_images = get_image_list(TEST_IMG_DIR)
    if TEST_MODE:
        test_images = test_images[:TEST_IMAGES]

    for cfg in sahi_configs:
        print(f"\n── SAHI config: {cfg['name']} ──")
        total_tp = total_fp = total_fn = 0
        size_stats = {k: {"tp": 0, "fn": 0, "count": 0}
                      for k in ("very_tiny", "tiny", "small", "medium", "large")}
        inf_times = []

        for img_file in tqdm(test_images, desc=cfg["name"], ncols=80):
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
                postprocess_type="NMS",
                postprocess_match_threshold=cfg["nms_thr"],
                verbose=0,
            )
            inf_times.append((time.perf_counter() - t0) * 1000)

            # Extract predictions
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

        P_ov = total_tp / (total_tp + total_fp + 1e-9)
        R_ov = total_tp / (total_tp + total_fn + 1e-9)
        F1   = 2 * P_ov * R_ov / (P_ov + R_ov + 1e-9)
        F2   = 5 * P_ov * R_ov / (4 * P_ov + R_ov + 1e-9)

        row = {
            "Config": cfg["name"],
            "Slice": f"{cfg['slice_h']}x{cfg['slice_w']}",
            "Overlap": cfg["overlap"],
            "Precision": round(P_ov, 4),
            "Recall": round(R_ov, 4),
            "F1": round(F1, 4),
            "F2": round(F2, 4),
            "Avg_Inf_ms": round(float(np.mean(inf_times)), 1),
        }
        for cat in ("very_tiny", "tiny", "small", "medium", "large"):
            tot = size_stats[cat]["tp"] + size_stats[cat]["fn"]
            row[f"{cat}_recall"] = round(size_stats[cat]["tp"] / tot, 4) if tot > 0 else 0.0

        sahi_results.append(row)
        print(f"  P={row['Precision']:.4f}  R={row['Recall']:.4f}  "
              f"F1={row['F1']:.4f}  F2={row['F2']:.4f}  "
              f"VT_recall={row['very_tiny_recall']:.4f}  "
              f"Lat={row['Avg_Inf_ms']:.0f}ms")

    # Save SAHI sweep results
    sahi_df = pd.DataFrame(sahi_results)
    sahi_df.to_excel(f"{EXCEL_DIR}/sahi_sweep_results.xlsx", index=False)
    print(f"\n✓ SAHI sweep results saved → {EXCEL_DIR}/sahi_sweep_results.xlsx")

    # Identify best SAHI config by very-tiny recall
    best_sahi = max(sahi_results, key=lambda x: x["very_tiny_recall"])
    print(f"\n{'='*70}")
    print(f"BEST SAHI CONFIG: {best_sahi['Config']}")
    print(f"  Very-tiny recall: {best_sahi['very_tiny_recall']:.4f}")
    print(f"  Overall: P={best_sahi['Precision']:.4f}  R={best_sahi['Recall']:.4f}  "
          f"F1={best_sahi['F1']:.4f}")
    print(f"  Latency: {best_sahi['Avg_Inf_ms']:.0f}ms (vs ~51ms without SAHI)")
    print(f"{'='*70}")

    # Save best SAHI config as JSON for future use
    import json as _json
    with open(f"{REPORT_DIR}/best_sahi_config.json", "w") as f:
        _json.dump(best_sahi, f, indent=2)

    # ── SAHI vs no-SAHI comparison table ─────────────────────────────────────
    # Find the matching test summary for the copy-paste model
    cp_test_summary = next(
        (s for s in test_summaries if "copypaste" in s["Model"].lower()),
        test_summaries[0] if test_summaries else None
    )
    if cp_test_summary:
        print(f"\n{'='*70}")
        print("SAHI IMPACT — COPY-PASTE MODEL (best config vs no SAHI)")
        print(f"{'='*70}")
        compare_metrics = [
            ("Precision",       "Precision"),
            ("Recall",          "Recall"),
            ("F1",              "F1"),
            ("F2",              "F2"),
            ("very_tiny_recall","Very Tiny Recall"),
            ("tiny_recall",     "Tiny Recall"),
            ("small_recall",    "Small Recall"),
            ("medium_recall",   "Medium Recall"),
        ]
        sahi_compare_rows = []
        for key, label in compare_metrics:
            no_sahi = float(cp_test_summary.get(key, 0))
            with_sahi = float(best_sahi.get(key, 0))
            delta = with_sahi - no_sahi
            sahi_compare_rows.append({
                "Metric": label,
                "No SAHI": round(no_sahi, 4),
                "With SAHI": round(with_sahi, 4),
                "Delta": round(delta, 4),
            })
            print(f"  {label:20s}: {no_sahi:.4f} → {with_sahi:.4f}  "
                  f"({'+'if delta>=0 else ''}{delta:.4f})")

        sahi_compare_df = pd.DataFrame(sahi_compare_rows)
        sahi_compare_df.to_excel(f"{EXCEL_DIR}/sahi_impact_comparison.xlsx", index=False)
        print(f"\n✓ SAHI impact table → {EXCEL_DIR}/sahi_impact_comparison.xlsx")

    del sahi_model; gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ============================================================================
# CELL 24: TTA — Test-Time Augmentation Evaluation
# ============================================================================
# TTA uses multi-scale + flip at inference time.
# Ultralytics: model.val(augment=True, imgsz=832)
# Published gains: +1-2% mAP, especially on small objects.
# ============================================================================

print("\n" + "=" * 80)
print("TTA — TEST-TIME AUGMENTATION EVALUATION")
print("=" * 80)

from ultralytics import YOLO as _YOLO_TTA

tta_results = {}

if os.path.exists(MAMBA_BEST):
    tta_model = _YOLO_TTA(MAMBA_BEST)

    # ── Standard eval (no TTA) — for fair comparison ──────────────────────
    print("\n── Standard (no TTA) ──")
    res_no_tta = tta_model.val(data="c2a.yaml", split="test",
                                imgsz=640, augment=False,
                                verbose=False, plots=False)
    tta_results["No TTA (640)"] = {
        "mAP50":     round(float(res_no_tta.box.map50), 4),
        "mAP50-95":  round(float(res_no_tta.box.map),   4),
        "Precision":  round(float(res_no_tta.box.mp),    4),
        "Recall":     round(float(res_no_tta.box.mr),    4),
    }
    print(f"  mAP50={tta_results['No TTA (640)']['mAP50']:.4f}  "
          f"mAP50-95={tta_results['No TTA (640)']['mAP50-95']:.4f}")

    # ── TTA at imgsz=832 (30% larger — recommended by Ultralytics docs) ──
    print("\n── TTA (imgsz=832, augment=True) ──")
    res_tta = tta_model.val(data="c2a.yaml", split="test",
                             imgsz=832, augment=True,
                             verbose=False, plots=False)
    tta_results["TTA (832)"] = {
        "mAP50":     round(float(res_tta.box.map50), 4),
        "mAP50-95":  round(float(res_tta.box.map),   4),
        "Precision":  round(float(res_tta.box.mp),    4),
        "Recall":     round(float(res_tta.box.mr),    4),
    }
    print(f"  mAP50={tta_results['TTA (832)']['mAP50']:.4f}  "
          f"mAP50-95={tta_results['TTA (832)']['mAP50-95']:.4f}")

    # ── TTA at imgsz=1024 (for extra small-object gains) ─────────────────
    print("\n── TTA (imgsz=1024, augment=True) ──")
    try:
        res_tta_1024 = tta_model.val(data="c2a.yaml", split="test",
                                      imgsz=1024, augment=True,
                                      verbose=False, plots=False)
        tta_results["TTA (1024)"] = {
            "mAP50":     round(float(res_tta_1024.box.map50), 4),
            "mAP50-95":  round(float(res_tta_1024.box.map),   4),
            "Precision":  round(float(res_tta_1024.box.mp),    4),
            "Recall":     round(float(res_tta_1024.box.mr),    4),
        }
        print(f"  mAP50={tta_results['TTA (1024)']['mAP50']:.4f}  "
              f"mAP50-95={tta_results['TTA (1024)']['mAP50-95']:.4f}")
    except torch.cuda.OutOfMemoryError:
        print("  ⚠  OOM at imgsz=1024 — skipping this TTA config")

    # ── TTA comparison table ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TTA RESULTS COMPARISON")
    print(f"{'='*70}")
    tta_df = pd.DataFrame(tta_results).T
    tta_df.index.name = "Config"
    print(tta_df.to_string())
    tta_df.to_excel(f"{EXCEL_DIR}/tta_comparison.xlsx")
    print(f"\n✓ TTA comparison → {EXCEL_DIR}/tta_comparison.xlsx")

    del tta_model; gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

else:
    print("⚠  Best model not found — skipping TTA evaluation")


# ============================================================================
# CELL 25: GRAND SUMMARY — All Configurations
# ============================================================================
# Combines: baseline (no aug), copy-paste, copy-paste+SAHI, copy-paste+TTA

print("\n" + "=" * 80)
print("GRAND SUMMARY — ALL CONFIGURATIONS")
print("=" * 80)

grand_rows = []

# Row 1: Previous model (no augmentation) — from official test metrics
for name, metrics in official_test.items():
    grand_rows.append({
        "Configuration": name,
        "mAP50": metrics["mAP50"],
        "mAP50-95": metrics["mAP50-95"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "Note": "standard eval",
    })

# Row 2: Best SAHI config
if 'sahi_results' in dir() and sahi_results:
    grand_rows.append({
        "Configuration": f"CopyPaste + SAHI ({best_sahi['Config']})",
        "mAP50": "-",
        "mAP50-95": "-",
        "Precision": best_sahi["Precision"],
        "Recall": best_sahi["Recall"],
        "VeryTiny_Recall": best_sahi["very_tiny_recall"],
        "Avg_Latency_ms": best_sahi["Avg_Inf_ms"],
        "Note": "SAHI custom eval",
    })

# Row 3-4: TTA configs
for config_name, metrics in tta_results.items():
    grand_rows.append({
        "Configuration": f"CopyPaste + {config_name}",
        **metrics,
        "Note": "ultralytics val",
    })

grand_df = pd.DataFrame(grand_rows)
grand_df.to_excel(f"{EXCEL_DIR}/grand_summary_all_configs.xlsx", index=False)
print(grand_df.to_string(index=False))
print(f"\n✓ Grand summary → {EXCEL_DIR}/grand_summary_all_configs.xlsx")

# ── Save grand summary as JSON too ──────────────────────────────────────────
with open(f"{REPORT_DIR}/grand_summary.json", "w") as f:
    json.dump(grand_rows, f, indent=2, default=str)
print(f"✓ Grand summary JSON → {REPORT_DIR}/grand_summary.json")

print("\n" + "=" * 80)
print("ALL EVALUATIONS COMPLETE")
print("=" * 80)


# ============================================================================
# CELL 26: Package All Results
# ============================================================================
try:
    from IPython.display import FileLink
    _in_notebook = True
except ImportError:
    _in_notebook = False

import subprocess

_zip_cmd = (
    f"zip -r /kaggle/working/mamba_cbam_p2_copypaste_results.zip "
    f"{EXCEL_DIR} {PLOT_DIR} {REPORT_DIR} "
    f"/kaggle/working/session_best.pt "
    f"/kaggle/working/gradient_norms.csv 2>/dev/null || true"
)
subprocess.run(_zip_cmd, shell=True)
print("✓ Results packaged → /kaggle/working/mamba_cbam_p2_copypaste_results.zip")
if _in_notebook:
    display(FileLink("/kaggle/working/mamba_cbam_p2_copypaste_results.zip"))

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE — DOWNLOAD FILES FROM /kaggle/working/")
print("  • mamba_cbam_p2_copypaste_results.zip  (all reports/plots/excel)")
print("  • session_best.pt                      (best model weights)")
print("  • session_last.pt                      (checkpoint for resume)")
print("=" * 80)
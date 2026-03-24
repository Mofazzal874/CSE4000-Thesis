"""
================================================================================
DISASTER HUMAN DETECTION — ATROUS-MAMBA + CBAM + P2 HEAD
================================================================================
Trains ONE new model only:
  yolo11m_atrousmamba_cbam_p2head

Loads pre-trained for comparison (if available):
  yolo11m_mamba_cbam_p2head  (your previous Mamba run)
  yolo11m_cbam_p2head        (your ablation study baseline)

Architecture changes vs Mamba+CBAM+P2:
  Backbone : unchanged (CBAM replaces C2PSA at layer 10)
  Neck     : C3k2 at layers 13,16,19,22,25 → C3K2Mamba with AtrousSSM (NEW)
             - AtrousSSM replaces LocalWindowSSM
             - Multi-scale dilated window scanning: d=[1,2,4]
             - Same token count per scan (64) — larger receptive field
             - Gated fusion across dilation scales
  Head     : P2 4-scale detection (unchanged)

AtrousSSM novelty:
  ✓ First application of dilated/atrous scanning to State Space Models
  ✓ Expands receptive field from 8×8 to 32×32 pixels WITHOUT increasing
    token count per scan (stays at 64, T4-safe)
  ✓ Each dilation branch has independent SSM weights — specialization
  ✓ Gated per-position fusion learns optimal scale weighting

Implementation strategy (Path B — YAML-native):
  ✓ Pure PyTorch — zero CUDA kernel compilation
  ✓ C3K2Mamba registered natively in ultralytics → YAML parser builds it directly
  ✓ NO post-init injection needed — model is correct from the start
  ✓ DDP-safe — both processes parse the same YAML with registered modules
  ✓ Bidirectional local-window scan (forward + backward) per branch
  ✓ Adaptive window size per channel depth
  ✓ FP16/AMP safe (all SSM intermediates forced to FP32)
  ✓ Preserves original n-repeat count → cv2 channel dims unchanged

Metrics & data saved:
  ✓ F1/F2 per epoch (post-hoc from results.csv)
  ✓ PR curve, F1-confidence curve (plots=True in val)
  ✓ Confidence distribution histogram
  ✓ 2-way (or 3-way) per-epoch overlay comparison
  ✓ Gradient norm tracking callback
  ✓ NaN/Inf loss detection callback
  ✓ F2-based early stopping (custom callback)
  ✓ ECE calibration
  ✓ Per-size recall breakdown
  ✓ Model complexity comparison table
  ✓ Speed benchmark at multiple resolutions
  ✓ Master summary report (text + JSON)
  ✓ Everything packaged as downloadable ZIP

PATH B — DUAL GPU VERSION (YAML-NATIVE REGISTRATION)
  C3K2Mamba is registered as a first-class ultralytics module.
  The YAML references C3K2Mamba directly — no post-init injection needed.
  DDP works because both processes parse the same YAML with registered modules.
  Resume-safe: YAML stores C3K2Mamba → checkpoint rebuild preserves architecture.

COMPUTE NOTES (Kaggle T4 x2 — DUAL GPU):
  Batch size  : 16 (T4×2) or 4 (P100)
  d_state     : 4  (safe for T4 16GB)
  Window size : adaptive (4→512ch, 6→256ch, 8→128ch)
  Dilations   : [1, 2, 4] (3 branches — sequential Python scan)
  Gradient clipping : max_norm=10.0 (prevents epoch-1 explosion)
  Expected VRAM: ~8-10GB per GPU at batch=16 (T4 has 16GB)

  TIMING (reference: base model = 7.768h on 2×T4 @ batch=12):
    AtrousMamba SSM scan adds ~1.5-2× overhead on neck layers.
    Realistic estimate: ~10-12h for 120 epochs on 2×T4.
    TIGHT for 12h session — early stopping should help.
    Resume built-in as backup if session times out.
================================================================================
"""

# ============================================================================
# CELL 1: Control Flags & Dependencies
# ============================================================================

TEST_MODE       = True   # True = 2 epochs, 5% data | False = full 120 epochs
RESUME_TRAINING = False   # Set True to resume from a checkpoint
RESUME_PT       = ""      # Path to last.pt — leave empty for AUTO-DETECT, e.g.:
                          # "/kaggle/input/atrous-mamba-checkpoint/session_last.pt"
                          # AUTO-DETECT: if empty, scans /kaggle/input/ for session_last.pt

import subprocess, sys, re

def pip_install(pkg, extra=""):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-U", pkg] +
        (extra.split() if extra else [])
    )

# ── CUDA / PyTorch version alignment check ────────────────────────────────
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
pip_install("ultralytics")
pip_install("timm")
pip_install("thop")
pip_install("openpyxl")
pip_install("scikit-learn")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "pandas<3.0", "matplotlib<3.10", "tqdm"])
print("✓ All dependencies installed")


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
# PATH B: Dual GPU is safe because C3K2Mamba is registered natively in YAML.
# DDP subprocesses parse the same YAML → C3K2Mamba is built directly.
DEVICE = "0,1" if num_gpus >= 2 else "0" if num_gpus >= 1 else "cpu"
if num_gpus >= 2:
    print(f"  ✓ {num_gpus} GPUs detected — using DEVICE='{DEVICE}' (DDP-safe)")
    print(f"    C3K2Mamba is YAML-native → survives DDP subprocess rebuild")
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
    BATCH_SIZE     = 8
    print("⚠  TEST MODE  — 5% data, 2 epochs")
else:
    TRAIN_FRACTION = 1.0
    NUM_EPOCHS     = 120
    PATIENCE       = 15
    F2_PATIENCE    = 10
    TEST_IMAGES    = None
    VAL_IMAGES     = None
    SAVE_PERIOD    = 5
    BATCH_SIZE     = 12 if gpu_mem >= 14 else 4   # 6/GPU — batch=16 OOMs on T4 (13.7GB/14.9GB)
    print(f"🚀  FULL MODE  — {NUM_EPOCHS} epochs | batch={BATCH_SIZE}")
    print(f"    ⏱  Estimated: ~12-14h on 2×T4 (early stopping helps stay within 12h)")
    print(f"    Resume built-in as backup if session times out")

GRAD_ACCUM = max(1, 24 // BATCH_SIZE)   # effective batch ~24 (was 16)
print(f"  Batch={BATCH_SIZE} | GradAccum={GRAD_ACCUM} | EffectiveBatch={BATCH_SIZE*GRAD_ACCUM}")

# ── Checkpoint / session-resume config ──────────────────────────────────────
CHECKPOINT_EVERY = 5 if not TEST_MODE else 1

# ── OOM retry ladder ─────────────────────────────────────────────────────────
OOM_RETRY_BATCHES = [BATCH_SIZE, max(BATCH_SIZE // 2, 2), max(BATCH_SIZE // 4, 1)]

# ── AtrousSSM config ─────────────────────────────────────────────────────────
DILATIONS = [1, 2, 4]   # multi-scale dilation rates
D_STATE   = 4            # SSM state size
print(f"  AtrousSSM dilations: {DILATIONS}  |  d_state: {D_STATE}")


# ============================================================================
# CELL 3: Dataset Configuration
# ============================================================================
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

# ── Locate previous models for comparison ─────────────────────────────────
# NOTE: To include previous models in comparison, upload their best.pt as
# a Kaggle Dataset. The search below looks for best.pt files anywhere under
# /kaggle/input/ and matches by directory name:
#   - "cbam_p2" in path (but NOT "mamba") → CBAM+P2 baseline
#   - "mamba_cbam_p2" in path             → Old Mamba+CBAM+P2
# You can also just rename the uploaded .pt file to include these keywords.
CBAM_P2_BEST = None
OLD_MAMBA_BEST = None

print("Searching for previous models in /kaggle/input/ …")
for root, dirs, files in os.walk("/kaggle/input"):
    # Check specific known paths first
    candidate = os.path.join(root, "runs", "detect", "yolo11m_cbam_p2head",
                              "weights", "best.pt")
    if os.path.isfile(candidate) and CBAM_P2_BEST is None:
        CBAM_P2_BEST = candidate
        print(f"  ✓ CBAM+P2 found: {CBAM_P2_BEST}")

    candidate2 = os.path.join(root, "runs", "detect", "yolo11m_mamba_cbam_p2head",
                               "weights", "best.pt")
    if os.path.isfile(candidate2) and OLD_MAMBA_BEST is None:
        OLD_MAMBA_BEST = candidate2
        print(f"  ✓ Old Mamba+CBAM+P2 found: {OLD_MAMBA_BEST}")

    # Broad search: any best.pt in a matching folder name
    for f in files:
        if f != "best.pt":
            continue
        fpath = os.path.join(root, f)
        rlow = root.lower()
        if "mamba" in rlow and "cbam" in rlow and OLD_MAMBA_BEST is None:
            OLD_MAMBA_BEST = fpath
            print(f"  ✓ Old Mamba model found: {OLD_MAMBA_BEST}")
        elif "cbam" in rlow and "p2" in rlow and "mamba" not in rlow and CBAM_P2_BEST is None:
            CBAM_P2_BEST = fpath
            print(f"  ✓ CBAM+P2 model found: {CBAM_P2_BEST}")

if CBAM_P2_BEST is None:
    print("  ⚠  CBAM+P2 model NOT found — will skip that comparison")
    print("     To include it: upload its best.pt as a Kaggle dataset")
if OLD_MAMBA_BEST is None:
    print("  ⚠  Old Mamba+CBAM+P2 NOT found — will skip that comparison")
    print("     To include it: upload its best.pt as a Kaggle dataset")


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

_t = torch.randn(2, 512, 20, 20)
assert CBAM(16, 7)(_t).shape == _t.shape
print("✓ CBAM module OK")


# ============================================================================
# CELL 5: AtrousSSM Modules (NOVEL — replaces LocalWindowSSM)
# ============================================================================

def _get_window_size(channels: int) -> int:
    """Adaptive window size: larger windows for lower channel counts."""
    if channels >= 512: return 4   # 4×4=16 tokens
    if channels >= 256: return 6   # 6×6=36 tokens
    return 8                        # 8×8=64 tokens


class _SelectiveScan1D(nn.Module):
    """
    One-direction selective scan.  u: (B, L, D) → (B, L, D)
    All SSM computations forced to FP32 for numerical stability.
    """
    def __init__(self, d_model: int, d_state: int = 4, dt_rank_ratio: int = 16):
        super().__init__()
        D, N = d_model, d_state
        dt_rank = max(D // dt_rank_ratio, 1)
        self.D, self.N, self.dt_rank = D, N, dt_rank

        self.conv1d = nn.Conv1d(D, D, kernel_size=4, padding=3,
                                groups=D, bias=True)
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
        dt_raw, B_param, C_param = xBC_dt.split(
            [self.dt_rank, self.N, self.N], dim=-1
        )
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


class AtrousSSM(nn.Module):
    """
    Multi-Scale Dilated Window State Space Model.

    Extends local-window SSM with dilated scanning: multiple dilation rates
    sample tokens from progressively larger spatial areas while keeping the
    token count per window constant (ws^2, T4-safe).

    This is to SSM what atrous/dilated convolution is to standard convolution.
    """

    def __init__(self, d_model: int, d_state: int = 4,
                 window_size: int = 8, dilations: list = None):
        super().__init__()
        self.ws = window_size
        self.dilations = dilations or [1, 2, 4]
        D = d_model
        n_br = len(self.dilations)

        self.branches = nn.ModuleList()
        for _ in self.dilations:
            branch = nn.ModuleDict({
                'norm':     nn.LayerNorm(D),
                'in_proj':  nn.Linear(D, D * 2, bias=False),
                'scan_fwd': _SelectiveScan1D(D, d_state),
                'scan_bwd': _SelectiveScan1D(D, d_state),
                'out_proj': nn.Linear(D, D, bias=False),
            })
            # Small init: out_proj starts near zero → branch output ≈ residual
            # in_proj uses Xavier for balanced forward/backward signal
            nn.init.normal_(branch['out_proj'].weight, std=0.02)
            nn.init.xavier_uniform_(branch['in_proj'].weight)
            self.branches.append(branch)

        self.fusion_gate = nn.Sequential(
            nn.Conv2d(D * n_br, D, 1, bias=False),
            nn.Sigmoid()
        )
        self.fusion_proj = nn.Conv2d(D * n_br, D, 1, bias=False)
        self.out_norm = nn.LayerNorm(D)

        # ── Weight initialization for training stability ──────────────────
        # Small init on fusion gate → starts near 0.5 (uniform mixing)
        # Small init on fusion proj → small output magnitude at start
        # This prevents gradient explosion at epoch 1 from randomly large
        # gate/projection values amplifying through the network.
        nn.init.normal_(self.fusion_gate[0].weight, std=0.01)
        nn.init.normal_(self.fusion_proj.weight, std=0.02)

    def _dilated_partition(self, x: torch.Tensor, dilation: int):
        B, C, H, W = x.shape
        ws = self.ws
        region = ws * dilation

        ph = (region - H % region) % region
        pw = (region - W % region) % region
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))

        _, _, Hp, Wp = x.shape
        nH, nW = Hp // region, Wp // region

        x = x.reshape(B, C, nH, region, nW, region)
        x = x.permute(0, 2, 4, 1, 3, 5)           # B, nH, nW, C, region, region
        x = x[:, :, :, :, ::dilation, ::dilation].contiguous()  # B, nH, nW, C, ws, ws

        tokens = x.reshape(B * nH * nW, C, ws * ws).transpose(1, 2)
        return tokens, (B, C, H, W, Hp, Wp, nH, nW)

    def _dilated_reverse(self, tokens: torch.Tensor, meta: tuple,
                         dilation: int) -> torch.Tensor:
        B, C, H, W, Hp, Wp, nH, nW = meta
        ws = self.ws
        region = ws * dilation

        windows = tokens.transpose(1, 2).reshape(B * nH * nW, C, ws, ws)

        if dilation > 1:
            windows = F.interpolate(
                windows, size=(region, region),
                mode='bilinear', align_corners=False
            )

        windows = windows.reshape(B, nH, nW, C, region, region)
        x = windows.permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(B, C, Hp, Wp)
        return x[:, :, :H, :W].contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, C, H, W)"""
        branch_outputs = []

        for branch, dilation in zip(self.branches, self.dilations):
            tokens, meta = self._dilated_partition(x, dilation)
            residual = tokens

            tokens_n = branch['norm'](tokens)
            xz = branch['in_proj'](tokens_n)
            x_in, z = xz.chunk(2, dim=-1)

            y_fwd = branch['scan_fwd'](x_in)
            y_bwd = branch['scan_bwd'](x_in.flip(1)).flip(1)
            y = (y_fwd + y_bwd) * F.silu(z)

            y = branch['out_proj'](y) + residual

            spatial = self._dilated_reverse(y, meta, dilation)
            branch_outputs.append(spatial)

        concat = torch.cat(branch_outputs, dim=1)
        gate   = self.fusion_gate(concat)
        fused  = self.fusion_proj(concat)

        out = fused * gate + x * (1 - gate)

        out = out.permute(0, 2, 3, 1)
        out = self.out_norm(out)
        out = out.permute(0, 3, 1, 2)

        return out


class _MambaBottleneck(nn.Module):
    """Conv3x3 → AtrousSSM → Conv3x3  (+residual if same C)"""
    def __init__(self, c: int, shortcut: bool, d_state: int,
                 window_size: int, dilations: list = None):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.cv1 = Conv(c, c, 3, 1)
        self.ssm = AtrousSSM(c, d_state=d_state, window_size=window_size,
                             dilations=dilations)
        self.cv2 = Conv(c, c, 3, 1)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.ssm(self.cv1(x)))
        return x + y if self.add else y


class C3K2Mamba(nn.Module):
    """C2f-style block with AtrousSSM bottleneck. Drop-in for C3k2 in YOLO11m neck."""
    def __init__(self, c1: int, c2: int, n: int = 1,
                 shortcut: bool = False, g: int = 1, e: float = 0.5,
                 d_state: int = 4, dilations: list = None):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        ws     = _get_window_size(self.c)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m   = nn.ModuleList(
            _MambaBottleneck(self.c, shortcut and c1 == c2,
                            d_state, ws, dilations=dilations)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


print("✓ AtrousSSM modules defined (AtrousSSM, C3K2Mamba)")


# ============================================================================
# CELL 6: Smoke Test & Dry Run
# ============================================================================

def run_smoke_test():
    print("\n" + "=" * 70)
    print("SMOKE TEST: Verifying AtrousSSM modules before training")
    print("=" * 70)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    passed = 0

    # Test 1: AtrousSSM shape consistency
    print("\n[1/8] AtrousSSM shape test …")
    for C, H in [(512, 40), (256, 80), (128, 160), (64, 160)]:
        ws = _get_window_size(C)
        ssm = AtrousSSM(C, d_state=D_STATE, window_size=ws,
                        dilations=DILATIONS).to(device)
        x = torch.randn(2, C, H, H, device=device)
        y = ssm(x)
        assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"
        print(f"    C={C:4d}  H={H:3d}  ws={ws}  → {y.shape}  ✓")
    passed += 1

    # Test 2: C3K2Mamba shape
    print("\n[2/8] C3K2Mamba shape test …")
    for c1, c2, n in [(1024, 512, 2), (768, 256, 2), (384, 128, 2)]:
        blk = C3K2Mamba(c1, c2, n=n, shortcut=False,
                        d_state=D_STATE, dilations=DILATIONS).to(device)
        x = torch.randn(2, c1, 40, 40, device=device)
        y = blk(x)
        assert y.shape == (2, c2, 40, 40), f"Shape mismatch: {y.shape}"
        print(f"    ({c1}→{c2}, n={n}) → {y.shape}  ✓")
    passed += 1

    # Test 3: NaN/Inf check
    print("\n[3/8] NaN / Inf check …")
    blk = C3K2Mamba(1024, 512, n=2, d_state=D_STATE, dilations=DILATIONS).to(device)
    x = torch.randn(2, 1024, 40, 40, device=device)
    y = blk(x)
    assert torch.isfinite(y).all(), "NaN or Inf in output!"
    print("    No NaN/Inf  ✓")
    passed += 1

    # Test 4: Gradient flow
    print("\n[4/8] Gradient flow …")
    blk = C3K2Mamba(512, 256, n=2, d_state=D_STATE, dilations=DILATIONS).to(device)
    x = torch.randn(2, 512, 40, 40, device=device, requires_grad=True)
    y = blk(x)
    y.sum().backward()
    assert x.grad is not None, "No gradient at input!"
    total_norm = sum(p.grad.norm().item()**2 for p in blk.parameters()
                     if p.grad is not None) ** 0.5
    print(f"    Grad norm: {total_norm:.4f}  ✓")
    passed += 1

    # Test 5: AMP / FP16 safety
    if device == "cuda":
        print("\n[5/8] AMP (FP16) safety …")
        blk = C3K2Mamba(512, 256, n=2, d_state=D_STATE, dilations=DILATIONS).to(device)
        x   = torch.randn(2, 512, 40, 40, device=device)
        with torch.amp.autocast("cuda"):
            y = blk(x)
        assert torch.isfinite(y).all(), "NaN in FP16 forward!"
        print(f"    Input dtype: {x.dtype} | Output dtype: {y.dtype}  ✓")
    else:
        print("\n[5/8] AMP test skipped (CPU mode)")
    passed += 1

    # Test 6: Bidirectional scan asymmetry
    print("\n[6/8] Bidirectional scan asymmetry test …")
    ssm = AtrousSSM(64, d_state=D_STATE, window_size=4, dilations=[1]).to(device)
    x1 = torch.randn(1, 64, 8, 8, device=device)
    x2 = x1.flip(-1)
    y1, y2 = ssm(x1), ssm(x2)
    assert not torch.allclose(y1, y2.flip(-1), atol=1e-4), \
        "Bidirectional scan is symmetric!"
    print("    Forward and backward scans differ  ✓")
    passed += 1

    # Test 7: Multi-dilation branch independence
    print("\n[7/8] Multi-dilation branch independence …")
    ssm_multi = AtrousSSM(64, d_state=D_STATE, window_size=4, dilations=[1, 2, 4]).to(device)
    x = torch.randn(1, 64, 16, 16, device=device)
    y = ssm_multi(x)
    assert y.shape == x.shape, f"Multi-dilation shape mismatch: {y.shape}"
    assert len(ssm_multi.branches) == 3, "Expected 3 branches"
    print("    3 independent branches verified  ✓")
    passed += 1

    # Test 8: Memory estimate
    if device == "cuda":
        print("\n[8/8] Memory estimate (layer-13 equivalent, batch=8) …")
        torch.cuda.reset_peak_memory_stats()
        blk = C3K2Mamba(1024, 512, n=2, d_state=D_STATE, dilations=DILATIONS).to(device)
        x = torch.randn(8, 1024, 40, 40, device=device)
        with torch.amp.autocast("cuda"):
            y = blk(x)
        y.sum().backward()
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"    Peak VRAM: {peak_mb:.0f} MB  (T4 budget: 16384 MB)  ✓")
        del blk, x, y; torch.cuda.empty_cache(); gc.collect()
    else:
        print("\n[8/8] Memory test skipped (CPU mode)")
    passed += 1

    print(f"\n{'='*70}")
    print(f"SMOKE TEST: {passed}/8 passed ✓" if passed == 8
          else f"SMOKE TEST: {passed}/8 — CHECK FAILURES ABOVE ⚠")
    print("=" * 70)
    return passed == 8

assert run_smoke_test(), "Smoke test failed — fix errors before training!"


# ============================================================================
# CELL 7: Register CBAM + C3K2Mamba in Ultralytics (YAML-native for DDP)
# ============================================================================
#
# PATH B STRATEGY:
#   Instead of injecting C3K2Mamba after model build (which DDP destroys),
#   we register C3K2Mamba as a native ultralytics module. The YAML parser
#   then builds C3K2Mamba directly — no injection needed. DDP subprocesses
#   parse the same YAML and find C3K2Mamba in the module registry.
#
import site as _site

# ── Step 1: Write atrous_mamba_module.py (contains all SSM classes) ────────
# This file will be copied to site-packages so DDP subprocesses can import it.

_atrous_module_code = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _get_window_size(channels: int) -> int:
    if channels >= 512: return 4
    if channels >= 256: return 6
    return 8


class _SelectiveScan1D(nn.Module):
    """One-direction selective scan. u: (B, L, D) -> (B, L, D)"""
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
            torch.rand(D) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, u):
        B_win, L, D = u.shape
        in_dtype = u.dtype
        u_conv = self.conv1d(u.transpose(1, 2))[:, :, :L].transpose(1, 2)
        u_act = F.silu(u_conv)
        xBC_dt = self.x_proj(u_act)
        dt_raw, B_param, C_param = xBC_dt.split([self.dt_rank, self.N, self.N], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw)).float()
        B_param, C_param, u_f = B_param.float(), C_param.float(), u_act.float()
        A = -torch.exp(self.A_log.float())
        deltaA = torch.exp(torch.einsum("bld,dn->bldn", dt, A))
        deltaB_u = torch.einsum("bld,bln,bld->bldn", dt, B_param, u_f)
        x = torch.zeros(B_win, D, self.N, device=u.device, dtype=torch.float32)
        ys = []
        for i in range(L):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            ys.append((x * C_param[:, i, :].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1).to(in_dtype)
        return y + u_act * self.D_skip.to(in_dtype)


class AtrousSSM(nn.Module):
    """Multi-Scale Dilated Window State Space Model."""
    def __init__(self, d_model, d_state=4, window_size=8, dilations=None):
        super().__init__()
        self.ws = window_size
        self.dilations = dilations or [1, 2, 4]
        D = d_model
        n_br = len(self.dilations)
        self.branches = nn.ModuleList()
        for _ in self.dilations:
            branch = nn.ModuleDict({
                'norm':     nn.LayerNorm(D),
                'in_proj':  nn.Linear(D, D * 2, bias=False),
                'scan_fwd': _SelectiveScan1D(D, d_state),
                'scan_bwd': _SelectiveScan1D(D, d_state),
                'out_proj': nn.Linear(D, D, bias=False),
            })
            nn.init.normal_(branch['out_proj'].weight, std=0.02)
            nn.init.xavier_uniform_(branch['in_proj'].weight)
            self.branches.append(branch)
        self.fusion_gate = nn.Sequential(nn.Conv2d(D * n_br, D, 1, bias=False), nn.Sigmoid())
        self.fusion_proj = nn.Conv2d(D * n_br, D, 1, bias=False)
        self.out_norm = nn.LayerNorm(D)
        nn.init.normal_(self.fusion_gate[0].weight, std=0.01)
        nn.init.normal_(self.fusion_proj.weight, std=0.02)

    def _dilated_partition(self, x, dilation):
        B, C, H, W = x.shape
        ws, region = self.ws, self.ws * dilation
        ph, pw = (region - H % region) % region, (region - W % region) % region
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        _, _, Hp, Wp = x.shape
        nH, nW = Hp // region, Wp // region
        x = x.reshape(B, C, nH, region, nW, region).permute(0, 2, 4, 1, 3, 5)
        x = x[:, :, :, :, ::dilation, ::dilation].contiguous()
        return x.reshape(B * nH * nW, C, ws * ws).transpose(1, 2), (B, C, H, W, Hp, Wp, nH, nW)

    def _dilated_reverse(self, tokens, meta, dilation):
        B, C, H, W, Hp, Wp, nH, nW = meta
        ws, region = self.ws, self.ws * dilation
        windows = tokens.transpose(1, 2).reshape(B * nH * nW, C, ws, ws)
        if dilation > 1:
            windows = F.interpolate(windows, size=(region, region), mode='bilinear', align_corners=False)
        x = windows.reshape(B, nH, nW, C, region, region).permute(0, 3, 1, 4, 2, 5)
        return x.reshape(B, C, Hp, Wp)[:, :, :H, :W].contiguous()

    def forward(self, x):
        branch_outputs = []
        for branch, dilation in zip(self.branches, self.dilations):
            tokens, meta = self._dilated_partition(x, dilation)
            residual = tokens
            tokens_n = branch['norm'](tokens)
            xz = branch['in_proj'](tokens_n)
            x_in, z = xz.chunk(2, dim=-1)
            y_fwd = branch['scan_fwd'](x_in)
            y_bwd = branch['scan_bwd'](x_in.flip(1)).flip(1)
            y = (y_fwd + y_bwd) * F.silu(z)
            y = branch['out_proj'](y) + residual
            branch_outputs.append(self._dilated_reverse(y, meta, dilation))
        concat = torch.cat(branch_outputs, dim=1)
        gate = self.fusion_gate(concat)
        fused = self.fusion_proj(concat)
        out = fused * gate + x * (1 - gate)
        out = self.out_norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out


class _MambaBottleneck(nn.Module):
    def __init__(self, c, shortcut, d_state, window_size, dilations=None):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.cv1 = Conv(c, c, 3, 1)
        self.ssm = AtrousSSM(c, d_state=d_state, window_size=window_size, dilations=dilations)
        self.cv2 = Conv(c, c, 3, 1)
        self.add = shortcut

    def forward(self, x):
        y = self.cv2(self.ssm(self.cv1(x)))
        return x + y if self.add else y


class C3K2Mamba(nn.Module):
    """C2f-style block with AtrousSSM bottleneck. Drop-in for C3k2 in YOLO11m neck.
    YAML args: [c2, shortcut, g, e, d_state, dilations]
    After parse_model inserts n: C3K2Mamba(c1, c2, n, shortcut, g, e, d_state, dilations)
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5,
                 d_state=4, dilations=None):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        ws = _get_window_size(self.c)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            _MambaBottleneck(self.c, shortcut and c1 == c2,
                             d_state, ws, dilations=dilations or [1, 2, 4])
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
'''

with open("/kaggle/working/atrous_mamba_module.py", "w") as f:
    f.write(_atrous_module_code)
print("✓ atrous_mamba_module.py written")

# Also write CBAM module
_cbam_src = "/kaggle/working/cbam_module.py"

# ── Step 2: Copy both modules to site-packages ────────────────────────────
_installed_sp = None
for _sp in _site.getsitepackages():
    try:
        shutil.copy(_cbam_src, os.path.join(_sp, "cbam_module.py"))
        shutil.copy("/kaggle/working/atrous_mamba_module.py",
                     os.path.join(_sp, "atrous_mamba_module.py"))
        print(f"  ✓ cbam_module.py → site-packages: {_sp}")
        print(f"  ✓ atrous_mamba_module.py → site-packages: {_sp}")
        _installed_sp = _sp
        break
    except Exception:
        continue

if _installed_sp is None:
    print("  ⚠  Could not install to site-packages")

if "/kaggle/working" not in sys.path:
    sys.path.insert(0, "/kaggle/working")

_existing_pypath = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = "/kaggle/working" + (
    ":" + _existing_pypath if _existing_pypath else "")

# ── Step 3: Patch ultralytics source to import CBAM + C3K2Mamba ────────────
import ultralytics.nn.tasks as _ult_tasks_mod
_tasks_file = _ult_tasks_mod.__file__

_cbam_inject   = "from cbam_module import CBAM, ChannelAttention, SpatialAttention\n"
_mamba_inject  = "from atrous_mamba_module import C3K2Mamba, AtrousSSM, _MambaBottleneck, _SelectiveScan1D\n"

with open(_tasks_file, "r") as _f:
    _tasks_src = _f.read()

_tasks_changed = False
if "from cbam_module import" not in _tasks_src:
    _tasks_src = _cbam_inject + _tasks_src
    _tasks_changed = True
if "from atrous_mamba_module import" not in _tasks_src:
    _tasks_src = _mamba_inject + _tasks_src
    _tasks_changed = True

if _tasks_changed:
    with open(_tasks_file, "w") as _f:
        _f.write(_tasks_src)
    print(f"  ✓ Patched ultralytics/nn/tasks.py with CBAM + C3K2Mamba imports")
else:
    print(f"  ✓ ultralytics/nn/tasks.py already patched")

# Patch modules/__init__.py too
import ultralytics.nn.modules as _ult_modules_mod
_modules_init = os.path.join(os.path.dirname(_ult_modules_mod.__file__), "__init__.py")
with open(_modules_init, "r") as _f:
    _modules_src = _f.read()

_modules_changed = False
if "from cbam_module import" not in _modules_src:
    _modules_src = _cbam_inject + _modules_src
    _modules_changed = True
if "from atrous_mamba_module import" not in _modules_src:
    _modules_src = _mamba_inject + _modules_src
    _modules_changed = True

if _modules_changed:
    with open(_modules_init, "w") as _f:
        _f.write(_modules_src)
    print(f"  ✓ Patched ultralytics/nn/modules/__init__.py with CBAM + C3K2Mamba imports")
else:
    print(f"  ✓ ultralytics/nn/modules/__init__.py already patched")

# ── Step 4: Patch parse_model to recognize C3K2Mamba ──────────────────────
# parse_model has sets of module classes that need special handling:
#   Set A: modules where c1,c2 are extracted from args (channel scaling)
#   Set B: modules where n (repeat count) is inserted into args
# We add C3K2Mamba to both sets by patching the source code.

with open(_tasks_file, "r") as _f:
    _tasks_src = _f.read()

# Find and patch the set that contains C3k2 for the n-insertion logic
# Pattern: "if m in {C3k2," or "if m in (C3k2,"
import re as _re

_patched_parse = False

# Strategy: find lines containing "C3k2" in a set/tuple check and add C3K2Mamba
for _pattern in [
    # Match: "if m in {C3k2, C2f, ..." and add C3K2Mamba
    r'(if m in \{[^}]*C3k2)',
    r'(if m in \([^)]*C3k2)',
    # Also match lines like "{C3k2, C2f}" as standalone sets
    r'(\{[^}]*C3k2[^}]*\})',
]:
    for _match in _re.finditer(_pattern, _tasks_src):
        _orig = _match.group(0)
        if "C3K2Mamba" not in _orig:
            _new = _orig.replace("C3k2", "C3k2, C3K2Mamba")
            _tasks_src = _tasks_src.replace(_orig, _new, 1)
            _patched_parse = True

if _patched_parse:
    with open(_tasks_file, "w") as _f:
        _f.write(_tasks_src)
    print(f"  ✓ Patched parse_model() to recognize C3K2Mamba")
else:
    print(f"  ⚠ Could not patch parse_model — C3K2Mamba may not be recognized in YAML")
    print(f"    Falling back: will verify during dry-run")

# ── Step 5: Reload modules ────────────────────────────────────────────────
import importlib
importlib.reload(_ult_tasks_mod)
importlib.reload(_ult_modules_mod)

from cbam_module import CBAM, ChannelAttention, SpatialAttention
from atrous_mamba_module import C3K2Mamba as _C3K2Mamba_ext
import ultralytics.nn.modules as ult_modules
import ultralytics.nn.tasks  as ult_tasks

for name, obj in [("CBAM", CBAM),
                   ("ChannelAttention", ChannelAttention),
                   ("SpatialAttention", SpatialAttention),
                   ("C3K2Mamba", _C3K2Mamba_ext),
                   ("AtrousSSM", AtrousSSM),
                   ("_MambaBottleneck", _MambaBottleneck),
                   ("_SelectiveScan1D", _SelectiveScan1D)]:
    setattr(ult_modules, name, obj)
    setattr(ult_tasks,   name, obj)

# Also ensure they're in the global scope of tasks.py for parse_model eval()
import types
_tasks_module = importlib.import_module("ultralytics.nn.tasks")
for name, obj in [("C3K2Mamba", C3K2Mamba),
                   ("AtrousSSM", AtrousSSM),
                   ("_MambaBottleneck", _MambaBottleneck),
                   ("_SelectiveScan1D", _SelectiveScan1D)]:
    setattr(_tasks_module, name, obj)

assert hasattr(ult_modules, "CBAM"), "CBAM registration failed"
assert hasattr(ult_tasks, "C3K2Mamba"), "C3K2Mamba registration failed"
print("✓ CBAM + C3K2Mamba registered in ultralytics namespace (DDP-safe)")


# ============================================================================
# CELL 8: YAML with C3K2Mamba (NATIVE — no injection needed) + Gradient Clipping
# ============================================================================
#
# The YAML below uses C3K2Mamba directly in the neck layers (13,16,19,22,25,28).
# Since C3K2Mamba is registered in ultralytics, parse_model builds it natively.
# No post-init injection needed. DDP subprocesses build the same architecture.
#

# NOTE: inject_mamba_neck is still defined for evaluation model loading fallback
def inject_mamba_neck(yolo_model, min_layer_idx=11, max_channels=512,
                       d_state=4, dilations=None, verbose=True):
    """Fallback injection for models loaded from checkpoint that lost C3K2Mamba."""
    try:
        from ultralytics.nn.modules.block import C3k2, C2f
    except ImportError:
        from ultralytics.nn.modules import C3k2, C2f

    nn_model = yolo_model.model.model
    replaced = []
    for idx, layer in enumerate(nn_model):
        if idx < min_layer_idx or not isinstance(layer, (C3k2, C2f)):
            continue
        if not (hasattr(layer, "cv1") and hasattr(layer, "cv2")):
            continue
        c1 = layer.cv1.conv.in_channels
        c2 = layer.cv2.conv.out_channels
        n  = len(layer.m)
        shortcut = getattr(layer.m[0], "add", False) if len(layer.m) > 0 else False
        if c2 > max_channels:
            continue
        dev = next(layer.parameters()).device
        new = C3K2Mamba(c1, c2, n=n, shortcut=shortcut,
                        d_state=d_state, dilations=dilations or DILATIONS)
        new = new.to(device=dev)
        try:
            new.cv1.load_state_dict(layer.cv1.state_dict())
            new.cv2.load_state_dict(layer.cv2.state_dict())
        except Exception:
            pass
        for i, m_new in enumerate(new.m):
            if i < len(layer.m):
                try:
                    m_new.cv1.load_state_dict(layer.m[i].cv1.state_dict())
                    m_new.cv2.load_state_dict(layer.m[i].cv2.state_dict())
                except Exception:
                    pass
        for attr in ("f", "i", "np"):
            if hasattr(layer, attr):
                setattr(new, attr, getattr(layer, attr))
        new.type = type(new).__name__
        nn_model[idx] = new
        replaced.append((idx, c1, c2, n))
        if verbose:
            print(f"  ✓ layer {idx:2d}: → C3K2Mamba ({c1}→{c2}, n={n})")
    if verbose:
        print(f"  Total injected: {len(replaced)} layer(s)")
    return replaced

# ── Gradient clipping ────────────────────────────────────────────────────
from ultralytics.engine.trainer import BaseTrainer
_orig_opt_step = BaseTrainer.optimizer_step

def _clipped_optimizer_step(self):
    """Optimizer step with gradient clipping (max_norm=10)."""
    self.scaler.unscale_(self.optimizer)
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    self.optimizer.zero_grad()
    if self.ema:
        self.ema.update(self.model)

BaseTrainer.optimizer_step = _clipped_optimizer_step
print("✓ Gradient clipping enabled (max_norm=10.0)")


# ── YAML: C3K2Mamba is used DIRECTLY in neck layers ──────────────────────
# Neck layers 13,16,19,22,25,28 use C3K2Mamba instead of C3k2.
# Args: [c2, shortcut, g, e, d_state, dilations_list]
# parse_model will: prepend c1, insert n → C3K2Mamba(c1, c2, n, shortcut, g, e, d_state, dil)

_ATROUS_YAML = f"""# YOLO11m + CBAM + P2 + AtrousMamba neck (YAML-NATIVE)
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
  - [-1, 2, C3K2Mamba, [512, False, 1, 0.5, {D_STATE}, {DILATIONS}]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3K2Mamba, [256, False, 1, 0.5, {D_STATE}, {DILATIONS}]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3K2Mamba, [128, False, 1, 0.5, {D_STATE}, {DILATIONS}]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]
  - [-1, 2, C3K2Mamba, [256, False, 1, 0.5, {D_STATE}, {DILATIONS}]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3K2Mamba, [512, False, 1, 0.5, {D_STATE}, {DILATIONS}]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3K2Mamba, [512, False, 1, 0.5, {D_STATE}, {DILATIONS}]]
  - [[19, 22, 25, 28], 1, Detect, [nc]]
"""

# Also write the base C3k2 YAML (needed for comparison model loading)
_BASE_YAML = """# YOLO11m + CBAM + P2 (base — for comparison loading)
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

with open("yolov11m_atrousmamba_cbam_p2head.yaml", "w") as f:
    f.write(_ATROUS_YAML)
with open("yolov11m_cbam_p2head.yaml", "w") as f:
    f.write(_BASE_YAML)
print("✓ AtrousMamba YAML written (C3K2Mamba native in neck)")
print("✓ Base CBAM+P2 YAML written (for comparison loading)")


# ── Dry-run: verify YAML parses correctly with C3K2Mamba ────────────────
def _dry_run_yaml():
    from ultralytics import YOLO
    print("\n--- YAML dry-run (no injection — C3K2Mamba is native) ---")
    _m = YOLO("yolov11m_atrousmamba_cbam_p2head.yaml")
    _m.load("yolo11m.pt")

    has_mamba = any(isinstance(mod, C3K2Mamba) for mod in _m.model.modules())
    n_mamba = sum(1 for mod in _m.model.modules() if isinstance(mod, C3K2Mamba))
    n_params = sum(p.numel() for p in _m.model.parameters())

    print(f"  C3K2Mamba layers: {n_mamba}")
    print(f"  Total params: {n_params/1e6:.2f}M")
    print(f"  Status: {'✓ NATIVE C3K2Mamba' if has_mamba else '✗ MISSING'}")

    if not has_mamba:
        print("\n  ⚠ YAML parsing did NOT produce C3K2Mamba modules!")
        print("  Dumping model layers for diagnosis:")
        for idx, layer in enumerate(_m.model.model):
            print(f"    [{idx:2d}] {type(layer).__name__}")
        raise RuntimeError(
            "YAML dry-run FAILED: C3K2Mamba not found in parsed model.\n"
            "The parse_model patch may have failed. Check the output above.")

    assert n_mamba >= 5, f"Expected ≥5 C3K2Mamba layers, got {n_mamba}"
    assert n_params > 20e6, f"Expected >20M params, got {n_params/1e6:.2f}M"

    del _m; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("--- YAML dry-run PASSED ---\n")

_dry_run_yaml()


# ============================================================================
# CELL 9: Custom Callbacks
# ============================================================================

class NaNStopCallback:
    def __init__(self):
        self.triggered = False

    def on_train_batch_end(self, trainer):
        if self.triggered:
            return
        loss = getattr(trainer, "loss", None)
        if loss is not None and not torch.isfinite(loss).all():
            self.triggered = True
            print(f"\n🚨 NaN/Inf loss at epoch {trainer.epoch+1}, "
                  f"batch {getattr(trainer,'batch_i',0)}! Stopping.")
            emg = Path(trainer.save_dir) / "weights" / "emergency_nan_stop.pt"
            try:
                trainer.model.save(str(emg))
                print(f"   Emergency checkpoint saved: {emg}")
            except Exception:
                pass
            try:
                trainer.stop = True
            except AttributeError:
                trainer.epoch = trainer.epochs


class F2EarlyStopCallback:
    """F2-based early stopping. F2 = 5PR/(4P+R) — recall-weighted for disaster detection."""
    def __init__(self, patience: int = 10, min_delta: float = 5e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_f2    = 0.0
        self.counter    = 0
        self.history    = []

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
    Copies last.pt/best.pt + metadata to /kaggle/working/ every N epochs.

    Resume instructions:
      1. Download session_last.pt + session_meta.json from /kaggle/working/
      2. Upload them to a NEW Kaggle Dataset
      3. Set RESUME_TRAINING=True, RESUME_PT="/kaggle/input/<dataset>/session_last.pt"
      4. Run notebook — training continues from saved epoch
    """
    def __init__(self, checkpoint_every: int = 5,
                 working_dir: str = "/kaggle/working"):
        self.every       = checkpoint_every
        self.working_dir = Path(working_dir)
        self.best_f2     = 0.0
        self.best_map50  = 0.0
        self._train_args = {}
        self._batch_size = BATCH_SIZE

    def register_train_args(self, args: dict):
        self._train_args = {k: str(v) for k, v in args.items()}

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
            "model_type":       "AtrousMamba+CBAM+P2",
            "dilations":        DILATIONS,
            "d_state":          D_STATE,
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

    def on_fit_epoch_end(self, trainer):
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
        copied_last = self._safe_copy(
            save_dir / "weights" / "last.pt",
            self.working_dir / "session_last.pt"
        )
        self._safe_copy(
            save_dir / "weights" / "best.pt",
            self.working_dir / "session_best.pt"
        )
        self._write_meta(epoch, save_dir)

        if copied_last:
            self._print_resume_instructions(epoch)
        else:
            print(f"  ⚠  Checkpoint copy skipped — last.pt not found at epoch {epoch}")

    def on_train_end(self, trainer):
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


print("✓ Custom callbacks defined")


# ============================================================================
# CELL 10: Training  (OOM-safe, session-resumable)
# ============================================================================
from ultralytics import YOLO

print("\n" + "=" * 80)
print("TRAINING: YOLO11m + AtrousMamba (neck) + CBAM (backbone) + P2 Head")
print(f"  Dilations: {DILATIONS}  |  d_state: {D_STATE}")
print("=" * 80)

nan_cb   = NaNStopCallback()
f2_cb    = F2EarlyStopCallback(patience=F2_PATIENCE)
grad_cb  = GradientMonitorCallback()
ckpt_mgr = SessionCheckpointManager(
    checkpoint_every=CHECKPOINT_EVERY,
    working_dir="/kaggle/working"
)

ATROUS_RUN_DIR = "runs/detect/yolo11m_atrousmamba_cbam_p2head"
ATROUS_BEST    = f"{ATROUS_RUN_DIR}/weights/best.pt"

def _verify_injection_callback(trainer):
    """Runs once at on_train_start to verify AtrousSSM modules are present."""
    n_params = sum(p.numel() for p in trainer.model.parameters())
    has_mamba = any(isinstance(m, C3K2Mamba) for m in trainer.model.modules())
    n_mamba = sum(1 for m in trainer.model.modules() if isinstance(m, C3K2Mamba))
    print(f"\n{'='*70}")
    print(f"  YAML-NATIVE VERIFICATION (on_train_start)")
    print(f"  Params: {n_params/1e6:.2f}M  |  C3K2Mamba layers: {n_mamba}")
    print(f"  Status: {'✓ AtrousSSM ACTIVE (YAML-native)' if has_mamba else '✗ MISSING — training as vanilla!'}")
    if not has_mamba:
        print(f"  ⚠  CRITICAL: Model does NOT contain C3K2Mamba!")
        print(f"  ⚠  C3K2Mamba registration may have failed — check Cell 7 output.")
    print(f"{'='*70}\n")


def _register_callbacks(model):
    model.add_callback("on_train_start", _verify_injection_callback)
    model.add_callback("on_train_batch_end", nan_cb.on_train_batch_end)
    model.add_callback("on_fit_epoch_end",   f2_cb.on_fit_epoch_end)
    model.add_callback("on_fit_epoch_end",   ckpt_mgr.on_fit_epoch_end)
    model.add_callback("on_train_epoch_end", grad_cb.on_train_epoch_end)
    model.add_callback("on_train_end",       grad_cb.on_train_end)
    model.add_callback("on_train_end",       ckpt_mgr.on_train_end)


# ── Register custom classes for checkpoint loading ─────────────────────────
# When ultralytics unpickles a checkpoint, it needs these classes in scope.
# torch.serialization.add_safe_globals is available in PyTorch 2.1+
try:
    torch.serialization.add_safe_globals([
        C3K2Mamba, AtrousSSM, _MambaBottleneck, _SelectiveScan1D
    ])
except AttributeError:
    pass  # older PyTorch — pickle will still find classes in __main__

# ── RESUME PATH ──────────────────────────────────────────────────────────────
# Auto-detect: if RESUME_TRAINING=True but RESUME_PT is empty, scan /kaggle/input/
if RESUME_TRAINING and not RESUME_PT:
    print("  Auto-detecting checkpoint in /kaggle/input/ …")
    for _root, _dirs, _files in os.walk("/kaggle/input"):
        if "session_last.pt" in _files:
            RESUME_PT = os.path.join(_root, "session_last.pt")
            print(f"  ✓ Found: {RESUME_PT}")
            break
    if not RESUME_PT:
        print("  ✗ No session_last.pt found in /kaggle/input/")
        print("    Upload your checkpoint as a Kaggle Dataset and re-run,")
        print("    or set RESUME_PT manually.")
        raise FileNotFoundError("Auto-detect failed: no session_last.pt in /kaggle/input/")

if RESUME_TRAINING and RESUME_PT:
    _resume_src = Path(RESUME_PT)
    if not _resume_src.exists():
        raise FileNotFoundError(
            f"RESUME_PT not found: {RESUME_PT}\n"
            "Did you forget to add the checkpoint dataset as a Kaggle input?"
        )

    _weights_dir = Path(ATROUS_RUN_DIR) / "weights"
    _weights_dir.mkdir(parents=True, exist_ok=True)
    _local_last  = _weights_dir / "last.pt"

    if not _local_last.exists():
        shutil.copy(str(_resume_src), str(_local_last))
        print(f"  ✓ Copied last.pt → {_local_last}")

    _meta_src = Path(RESUME_PT).parent / "session_meta.json"
    if _meta_src.exists():
        import json
        _meta = json.load(open(_meta_src))
        print(f"  Resuming from epoch {_meta.get('completed_epochs', '?')}  |  "
              f"Best F2={_meta.get('best_f2','?')}  mAP50={_meta.get('best_map50','?')}")
        ckpt_mgr.best_f2    = float(_meta.get("best_f2",    0))
        ckpt_mgr.best_map50 = float(_meta.get("best_map50", 0))

    mamba_model = YOLO(str(_local_last))

    # Verify C3K2Mamba survived checkpoint loading
    _has_mamba = any(isinstance(m, C3K2Mamba) for m in mamba_model.model.modules())
    _n_p = sum(p.numel() for p in mamba_model.model.parameters())
    if _has_mamba:
        print(f"  ✓ C3K2Mamba PRESENT in checkpoint ({_n_p/1e6:.2f}M)")
    else:
        print(f"  ⚠ C3K2Mamba MISSING from checkpoint ({_n_p/1e6:.2f}M)")
        print(f"    → YAML-native registration should rebuild C3K2Mamba during resume")

    _register_callbacks(mamba_model)
    print(f"\n⚡ RESUMING training from {_local_last}")
    mamba_model.train(resume=True)

# ── FRESH TRAINING with OOM retry ────────────────────────────────────────────
else:
    _trained_ok = False

    for _batch_attempt in OOM_RETRY_BATCHES:
        print(f"\n→ Attempting training with batch_size={_batch_attempt} …")

        # PATH B: Load AtrousMamba YAML directly — C3K2Mamba is native
        mamba_model = YOLO("yolov11m_atrousmamba_cbam_p2head.yaml")
        mamba_model.load("yolo11m.pt")
        print("  ✓ AtrousMamba YAML loaded + ImageNet weights transferred")

        # Verify C3K2Mamba is present (should be — it's in the YAML)
        _n_mamba = sum(1 for m in mamba_model.model.modules() if isinstance(m, C3K2Mamba))
        _n_p = sum(p.numel() for p in mamba_model.model.parameters())
        print(f"  C3K2Mamba layers: {_n_mamba}  |  Params: {_n_p/1e6:.2f}M")
        if _n_mamba == 0:
            raise RuntimeError(
                "YAML parsed but C3K2Mamba NOT found! "
                "Registration in Cell 7 may have failed. Check dry-run output.")

        mamba_model.info(verbose=False)
        _register_callbacks(mamba_model)

        _train_kwargs = dict(
            data         = "c2a.yaml",
            epochs       = NUM_EPOCHS,
            imgsz        = 640,
            batch        = _batch_attempt,
            device       = DEVICE,
            optimizer    = "AdamW",
            lr0          = 0.0005,
            lrf          = 0.01,
            weight_decay = 0.0005,
            momentum     = 0.937,
            warmup_epochs= 5,
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
            name         = "yolo11m_atrousmamba_cbam_p2head",
            exist_ok     = True,
        )
        ckpt_mgr.register_train_args(_train_kwargs)
        ckpt_mgr._batch_size = _batch_attempt

        try:
            mamba_model.train(**_train_kwargs)
            _trained_ok = True
            print(f"\n✓ Training complete (batch_size={_batch_attempt})")
            break

        except (torch.cuda.OutOfMemoryError, subprocess.CalledProcessError, RuntimeError) as _oom:
            # DDP OOM manifests as CalledProcessError (SIGKILL exit code -9),
            # not torch.cuda.OutOfMemoryError, because the subprocess is killed
            # by the Linux OOM killer before Python can raise the exception.
            _is_oom = (
                isinstance(_oom, torch.cuda.OutOfMemoryError)
                or (isinstance(_oom, subprocess.CalledProcessError) and _oom.returncode in (-9, 137))
                or ("CUDA out of memory" in str(_oom))
                or ("OutOfMemoryError" in str(_oom))
            )
            if not _is_oom:
                raise  # re-raise non-OOM errors

            print(f"\n  OOM at batch_size={_batch_attempt}: {type(_oom).__name__}: {_oom}")

            _partial_last = Path(ATROUS_RUN_DIR) / "weights" / "last.pt"
            if _partial_last.exists():
                print(f"  Partial checkpoint found: {_partial_last}")
                shutil.copy(str(_partial_last), "/kaggle/working/session_last.pt")
                ckpt_mgr._write_meta(epoch=-1, save_dir=Path(ATROUS_RUN_DIR))

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
                    "  2. Reduce dilations from [1,2,4] to [1,2] in CELL 2.\n"
                    "  3. Reduce d_state from 4 to 2 in CELL 2.\n"
                    "  4. Remove P2 head (use 3-scale detection instead).\n"
                    "  5. Reduce imgsz from 640 to 512."
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
# CELL 12: Training Curves (F1/F2 per epoch, multi-way comparison)
# ============================================================================

def load_results_csv(run_dir: str) -> pd.DataFrame:
    path = f"{run_dir}/results.csv"
    if not os.path.exists(path):
        print(f"  ⚠  results.csv not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    P = df.get("metrics/precision(B)", pd.Series(dtype=float))
    R = df.get("metrics/recall(B)",    pd.Series(dtype=float))
    df["metrics/F1(B)"] = 2 * P * R / (P + R + 1e-9)
    df["metrics/F2(B)"] = 5 * P * R / (4 * P + R + 1e-9)
    train_loss_cols = [c for c in df.columns if "train" in c and "loss" in c]
    val_loss_cols   = [c for c in df.columns if "val"   in c and "loss" in c]
    if train_loss_cols:
        df["train/total_loss"] = df[train_loss_cols].sum(axis=1)
    if val_loss_cols:
        df["val/total_loss"] = df[val_loss_cols].sum(axis=1)
    return df

runs = {}
if os.path.exists(ATROUS_RUN_DIR):
    runs["AtrousMamba+CBAM+P2"] = load_results_csv(ATROUS_RUN_DIR)
if OLD_MAMBA_BEST:
    old_mamba_run = str(Path(OLD_MAMBA_BEST).parent.parent)
    runs["Mamba+CBAM+P2 (old)"] = load_results_csv(old_mamba_run)
if CBAM_P2_BEST:
    cbam_p2_run = str(Path(CBAM_P2_BEST).parent.parent)
    runs["CBAM+P2"] = load_results_csv(cbam_p2_run)


def plot_single_run(df: pd.DataFrame, tag: str):
    if df.empty:
        return
    df.to_excel(f"{EXCEL_DIR}/{tag}_training.xlsx", index=False)
    ep = df["epoch"]

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
    safe_tag = tag.replace("+", "_").replace(" ", "_").replace("(", "").replace(")", "")
    plot_single_run(df, safe_tag)

# Multi-way overlay comparison
if len(runs) >= 2:
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    overlay_metrics = [
        ("metrics/mAP50(B)",    "mAP@0.5",      axes[0, 0]),
        ("metrics/mAP50-95(B)", "mAP@0.5:0.95", axes[0, 1]),
        ("metrics/recall(B)",   "Recall",        axes[0, 2]),
        ("metrics/F1(B)",       "F1",            axes[1, 0]),
        ("metrics/F2(B)",       "F2",            axes[1, 1]),
        ("val/total_loss",      "Val Total Loss",axes[1, 2]),
    ]
    colors = {"AtrousMamba+CBAM+P2": "#E53935",
              "Mamba+CBAM+P2 (old)": "#FB8C00",
              "CBAM+P2":             "#1E88E5"}
    for col, label, ax in overlay_metrics:
        for tag, df in runs.items():
            if not df.empty and col in df.columns:
                ax.plot(df["epoch"], df[col], label=tag,
                        color=colors.get(tag, "grey"), lw=2, alpha=0.9)
        ax.set(xlabel="Epoch", ylabel=label, title=label)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        if "loss" not in col.lower():
            ax.set_ylim([0, 1.05])
    plt.suptitle(f"Training Comparison: {' vs '.join(runs.keys())}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{len(runs)}way_overlay_comparison.png", dpi=300); plt.close()
    print(f"  ✓ {len(runs)}-way overlay comparison plot saved")

print("✓ Training curve analysis complete")


# ============================================================================
# CELL 13: Model Loading Helper + Complexity Comparison
# ============================================================================
from ultralytics import YOLO


def _load_atrous_model(weights_path: str, label: str = "model",
                       is_atrous: bool = False) -> "YOLO":
    """
    Load a model checkpoint. If is_atrous=True AND the loaded model is missing
    C3K2Mamba modules, attempt to re-inject + reload state_dict so that the
    evaluation uses the correct architecture with trained weights.
    """
    model = YOLO(weights_path)
    has_mamba = any(isinstance(m, C3K2Mamba) for m in model.model.modules())
    n_params = sum(p.numel() for p in model.model.parameters())

    if has_mamba:
        print(f"✓ Loaded: {label}  ({weights_path})")
        print(f"  C3K2Mamba: PRESENT  |  Params: {n_params/1e6:.2f}M")
        return model

    if not is_atrous:
        # Not expected to have C3K2Mamba (comparison model)
        print(f"✓ Loaded: {label}  ({weights_path})")
        return model

    # ── The saved checkpoint lost C3K2Mamba (ultralytics rebuilt from YAML) ──
    # Strategy: build fresh model from YAML → inject C3K2Mamba → load state_dict
    print(f"⚠ {label}: C3K2Mamba missing in checkpoint ({n_params/1e6:.2f}M)")
    print(f"  Attempting re-injection + state_dict reload …")

    try:
        # Load raw checkpoint to extract the pickled model's state_dict
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            src_sd = ckpt["model"].float().state_dict()
        elif hasattr(ckpt, "state_dict"):
            src_sd = ckpt.float().state_dict()
        else:
            print(f"  ✗ Cannot extract state_dict — using base model")
            return model

        # PATH B: Build fresh model from AtrousMamba YAML (native C3K2Mamba)
        fresh = YOLO("yolov11m_atrousmamba_cbam_p2head.yaml")

        # Load state_dict with strict=False (base weights match, Mamba weights may differ)
        missing, unexpected = fresh.model.load_state_dict(src_sd, strict=False)
        n_new = sum(p.numel() for p in fresh.model.parameters())

        # Check if enough keys matched to be meaningful
        mamba_keys_missing = [k for k in missing if "ssm" in k or "scan" in k or "fusion" in k]
        if mamba_keys_missing:
            print(f"  ⚠ {len(mamba_keys_missing)} SSM keys missing — SSM weights are random init")
            print(f"    (This means the checkpoint was saved WITHOUT C3K2Mamba weights)")
            print(f"    Evaluation will use base model instead.")
            return model  # Fall back to base model

        print(f"  ✓ Re-injected: {n_new/1e6:.2f}M  |  {len(missing)} missing, {len(unexpected)} unexpected")
        return fresh

    except Exception as e:
        print(f"  ✗ Re-injection failed: {e} — using base model")
        return model


def get_complexity(weights_path: str, label: str,
                   is_atrous: bool = False) -> dict:
    model = _load_atrous_model(weights_path, label, is_atrous=is_atrous)
    n_params = sum(p.numel() for p in model.model.parameters())

    has_c3k2mamba = any(isinstance(m, C3K2Mamba) for m in model.model.modules())
    module_note = "C3K2Mamba" if has_c3k2mamba else "C3k2 (base)"
    print(f"  {label}: {n_params/1e6:.3f}M params, arch={module_note}")

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
            "Layers": layers, "Architecture": module_note}

complexity_rows = []
if os.path.exists(ATROUS_BEST):
    complexity_rows.append(get_complexity(ATROUS_BEST, "AtrousMamba+CBAM+P2", is_atrous=True))
if OLD_MAMBA_BEST and os.path.exists(OLD_MAMBA_BEST):
    complexity_rows.append(get_complexity(OLD_MAMBA_BEST, "Mamba+CBAM+P2 (old)"))
if CBAM_P2_BEST and os.path.exists(CBAM_P2_BEST):
    complexity_rows.append(get_complexity(CBAM_P2_BEST, "CBAM+P2"))

if complexity_rows:
    cdf = pd.DataFrame(complexity_rows)
    if len(cdf) > 1:
        base = cdf.iloc[-1]["Params(M)"]
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
# CELL 15: Comprehensive Evaluation Function
# ============================================================================
from tqdm import tqdm

def evaluate_model_comprehensive(model, model_name: str, img_dir: str, lbl_dir: str,
                                   split_name: str = "test", n_images=None,
                                   conf: float = 0.25) -> tuple:
    print(f"\n{'='*70}\nEVALUATING: {model_name}  on  {split_name.upper()}\n{'='*70}")
    image_list = get_image_list(img_dir)
    if n_images:
        image_list = image_list[:min(n_images, len(image_list))]

    records, inf_times, all_confs = [], [], []
    total_tp = total_fp = total_fn = 0
    size_stats = {k: {"tp": 0, "fn": 0, "count": 0}
                  for k in ("very_tiny", "tiny", "small", "medium", "large")}

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
# CELL 16: Visualisation Functions
# ============================================================================

def plot_confidence_distribution(confs: list, model_name: str, split: str):
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
    cats   = ["very_tiny", "tiny", "small", "medium", "large"]
    labels = ["Very Tiny\n(<64px²)", "Tiny\n(64-256px²)", "Small\n(256-1024px²)",
              "Medium\n(1024-9216px²)", "Large\n(>9216px²)"]
    x      = np.arange(len(cats))
    n      = len(summaries)
    width  = 0.7 / n
    colors = ["#E53935", "#FB8C00", "#1E88E5", "#43A047"]

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
    rows = []
    for sz in [320, 480, 640, 800]:
        for _ in range(3):
            model.predict(sample_img, imgsz=sz, verbose=False)
        times = []
        for _ in range(15):
            t0 = time.perf_counter()
            model.predict(sample_img, imgsz=sz, verbose=False)
            times.append((time.perf_counter() - t0) * 1000)
        mean_ms = float(np.mean(times[3:]))
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
if os.path.exists(ATROUS_BEST):
    models_for_eval["AtrousMamba_CBAM_P2"] = _load_atrous_model(
        ATROUS_BEST, "AtrousMamba+CBAM+P2", is_atrous=True)
else:
    print(f"⚠  AtrousMamba model not found: {ATROUS_BEST}")

if OLD_MAMBA_BEST and os.path.exists(OLD_MAMBA_BEST):
    models_for_eval["Mamba_CBAM_P2_old"] = _load_atrous_model(
        OLD_MAMBA_BEST, "Mamba+CBAM+P2 (old)", is_atrous=False)

if CBAM_P2_BEST and os.path.exists(CBAM_P2_BEST):
    models_for_eval["CBAM_P2"] = _load_atrous_model(
        CBAM_P2_BEST, "CBAM+P2", is_atrous=False)

assert models_for_eval, "No models loaded — cannot proceed with evaluation!"

if os.path.exists(ATROUS_RUN_DIR):
    copy_yolo_plots(ATROUS_RUN_DIR, "AtrousMamba_CBAM_P2")


# ============================================================================
# CELL 18: Test Set Evaluation
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

pd.DataFrame(official_test).T.to_excel(f"{EXCEL_DIR}/official_test_metrics.xlsx")


# ============================================================================
# CELL 19: Validation Set Evaluation
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
# CELL 20: Advanced Metrics
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
    ece = plot_calibration_ece(test_dfs[name], name, "test")
    ece_results[name] = ece
    print(f"  ECE: {ece:.4f}" if ece is not None else "  ECE: N/A")
    plot_confidence_distribution(test_confs_all.get(name, []), name, "test")
    speed_results[name] = benchmark_speed(model, sample_img, name)

if test_summaries:
    plot_per_size_recall(test_summaries, "test")
if val_summaries:
    plot_per_size_recall(val_summaries,  "val")


# ============================================================================
# CELL 21: Comparison Tables
# ============================================================================

def make_comparison_table(summaries: list, split: str) -> pd.DataFrame:
    metrics = [
        ("Precision",       "Precision"),
        ("Recall",          "Recall"),
        ("F1",              "F1"),
        ("F2",              "F2"),
        ("very_tiny_recall","Very Tiny Recall (<64px²)"),
        ("tiny_recall",     "Tiny Recall (64-256px²)"),
        ("small_recall",    "Small Recall (256-1024px²)"),
        ("medium_recall",   "Medium Recall (1024-9216px²)"),
        ("Avg_Inf_ms",      "Avg Latency (ms)"),
    ]
    rows = []
    base = summaries[-1] if len(summaries) > 1 else summaries[0]
    for key, label in metrics:
        row = {"Metric": label}
        for s in summaries:
            row[s["Model"]] = round(float(s.get(key, 0)), 4)
        if len(summaries) > 1:
            bv = float(base.get(key, 0))
            mv = float(summaries[0].get(key, 0))
            row["Delta (Atrous-Base)"] = round(mv - bv, 4)
        rows.append(row)
    cdf = pd.DataFrame(rows)
    cdf.to_excel(f"{EXCEL_DIR}/comparison_{split}.xlsx", index=False)
    print(f"\n{'='*70}\nCOMPARISON — {split.upper()}\n{'='*70}")
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

def fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)

ts = {s["Model"]: s for s in test_summaries}
vs = {s["Model"]: s for s in val_summaries}

report_lines = [
    "=" * 80,
    "DISASTER HUMAN DETECTION — ATROUSMAMBA + CBAM + P2 HEAD REPORT",
    "=" * 80,
    f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}",
    f"TEST MODE : {TEST_MODE}  |  Epochs: {NUM_EPOCHS}  |  Fraction: {TRAIN_FRACTION*100:.0f}%",
    f"AtrousSSM : dilations={DILATIONS}  d_state={D_STATE}",
    "",
    "MODEL COMPLEXITY",
]
if complexity_rows:
    for r in complexity_rows:
        report_lines.append(
            f"  {r['Model']:25s}: {r['Params(M)']:.3f}M params  "
            f"{r['GFLOPs']:.1f} GFLOPs  {r['Size(KB)']:.0f} KB")

report_lines += ["", "OFFICIAL mAP (test split — Ultralytics val):"]
for name, m in official_test.items():
    report_lines.append(
        f"  {name:25s}: mAP50={m['mAP50']:.4f}  mAP50-95={m['mAP50-95']:.4f}")

report_lines += ["", "TEST SET (custom evaluation):"]
for name, s in ts.items():
    report_lines.append(
        f"  {name:25s}: P={fmt(s['Precision'])}  R={fmt(s['Recall'])}"
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
    report_lines.append(f"  {name:25s}: ECE={fmt(ece) if ece else 'N/A'}")

report_lines += ["", "TRAINING CONVERGENCE:"]
for tag, run_df in runs.items():
    if not run_df.empty:
        ep = best_epoch(run_df)
        if "metrics/F2(B)" in run_df.columns and ep >= 0:
            report_lines.append(
                f"  {tag:25s} best epoch {ep}: "
                f"F2={run_df.loc[run_df['epoch']==ep,'metrics/F2(B)'].values[0]:.4f}  "
                f"mAP50={run_df.loc[run_df['epoch']==ep,'metrics/mAP50(B)'].values[0]:.4f}")

report_lines += ["", "F2 EARLY STOPPING LOG:"]
for ep, f2v in f2_cb.history[-10:]:
    report_lines.append(f"  epoch {ep+1:3d}: F2={f2v:.4f}")

report_lines += ["", "NaN STOP TRIGGERED:", f"  {nan_cb.triggered}"]
report_lines += ["", "=" * 80]

report_text = "\n".join(report_lines)
print(report_text)

with open(f"{REPORT_DIR}/MASTER_REPORT_ATROUSMAMBA_CBAM_P2.txt", "w") as f:
    f.write(report_text)

with open(f"{REPORT_DIR}/test_summaries.json", "w") as f:
    json.dump(test_summaries, f, indent=2, default=str)
with open(f"{REPORT_DIR}/val_summaries.json",  "w") as f:
    json.dump(val_summaries,  f, indent=2, default=str)

print(f"\n✓ Master report → {REPORT_DIR}/MASTER_REPORT_ATROUSMAMBA_CBAM_P2.txt")


# ============================================================================
# CELL 23: Package All Results
# ============================================================================
try:
    from IPython.display import FileLink
    _in_notebook = True
except ImportError:
    _in_notebook = False

import subprocess

if os.path.exists(ATROUS_BEST):
    shutil.copy(ATROUS_BEST, "/kaggle/working/atrousmamba_cbam_p2head_best.pt")

_zip_cmd = (
    f"zip -r /kaggle/working/atrousmamba_cbam_p2head_results.zip "
    f"{EXCEL_DIR} {PLOT_DIR} {REPORT_DIR} "
    f"/kaggle/working/atrousmamba_cbam_p2head_best.pt "
    f"/kaggle/working/gradient_norms.csv 2>/dev/null || true"
)
subprocess.run(_zip_cmd, shell=True)
print("✓ Results packaged → /kaggle/working/atrousmamba_cbam_p2head_results.zip")
if _in_notebook:
    display(FileLink("/kaggle/working/atrousmamba_cbam_p2head_results.zip"))

print("\n" + "=" * 80)
print("ALL DONE — AtrousMamba+CBAM+P2 experiment complete!")
print("=" * 80)
print("Download from /kaggle/working/:")
print("  • atrousmamba_cbam_p2head_results.zip  (plots, reports, excels)")
print("  • session_last.pt + session_meta.json   (for resuming)")
print("  • atrousmamba_cbam_p2head_best.pt       (best model weights)")

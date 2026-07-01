"""
joint_c2a_sard_train.py
=======================

PHASE 2 -- THE DEPLOYABLE MODEL (joint C2A + SARD training).

Goal: ONE CBAM+P2 model that is strong on BOTH disaster scenes (C2A) AND real
search-and-rescue humans (SARD) -- i.e. a model a drone could actually use, not the
C2A-only model whose zero-shot SARD transfer collapsed to ~0.

Why this design (docs/2026-06-16_endgame_plan_and_deployable_model.md):
  * C2A-only is NOT deployable -- proven by your own zero-shot SARD result (~0.006 mAP50).
  * The C2A paper's own Table 5: C2A-only -> SARD 0.259, but "General-Human + C2A" -> SARD
    0.660 AND C2A-val 0.874. Mixing real human data with C2A is the proven path to a model
    good on both. This script does the C2A+SARD analogue.
  * Architecture = CBAM+P2 (deployment choice: ~14.6 ms, best small-object recall). NOT Mamba
    (41 ms, +0 accuracy -- disqualified for edge real-time). So NO Mamba-injection patch needed:
    CBAM is YAML-native and survives Ultralytics' train()-rebuild.

What it does:
  1. Auto-discovers C2A + SARD roots and (default) the C2A-trained CBAM+P2 best.pt to init from
     (keeps the strong C2A model; CBAM deserializes because it's registered at import). Or start
     fresh from yolo11m.pt (INIT_FROM_C2A_BEST=False).
  2. Builds an explicit joint image-list: C2A-train (all) + SARD-train (x SARD_OVERSAMPLE, to
     offset C2A's ~6:1 size dominance). Same for val. Reliable oversampling via a .txt file.
  3. Trains CBAM+P2 on the joint set (AdamW lr0=0.001, cos, the proven chain config).
  4. Evaluates the best.pt SEPARATELY on C2A-test AND SARD-test (identical protocol to the
     C2A chain + sard_eval.py): ultralytics mAP, COCO AP_small/med/large, per-size recall.
  5. Writes a deployable_summary.json + a comparison vs the C2A-only model (which had SARD~0).

RUN (AnyDesk PC, after the few-shot phase):
    E:\\Thesis_mofazzal_2007074\\mofazzal1\\Scripts\\Activate.ps1
    cd "E:\\Thesis_mofazzal_2007074\\cross_dataset_SARD"
    python joint_c2a_sard_train.py          # SMOKE_TEST=True first, then False

OUTPUTS
    runs_joint/<run_id>/                      training run + weights/best.pt (THE deployable model)
    runs_joint/<run_id>/metrics/deployable_summary.json   C2A-test AND SARD-test results
    runs_joint/<run_id>/metrics/per_size_{c2a,sard}.csv
"""

# =============================================================================
# QA / TEST CASES  (manual verification log -- PC-2: 2x RTX A6000 48GB, Windows, venv "2007074")
# Run order: TC-00 (env) -> TC-01 (smoke, SMOKE_TEST=True) -> flip SMOKE_TEST=False -> TC-02..TC-10.
# Legend: [PASS]=observed OK | [FIX]=bug found & fixed | [DESIGN]=guaranteed by code, confirm on use.
# -----------------------------------------------------------------------------
# TC-00  Environment / venv active
#   Steps : activate venv -> prompt shows "(2007074)"; python -c "import torch; print(torch.cuda.is_available())"
#   Expect: True. If `import torch` -> ModuleNotFoundError, the venv is NOT active (prompt lacks "(2007074)").
#   Status: [PASS] torch 2.12.1+cu126, cuda.is_available()=True, device_count()=2.
#
# TC-01  Pipeline smoke (SMOKE_TEST=True)
#   Steps : SMOKE_TEST=True -> python joint_c2a_sard_train.py
#   Expect: 2-epoch train on 2% data -> dual eval (C2A+SARD) -> COCO -> "DEPLOYABLE MODEL RESULT", no traceback.
#   Status: [PASS] 2026-06-25  C2A mAP50=0.838 (init retained); SARD mAP50~0.0076 (expected low: 2ep/2%).
#
# TC-02  GPU auto-pick on a shared box (GPU 0 = other user, GPU 1 = mine)
#   Steps : launch while the other user occupies GPU 0; read the first "[gpu] usage ..." line.
#   Expect: GPU 0 busy (>1GB) -> picks GPU 1; both free -> GPU 1; neither free -> GPU 1 + WARNING.
#   Status: [PASS] observed "free: [1] -> picking GPU 1" (GPU0 ~3GB/64% other user); CUDA_VISIBLE_DEVICES=1.
#
# TC-03  VRAM must NOT spill into Windows shared memory (the speed killer)
#   Steps : watch the training progress-bar "GPU_mem" column + `nvidia-smi -i 1`.
#   Expect: stays ~37-38 G (well under 48 G); s/it ~0.7-1.0. NEVER 48+ G.
#   Status: [FIX] batch 40 -> 52.6 G spill -> ~4.4 s/it (FAIL). batch 28 + GPU_MEM_FRACTION=0.90 cap -> no spill.
#
# TC-04  Power-cut / crash resume (failproof for load-shedding)
#   Steps : Ctrl+C (or kill power) mid-run; re-run the SAME command (SMOKE_TEST=False).
#   Expect: "[resume] incomplete joint run found ... resuming from last.pt"; continues, not from epoch 0.
#           Corrupt last.pt -> "restoring from epochXX.pt"; no checkpoint yet -> starts fresh (no crash).
#   Status: [DESIGN] find_incomplete_joint_run + _ckpt_loadable + _newest_healthy_joint_ckpt + save_period=25.
#
# TC-05  Init from the C2A CBAM+P2 best.pt
#   Steps : ensure EXPLICIT_C2A_BEST_PT exists; read "[model] init from C2A-trained CBAM+P2: ...".
#   Expect: loads it (CBAM deserializes via register_cbam); C2A-test stays ~0.85 -> proves init worked.
#   Status: [PASS] C2A mAP50=0.838 in TC-01 confirms the checkpoint loaded. (Falls back to yolo11m.pt if missing.)
#
# TC-06  Reproducibility / determinism
#   Steps : inspect run env.json + startup log.
#   Expect: seed=0, deterministic=True, cache='disk', cudnn.benchmark OFF, TF32 OFF; env.json logs all
#           hyperparams + gpu_physical_id. Two runs with identical inputs match.
#   Status: [DESIGN] configured; cache='disk' chosen over 'ram' to avoid Ultralytics' non-determinism warning.
#
# TC-07  Dataset discovery + SARD single-class collapse
#   Steps : read "[data] C2A=..."/"[data] SARD=..." and "[sard] collapsed labels -> person(0)".
#   Expect: both roots auto-found under SEARCH_ROOTS; all SARD labels class 0 (backup in labels_orig/).
#   Status: [PASS] both roots found in TC-01; SARD test=570 imgs; labels collapsed.
#
# TC-08  Windows worker GPU-pick guard (no duplicate nvidia-smi per worker)
#   Steps : count "[gpu] CUDA_VISIBLE_DEVICES=..." lines at startup with NUM_WORKERS=16.
#   Expect: exactly ONE (main proc); workers inherit via _JOINT_GPU_PINNED sentinel.
#   Status: [FIX] earlier printed ~7x; sentinel guard added -> now single line.
#
# TC-09  OOM batch ladder (now a REAL OOM thanks to the mem cap, not a silent spill)
#   Steps : (only if a batch is too big) observe "[train] OOM at batch=NN -- shrinking".
#   Expect: drops 28 -> 24 -> 20 -> 16 until it fits; raises only if all fail.
#   Status: [DESIGN] not triggered at batch 28; cap converts would-be spill into a catchable OOM.
#
# TC-10  Eval completeness (system_spec_thesis.md Sec 6, BOTH test sets)
#   Steps : after training, check runs_joint/<id>/metrics/ + plots/.
#   Expect: per-test-set mAP/COCO AP_s/m/l, per-size recall, per-image P/R/F1/F2, PR & F1-vs-conf curves,
#           calibration ECE/MCE/Brier, confusion, + one efficiency block (params/GFLOPs/latency/FPS).
#   Status: [PASS] all CSVs/plots + deployable_summary.json written in TC-01.
# =============================================================================

# =============================================================================
# 0. CONFIG
# =============================================================================
MODEL_TAG          = "cbam_p2head_joint_c2a_sard"
# --- PC-2 (A6000 box) GPU pinning: AUTO-PICK the GPU that is 100% free ---
# Shared box. At launch we query BOTH GPUs via nvidia-smi and pick one that is essentially
# idle (<FREE_MEM_MB in use AND <FREE_UTIL_PCT util). Rules:
#   * BOTH free   -> pick your ASSIGNED GPU (GPU_ID_FALLBACK = 1), so you never sit on the other
#                    user's GPU 0 when your own is free.
#   * ONE  free   -> pick that one (if only GPU 0 is free, use it).
#   * NEITHER free-> stay on the assigned GPU_ID_FALLBACK (1) and warn (do NOT auto-grab the
#                    other user's GPU when both are busy -> "only for me" safe choice).
#   * nvidia-smi unreadable -> fall back to GPU_ID_FALLBACK.
# The chosen physical GPU is pinned via CUDA_VISIBLE_DEVICES BEFORE torch import, so it is the
# ONLY visible device (= cuda:0 in-process) and any other user's GPU is invisible -> zero risk
# of touching it. device=0 in code == the picked physical GPU.
GPU_AUTO_SELECT    = True        # True: auto-pick the free GPU. False: force GPU_ID_FALLBACK.
GPU_ID_FALLBACK    = 1           # used if auto-select OFF, OR nvidia-smi fails, OR neither GPU is free.
FREE_MEM_MB        = 1024        # a GPU counts as "free" if <1 GB is in use (driver/display only)...
FREE_UTIL_PCT      = 10          # ...AND utilization < this %. Multi-GB in use => another job => skip it.
GPU_MEM_FRACTION   = 0.90        # HARD cap on THIS process's share of the 48 GB GPU. On Windows WDDM,
                                 # CUDA silently SPILLS past-VRAM allocations into "shared GPU memory"
                                 # (system RAM over PCIe = ~10x slower) instead of raising OOM. This cap
                                 # forces a real OOM instead -> the batch ladder catches it AND we never
                                 # spill. 0.90 of 48 GB = ~43 GB for torch, leaving room for the desktop.
INIT_FROM_C2A_BEST = True       # True: start from the C2A-trained CBAM+P2 best.pt (recommended).
                                # False: build CBAM+P2 fresh from yolo11m.pt and joint-train.
PRETRAINED_WEIGHTS = "yolo11m.pt"
# On PC-2 the C2A run dir may not be present for auto-glob -- copy the C2A CBAM+P2 best.pt to THIS
# exact path and it is used directly. Set to None to rely on auto-discovery under SEARCH_ROOTS.
EXPLICIT_C2A_BEST_PT = r"D:\student_2k20\2007074\c2a_cbam_p2head_best.pt"

CBAM_REDUCTION   = 16
CBAM_KERNEL_SIZE = 7

# Joint-data composition
SARD_OVERSAMPLE  = 1            # SARD train (~4041 imgs) vs C2A train (6129) is already ~40:60 --
                                # balanced, so 1 is the safe default. SARD images are augmented
                                # (~1400 unique source scenes), so oversampling risks overfitting
                                # those scenes. If C2A-test stays high but SARD-test underperforms,
                                # bump to 2 and watch C2A-test doesn't drop.
COLLAPSE_SARD_LABELS_TO_PERSON = True

# Training (proven chain config)
NUM_EPOCHS  = 300              # matches your other runs (baseline/CBAM/P2 = 300) for consistency.
                               # It inits from the converged C2A best.pt (an adaptation, not from
                               # scratch), so patience=50 early-stop will very likely cut well before
                               # 300 -> 300 is just a safe ceiling, not wasted compute.
PATIENCE    = 50
IMG_SIZE    = 640
# A6000 48 GB. MEASURED: batch 40 needed 52.6 GB -> OVERFLOWED the card into Windows "shared GPU
# memory" (system RAM over PCIe) and crawled at ~4.4 s/it. On WDDM, CUDA does NOT raise OOM when it
# exceeds VRAM -- it silently spills, so the ladder never fired. Fix = batch 28 (~37-38 GB, fits fully
# in VRAM with headroom for the desktop on GPU 1) + the GPU_MEM_FRACTION cap so it can NEVER spill.
# ~1.25 GB/sample measured. Effective batch = NOMINAL_BATCH (=BATCH_SIZE here -> accumulate 1).
BATCH_SIZE  = 28                # fits fully in VRAM (no spill) -> ~4-6x faster than the spilling batch 40.
NOMINAL_BATCH = 28              # = BATCH_SIZE -> no grad accumulation, effective batch 28
OPTIMIZER   = "AdamW"
LR0         = 0.0005          # was 0.001; HALVED for this joint fine-tune after the 0.001+TF32 run
                              # diverged to NaN ~epoch 75. A fine-tune from a converged model legitimately
                              # uses a gentler lr; the from-scratch chain runs keep 0.001 (separate, stable).
LRF         = 0.01
COS_LR      = True
NUM_WORKERS = 16               # A6000 box has many cores; feed the GPU so util stays >90%
CACHE       = "disk"           # DETERMINISTIC: Ultralytics warns cache='ram' "may produce non-deterministic
                               # training results" -> 'disk' preserves your seed/reproducibility guarantee.
                               # Builds a small .npy cache on the SSD (1.6 TB free); nearly as fast, GPU stays fed.
SEED        = 0
OOM_RETRY_BATCHES = [BATCH_SIZE, 24, 20, 16]   # 28 -> drop if OOM (now REAL: the mem cap forces OOM, not a silent spill)

# Eval thresholds -- IDENTICAL to the chain (spec Sec 5)
CONF_AP, IOU_AP = 0.001, 0.7
CONF_OP, IOU_OP = 0.25, 0.5

SMOKE_TEST  = False             # smoke validated -> REAL run. (Set True for a 2-epoch tiny dry-run.)
SMOKE_FRACTION = 0.02

CLASS_NAMES = ["person"]
NC = 1

# =============================================================================
# 1. IMPORTS + PACKAGE GUARD
# =============================================================================
import os, sys, json, time, gc, shutil, math, traceback, subprocess

# --- pick the free GPU BEFORE importing torch (CUDA_VISIBLE_DEVICES must be set pre-import) ---
def _query_gpus_via_nvsmi():
    """[{index, mem_used, mem_total, util}, ...] (MB / %) from nvidia-smi, or None on failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL, timeout=15
        ).decode("utf-8", "ignore")
    except Exception:
        return None
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append({"index": int(parts[0]), "mem_used": float(parts[1]),
                         "mem_total": float(parts[2]), "util": float(parts[3])})
        except ValueError:
            continue
    return gpus or None

def _pick_gpu() -> int:
    if not GPU_AUTO_SELECT:
        print(f"[gpu] auto-select OFF -> using assigned GPU {GPU_ID_FALLBACK}")
        return GPU_ID_FALLBACK
    gpus = _query_gpus_via_nvsmi()
    if not gpus:
        print(f"[gpu] nvidia-smi query failed -> falling back to assigned GPU {GPU_ID_FALLBACK}")
        return GPU_ID_FALLBACK
    snap = ", ".join(f"GPU{g['index']}:{int(g['mem_used'])}MB/{int(g['util'])}%" for g in gpus)
    free_idx = [g["index"] for g in gpus if g["mem_used"] < FREE_MEM_MB and g["util"] < FREE_UTIL_PCT]
    if free_idx:
        # prefer your ASSIGNED GPU (GPU_ID_FALLBACK) when it's free; else take the only free one.
        chosen = GPU_ID_FALLBACK if GPU_ID_FALLBACK in free_idx else min(free_idx)
        print(f"[gpu] usage [{snap}] -> free: {free_idx} -> picking GPU {chosen}")
        return chosen
    print(f"[gpu] WARNING: no fully-free GPU [{snap}] -> staying on assigned GPU {GPU_ID_FALLBACK} "
          f"(it may be shared right now; consider waiting until it's free).")
    return GPU_ID_FALLBACK

# On Windows the DataLoader workers RE-IMPORT this module (spawn), which would re-run the picker
# in every worker (the duplicate "[gpu] ..." lines + a redundant nvidia-smi per worker). Fix: pick
# ONCE in the main process, set a sentinel env var, and let spawned workers inherit the choice.
_GPU_ALREADY_PINNED = os.environ.get("_JOINT_GPU_PINNED") == "1"
if _GPU_ALREADY_PINNED:                                  # spawned worker: reuse the parent's pick
    GPU_ID = int((os.environ.get("CUDA_VISIBLE_DEVICES") or str(GPU_ID_FALLBACK)).split(",")[0])
else:                                                    # main process: pick once, pin, mark
    GPU_ID = _pick_gpu()
    # Pin GPU BEFORE importing torch -> the chosen physical GPU becomes the ONLY visible device
    # (cuda:0 in-process); any other user's GPU is invisible. device=0 in code == physical GPU_ID.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    os.environ["_JOINT_GPU_PINNED"] = "1"                # children inherit this -> they won't re-pick
    print(f"[gpu] CUDA_VISIBLE_DEVICES={GPU_ID} (this process sees it as cuda:0; the other GPU is invisible)")
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

def _ensure(pkgs):
    import importlib, subprocess
    namemap = {"opencv-python": "cv2", "PyYAML": "yaml"}
    for name in pkgs:
        mod = namemap.get(name, name.replace("-", "_"))
        try:
            importlib.import_module(mod)
        except ImportError:
            print(f"[deps] installing {name} ...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", name])
            except Exception as e:
                print(f"[deps] WARN could not install {name}: {e}")
_ensure(["numpy", "pandas", "PyYAML", "matplotlib", "opencv-python", "pycocotools", "openpyxl"])

import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn as nn
# TF32 DESTABILIZED this joint fine-tune: the lr0=0.001 + TF32 run diverged to NaN ~epoch 75
# (mAP collapsed 0.86 -> 0). Your stable chain runs never used TF32. Disabled -> back to the
# proven-stable fp32 numerics. (Costs ~30% conv speed; stability >> speed here.)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# cudnn.benchmark is intentionally OFF: training runs with deterministic=True + pinned SEED for
# REPRODUCIBILITY, and benchmark's autotuned (run-varying) algorithm choice would break that.
# (Ultralytics deterministic=True also enforces this + sets CUBLAS_WORKSPACE_CONFIG.)
if torch.cuda.is_available() and not _GPU_ALREADY_PINNED:   # main proc only (workers must NOT init CUDA)
    try:
        torch.cuda.set_per_process_memory_fraction(GPU_MEM_FRACTION, 0)  # HARD cap -> no WDDM spillover
    except Exception as _e:
        print(f"[gpu] WARN: set_per_process_memory_fraction failed: {_e}")
    print(f"[gpu] visible device (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}): "
          f"{torch.cuda.get_device_name(0)} | {round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB"
          f" | mem cap {int(GPU_MEM_FRACTION*100)}% (~{round(GPU_MEM_FRACTION*48,1)} GB) -> cannot spill")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAS_PYCOCO = True
except Exception:
    HAS_PYCOCO = False

try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"FATAL: ultralytics import failed -- {e}")
    sys.exit(1)

# =============================================================================
# 2. CBAM MODULES + REGISTRATION (verbatim -- needed to deserialize the C2A CBAM+P2 ckpt
#    AND so the YAML 'CBAM' layer parses if building fresh)
# =============================================================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1); self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        a = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        m = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return x * self.sigmoid(a + m)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=3 if kernel_size == 7 else 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class CBAM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.reduction = 16; self.kernel_size = 7
        if len(args) == 1 and isinstance(args[0], int) and args[0] <= 32:
            self.reduction = args[0]
        elif len(args) == 2 and isinstance(args[0], int) and args[0] <= 32:
            self.reduction = args[0]; self.kernel_size = args[1] if isinstance(args[1], int) else 7
        elif len(args) >= 4:
            self.reduction = args[2] if isinstance(args[2], int) else 16
            self.kernel_size = args[3] if isinstance(args[3], int) else 7
        self.reduction = kwargs.get("reduction", self.reduction)
        self.kernel_size = kwargs.get("kernel_size", self.kernel_size)
        if self.kernel_size not in (3, 7):
            self.kernel_size = 7
        self._initialized = False
        self.channel_attention = None; self.spatial_attention = None; self._channels = None
    def _lazy_init(self, channels, device, dtype):
        self._channels = channels
        self.channel_attention = ChannelAttention(channels, self.reduction).to(device=device, dtype=dtype)
        self.spatial_attention = SpatialAttention(self.kernel_size).to(device=device, dtype=dtype)
        self._initialized = True
    def forward(self, x):
        if not self._initialized:
            self._lazy_init(x.size(1), x.device, x.dtype)
        return self.spatial_attention(self.channel_attention(x))

def register_cbam():
    import ultralytics.nn.modules as _m
    import ultralytics.nn.tasks as _t
    for ns in (_m, _t):
        ns.CBAM = CBAM; ns.ChannelAttention = ChannelAttention; ns.SpatialAttention = SpatialAttention
    for cls in (CBAM, ChannelAttention, SpatialAttention):
        globals()[cls.__name__] = cls
register_cbam()

CBAM_P2HEAD_YAML = f"""# YOLOv11m + CBAM + P2 Extra Detection Head
nc: {{nc}}
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
  - [-1, 1, CBAM, [{CBAM_REDUCTION}, {CBAM_KERNEL_SIZE}]]
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
  - [[19, 22, 25, 28], 1, Detect, [{{nc}}]]
"""

# =============================================================================
# 3. EVAL HELPERS (identical protocol to the chain / sard_eval.py)
# =============================================================================
def _box_iou_xyxy(a, b):
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    al = a[:, None, :]; bl = b[None, :, :]
    ix1 = np.maximum(al[..., 0], bl[..., 0]); iy1 = np.maximum(al[..., 1], bl[..., 1])
    ix2 = np.minimum(al[..., 2], bl[..., 2]); iy2 = np.minimum(al[..., 3], bl[..., 3])
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-12)

def _yolo_label_to_xyxy(label_path: Path, w: int, h: int):
    if not label_path.is_file():
        return np.zeros((0, 4), dtype=np.float32)
    out = []
    for ln in label_path.read_text().splitlines():
        p = ln.strip().split()
        if len(p) < 5:
            continue
        _, cx, cy, bw, bh = (float(x) for x in p[:5])
        out.append([(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h])
    return np.asarray(out, dtype=np.float32) if out else np.zeros((0, 4), dtype=np.float32)

def _list_images(img_dir: Path) -> List[Path]:
    files = []
    for e in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        files += list(img_dir.glob(e))
    return sorted(files)

def per_size_recall(model, img_dir: Path, lbl_dir: Path) -> pd.DataFrame:
    import cv2
    bins = [("very_tiny",0,8),("tiny",8,16),("small",16,32),("medium",32,96),("large",96,1_000_000)]
    tally = {n: {"matched":0,"total":0} for n,*_ in bins}
    for ip in _list_images(img_dir):
        im = cv2.imread(str(ip))
        if im is None:
            continue
        h, w = im.shape[:2]
        gt = _yolo_label_to_xyxy(lbl_dir / (ip.stem + ".txt"), w, h)
        if gt.shape[0] == 0:
            continue
        b = model.predict(str(ip), conf=CONF_OP, verbose=False)[0].boxes
        pb = (b.xyxy.cpu().numpy().astype(np.float32) if (b is not None and len(b) > 0)
              else np.zeros((0,4), np.float32))
        iou = _box_iou_xyxy(pb, gt) if pb.shape[0] else np.zeros((0, gt.shape[0]))
        gtm = (iou.max(axis=0) >= 0.5) if (pb.shape[0] and iou.size) else np.zeros(gt.shape[0], bool)
        side = np.sqrt((gt[:,2]-gt[:,0])*(gt[:,3]-gt[:,1]))
        for n, lo, hi in bins:
            mask = (side >= lo) & (side < hi)
            tally[n]["total"] += int(mask.sum()); tally[n]["matched"] += int(gtm[mask].sum())
    return pd.DataFrame([{"bin":n,"gt_total":tally[n]["total"],"matched":tally[n]["matched"],
                          "recall":(tally[n]["matched"]/tally[n]["total"]) if tally[n]["total"] else float("nan")}
                         for n,lo,hi in bins])

def build_coco_gt_from_yolo(img_dir: Path, lbl_dir: Path, out_json: Path) -> Optional[Path]:
    if out_json.is_file():
        return out_json
    import cv2
    images, anns, aid = [], [], 1
    for iid, ip in enumerate(_list_images(img_dir), start=1):
        im = cv2.imread(str(ip))
        if im is None:
            continue
        h, w = im.shape[:2]
        images.append({"id": iid, "file_name": ip.name, "width": w, "height": h})
        for (x1,y1,x2,y2) in _yolo_label_to_xyxy(lbl_dir / (ip.stem + ".txt"), w, h):
            bw, bh = float(x2-x1), float(y2-y1)
            anns.append({"id": aid, "image_id": iid, "category_id": 1, "bbox": [float(x1),float(y1),bw,bh],
                         "area": bw*bh, "iscrowd": 0}); aid += 1
    if not images:
        return None
    out_json.write_text(json.dumps({"images": images, "annotations": anns,
                                    "categories": [{"id": 1, "name": "person"}]}))
    return out_json

def coco_ap_eval(model, img_dir: Path, coco_json: Path, work: Path) -> Optional[Dict[str, float]]:
    if not (HAS_PYCOCO and coco_json and coco_json.is_file()):
        return None
    try:
        gt = COCO(str(coco_json))
        f2i = {im["file_name"]: im["id"] for im in gt.loadImgs(gt.getImgIds())}
        cats = gt.getCatIds(); dets = []
        for fn, iid in f2i.items():
            p = img_dir / fn
            if not p.is_file():
                continue
            b = model.predict(str(p), conf=CONF_AP, verbose=False)[0].boxes
            if b is None or len(b) == 0:
                continue
            cid = cats[0] if cats else 1
            for (x1,y1,x2,y2), s in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy()):
                dets.append({"image_id": int(iid), "category_id": int(cid),
                             "bbox": [float(x1),float(y1),float(x2-x1),float(y2-y1)], "score": float(s)})
        dp = work / "coco_dets.json"; dp.write_text(json.dumps(dets))
        ev = COCOeval(gt, gt.loadRes(str(dp)), "bbox"); ev.evaluate(); ev.accumulate(); ev.summarize()
        s = ev.stats
        return {"AP":float(s[0]),"AP50":float(s[1]),"AP75":float(s[2]),"AP_small":float(s[3]),
                "AP_medium":float(s[4]),"AP_large":float(s[5]),"AR_100":float(s[8])}
    except Exception as e:
        print(f"[coco] eval failed: {e}")
        return None

# =============================================================================
# 4. PATHS + DATASET DISCOVERY
# =============================================================================
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path(os.getcwd()).resolve()
OUT_ROOT = SCRIPT_DIR
RUNS_DIR = OUT_ROOT / "runs_joint"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

SEARCH_ROOTS = [p for p in [
    Path(os.environ["PROJECT_ROOT"]) if os.environ.get("PROJECT_ROOT") else None,
    SCRIPT_DIR.parent, SCRIPT_DIR.parent.parent, SCRIPT_DIR.parent.parent.parent,
    Path(r"D:\student_2k20\2007074"),                 # PC-2 (A6000 box) project root
    Path(r"D:\2007074"),                              # PC-3 project root
    Path(r"E:\Thesis_mofazzal_2007074"), Path(r"D:\Academics\thesis folder\Last Month"),
] if p is not None]

def _has_split(root: Path, split: str) -> bool:
    return (root / split / "images").is_dir() and (root / split / "labels").is_dir()

def find_c2a_root() -> Path:
    env = os.environ.get("C2A_ROOT")
    cands = [Path(env)] if env else []
    for r in SEARCH_ROOTS:
        cands += [r / "common" / "c2a" / "C2A_Dataset" / "new_dataset3",
                  r / "common" / "c2a" / "new_dataset3"]
    for c in cands:
        if c and _has_split(c, "train"):
            return c
    raise FileNotFoundError("C2A dataset not found (need train/val/test with images/+labels/). "
                            "Set $env:C2A_ROOT.")

def find_sard_root() -> Path:
    env = os.environ.get("SARD_ROOT")
    cands = [Path(env)] if env else []
    for r in SEARCH_ROOTS:
        cands += [r / "common" / "sard", r / "common" / "sard" / "search-and-rescue",
                  r / "sard"]
    nested = []
    for c in list(cands):
        try:
            if c and c.is_dir():
                nested += [d for d in c.iterdir() if d.is_dir()]
        except Exception:
            pass
    cands += nested
    for c in cands:
        if c and _has_split(c, "train"):
            return c
    raise FileNotFoundError("SARD dataset not found (need train/test with images/+labels/). "
                            "Set $env:SARD_ROOT.")

def pick_split(root: Path, prefs: List[str]) -> Tuple[Path, Path, str]:
    for s in prefs:
        if _has_split(root, s):
            return root / s / "images", root / s / "labels", s
    raise FileNotFoundError(f"No split {prefs} under {root}")

def collapse_labels_to_person(lbl_dir: Path):
    if not COLLAPSE_SARD_LABELS_TO_PERSON:
        return
    backup = lbl_dir.parent / "labels_orig"
    needs = any(
        (len(ln.split()) >= 5 and ln.split()[0] != "0")
        for f in lbl_dir.glob("*.txt") for ln in f.read_text().splitlines()
    )
    if not needs:
        return
    if not backup.exists():
        shutil.copytree(lbl_dir, backup)
    for f in lbl_dir.glob("*.txt"):
        out = []
        for ln in f.read_text().splitlines():
            p = ln.strip().split()
            if len(p) >= 5:
                p[0] = "0"; out.append(" ".join(p))
        f.write_text("\n".join(out) + ("\n" if out else ""))
    print(f"[sard] collapsed labels -> person(0) in {lbl_dir}")

def write_image_list(txt_path: Path, entries: List[Tuple[Path, int]]) -> int:
    """entries = [(images_dir, repeat), ...]. Writes absolute image paths (repeat x)."""
    lines = []
    for img_dir, rep in entries:
        imgs = [str(p).replace("\\", "/") for p in _list_images(img_dir)]
        lines += imgs * rep
    txt_path.write_text("\n".join(lines) + "\n")
    return len(lines)

def discover_c2a_cbam_p2_best() -> Optional[Path]:
    if EXPLICIT_C2A_BEST_PT and Path(EXPLICIT_C2A_BEST_PT).is_file():
        return Path(EXPLICIT_C2A_BEST_PT)
    found, found_run = None, ""
    for r in SEARCH_ROOTS:
        if not r.is_dir():
            continue
        for best in r.glob("**/runs/*/weights/best.pt"):
            run = best.parent.parent
            summ = run / "metrics" / "summary.json"
            if not summ.is_file():
                continue
            try:
                tag = json.loads(summ.read_text()).get("model_tag")
            except Exception:
                continue
            if tag == "yolo11m_cbam_p2head" and run.name > found_run:
                found, found_run = best, run.name
    return found

# --- power-cut robustness: resume + corrupt-checkpoint recovery (failproof) ---
def _ckpt_loadable(p: Path) -> bool:
    """True if the .pt is a complete zip (a power-cut-truncated write fails this)."""
    try:
        import zipfile
        return zipfile.is_zipfile(p) and zipfile.ZipFile(p).testzip() is None
    except Exception:
        return False

def _newest_healthy_joint_ckpt(rdir: Path) -> Optional[Path]:
    wdir = rdir / "ultra" / "weights"
    cands = sorted(wdir.glob("epoch*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
    if (wdir / "best.pt").is_file():
        cands.append(wdir / "best.pt")
    for c in cands:
        if _ckpt_loadable(c):
            return c
    return None

def find_incomplete_joint_run() -> Optional[Path]:
    """Newest joint run dir with a checkpoint but NO deployable_summary.json (crashed /
    power-cut mid-run) -> resume it instead of starting a fresh 100-epoch run."""
    if not RUNS_DIR.is_dir():
        return None
    for d in sorted([p for p in RUNS_DIR.iterdir()
                     if p.is_dir() and p.name.endswith(MODEL_TAG)],
                    key=lambda x: x.name, reverse=True):
        if (d / "metrics" / "deployable_summary.json").is_file():
            continue  # already completed
        if (d / "ultra" / "weights" / "last.pt").is_file():
            return d
    return None

# =============================================================================
# 5. MAIN
# =============================================================================
def _scalar(x):
    try:
        return float(x.mean()) if hasattr(x, "mean") else float(x)
    except Exception:
        return float("nan")

# --- system_spec_thesis.md Sec 6 MUST-have metric helpers (verbatim protocol) ---
def per_image_eval(model, img_dir: Path, lbl_dir: Path, conf=CONF_OP, iou_match=IOU_OP) -> pd.DataFrame:
    import cv2
    rows = []
    for ip in _list_images(img_dir):
        im = cv2.imread(str(ip))
        if im is None:
            continue
        h, w = im.shape[:2]
        gt = _yolo_label_to_xyxy(lbl_dir / (ip.stem + ".txt"), w, h)
        t0 = time.perf_counter()
        b = model.predict(str(ip), conf=conf, verbose=False)[0].boxes
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        if b is None or b.xyxy is None or len(b) == 0:
            pred = np.zeros((0, 4), np.float32); cf = np.zeros((0,), np.float32)
        else:
            pred = b.xyxy.cpu().numpy().astype(np.float32); cf = b.conf.cpu().numpy().astype(np.float32)
        tp = fp = fn = 0
        if pred.shape[0] == 0:
            fn = gt.shape[0]
        elif gt.shape[0] == 0:
            fp = pred.shape[0]
        else:
            iou = _box_iou_xyxy(pred, gt); matched = set()
            for pi in np.argsort(-cf):
                j = int(np.argmax(iou[pi]))
                if iou[pi, j] >= iou_match and j not in matched:
                    tp += 1; matched.add(j)
                else:
                    fp += 1
            fn = gt.shape[0] - len(matched)
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        rows.append({"image": ip.name, "gt": int(gt.shape[0]), "TP": tp, "FP": fp, "FN": fn,
                     "precision": round(p, 6), "recall": round(r, 6),
                     "F1": round(2*p*r/max(p+r, 1e-12), 6), "F2": round(5*p*r/max(4*p+r, 1e-12), 6),
                     "inference_ms": round(dt, 3)})
    return pd.DataFrame(rows)

def pr_and_f1_curves(model, img_dir: Path, lbl_dir: Path):
    import cv2
    grid = np.arange(0.0, 1.001, 0.01)
    PB, PC, GT = [], [], []
    for ip in _list_images(img_dir):
        im = cv2.imread(str(ip))
        if im is None:
            continue
        h, w = im.shape[:2]
        GT.append(_yolo_label_to_xyxy(lbl_dir / (ip.stem + ".txt"), w, h))
        b = model.predict(str(ip), conf=0.001, verbose=False)[0].boxes
        if b is None or len(b) == 0:
            PB.append(np.zeros((0, 4), np.float32)); PC.append(np.zeros((0,), np.float32))
        else:
            PB.append(b.xyxy.cpu().numpy().astype(np.float32)); PC.append(b.conf.cpu().numpy().astype(np.float32))
    allc = np.concatenate(PC) if PC else np.array([])
    hist, edges = np.histogram(allc, bins=50, range=(0, 1))
    hist_df = pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": hist})
    rows = []
    for c in grid:
        TP = FP = FN = 0
        for pb, pc, gt in zip(PB, PC, GT):
            k = pc >= c; pbf = pb[k]; pcf = pc[k]
            if pbf.shape[0] == 0:
                FN += gt.shape[0]; continue
            if gt.shape[0] == 0:
                FP += pbf.shape[0]; continue
            iou = _box_iou_xyxy(pbf, gt); m = set()
            for pi in np.argsort(-pcf):
                j = int(np.argmax(iou[pi]))
                if iou[pi, j] >= 0.5 and j not in m:
                    TP += 1; m.add(j)
                else:
                    FP += 1
            FN += gt.shape[0] - len(m)
        prec = TP/max(TP+FP, 1); rec = TP/max(TP+FN, 1)
        rows.append({"conf": round(float(c), 3), "precision": prec, "recall": rec,
                     "F1": 2*prec*rec/max(prec+rec, 1e-12), "F2": 5*prec*rec/max(4*prec+rec, 1e-12),
                     "TP": TP, "FP": FP, "FN": FN})
    pr = pd.DataFrame(rows)
    return pr, pr[["conf", "F1", "F2", "precision", "recall"]].copy(), hist_df

def calibration_table(confs: np.ndarray, correct: np.ndarray, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1); rows = []; ece = mce = 0.0
    tot = max(len(confs), 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        msk = (confs >= lo) & (confs < hi + (1e-9 if i == n_bins-1 else 0))
        n = int(msk.sum())
        if n == 0:
            rows.append({"bin_lo": lo, "bin_hi": hi, "count": 0, "mean_conf": float("nan"),
                         "mean_acc": float("nan"), "gap": float("nan")}); continue
        mc = float(confs[msk].mean()); ma = float(correct[msk].mean()); gap = abs(mc-ma)
        ece += (n/tot)*gap; mce = max(mce, gap)
        rows.append({"bin_lo": lo, "bin_hi": hi, "count": n, "mean_conf": mc, "mean_acc": ma, "gap": gap})
    brier = float(np.mean((confs-correct)**2)) if len(confs) else float("nan")
    return pd.DataFrame(rows), ece, mce, brier

def _calibration_pairs(model, img_dir: Path, lbl_dir: Path):
    import cv2
    cf, ok = [], []
    for ip in _list_images(img_dir):
        im = cv2.imread(str(ip))
        if im is None:
            continue
        h, w = im.shape[:2]
        gt = _yolo_label_to_xyxy(lbl_dir / (ip.stem + ".txt"), w, h)
        b = model.predict(str(ip), conf=0.001, verbose=False)[0].boxes
        if b is None or len(b) == 0:
            continue
        pb = b.xyxy.cpu().numpy().astype(np.float32); pc = b.conf.cpu().numpy().astype(np.float32)
        if gt.shape[0] == 0:
            cf += list(pc); ok += [0.0]*len(pc); continue
        iou = _box_iou_xyxy(pb, gt); m = set()
        for pi in np.argsort(-pc):
            j = int(np.argmax(iou[pi])); good = (iou[pi, j] >= 0.5) and (j not in m)
            if good:
                m.add(j)
            cf.append(float(pc[pi])); ok.append(1.0 if good else 0.0)
    return np.asarray(cf), np.asarray(ok)

def architecture_report(model) -> Dict[str, Any]:
    info = {}
    try:
        info["params_total_M"] = round(sum(p.numel() for p in model.model.parameters())/1e6, 4)
        info["layers_total"] = sum(1 for _, m in model.model.named_modules() if len(list(m.children())) == 0)
    except Exception:
        pass
    try:
        import thop
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=model.device)
        macs, _ = thop.profile(model.model, inputs=(dummy,), verbose=False)
        info["gflops"] = round(macs/1e9*2, 3)
    except Exception as e:
        info["gflops_error"] = str(e)
    return info

def latency_profile(model, warmup=30, runs=200) -> Dict[str, float]:
    try:
        dev = model.device
        for _ in range(warmup):
            with torch.no_grad():
                _ = model.model(torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=dev))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts = []
        for _ in range(runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model.model(torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=dev))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ts.append((time.perf_counter()-t0)*1000.0)
        a = np.asarray(ts)
        return {"latency_mean_ms": float(a.mean()), "latency_p50_ms": float(np.percentile(a, 50)),
                "latency_p95_ms": float(np.percentile(a, 95)), "fps_at_640": float(1000.0/a.mean())}
    except Exception as e:
        return {"latency_error": str(e)}


def evaluate_on(model, name: str, img_dir: Path, lbl_dir: Path, root: Path, split: str,
                coco_json: Optional[Path], rdir: Path) -> Dict[str, Any]:
    """Full system_spec_thesis.md Sec 6 detection-quality + calibration suite, per test set."""
    print(f"\n[eval] === {name} (split={split}, {len(_list_images(img_dir))} imgs) ===")
    mdir = rdir / "metrics"; pdir = rdir / "plots"; pdir.mkdir(parents=True, exist_ok=True)
    head: Dict[str, Any] = {}
    print(f"[eval] {name}: [1/6] ultralytics .val (mAP)...", flush=True)
    # 1) Ultralytics .val -> P/R/mAP50/mAP50-95/mAP75
    try:
        yml = rdir / f"_eval_{name}.yaml"
        rel = f"{split}/images"
        yml.write_text(yaml.safe_dump({"path": str(root).replace("\\", "/"),
                                       "train": rel, "val": rel, "names": {0: "person"}, "nc": 1},
                                      sort_keys=False))
        box = model.val(data=str(yml), split="val", verbose=False, conf=CONF_AP, iou=IOU_AP, imgsz=IMG_SIZE).box
        head = {"precision": _scalar(box.p), "recall": _scalar(box.r),
                "mAP50": _scalar(box.map50), "mAP50-95": _scalar(box.map),
                "mAP75": _scalar(getattr(box, "map75", float("nan")))}
    except Exception as e:
        print(f"[eval] {name} .val failed: {e}")
    # 2) COCO AP_small/medium/large + AR
    print(f"[eval] {name}: [2/6] COCO AP (silent loop over test imgs, ~1-2 min)...", flush=True)
    coco = coco_ap_eval(model, img_dir, coco_json, mdir) if coco_json else None
    if coco:
        head.update(coco)
    # 3) per-size recall (very-tiny..large)
    print(f"[eval] {name}: [3/6] per-size recall...", flush=True)
    psz = per_size_recall(model, img_dir, lbl_dir)
    psz.to_csv(mdir / f"per_size_{name}.csv", index=False)
    head["per_size_recall"] = {r["bin"]: (None if pd.isna(r["recall"]) else round(float(r["recall"]), 4))
                               for _, r in psz.iterrows()}
    # 4) per-image P/R/F1/F2 (operational, conf=0.25) + confusion counts
    print(f"[eval] {name}: [4/6] per-image P/R/F1/F2...", flush=True)
    per_img = per_image_eval(model, img_dir, lbl_dir)
    per_img.to_csv(mdir / f"per_image_{name}.csv", index=False)
    if not per_img.empty:
        head["F1_mean"] = float(per_img["F1"].mean()); head["F2_mean"] = float(per_img["F2"].mean())
        TP, FP, FN = int(per_img["TP"].sum()), int(per_img["FP"].sum()), int(per_img["FN"].sum())
        head["confusion_TP_FP_FN"] = [TP, FP, FN]
        head["e2e_latency_mean_ms"] = float(per_img["inference_ms"].mean())
    # 5) PR / F1-vs-conf / confidence histogram + optimal thresholds
    print(f"[eval] {name}: [5/6] PR / F1-vs-conf curves (SLOWEST step; silent, several min)...", flush=True)
    try:
        pr, f1c, hist = pr_and_f1_curves(model, img_dir, lbl_dir)
        pr.to_csv(mdir / f"pr_curve_{name}.csv", index=False)
        f1c.to_csv(mdir / f"f1_vs_conf_{name}.csv", index=False)
        hist.to_csv(mdir / f"confidence_hist_{name}.csv", index=False)
        i1, i2 = int(f1c["F1"].idxmax()), int(f1c["F2"].idxmax())
        head["opt_thresholds"] = {"OptThr_F1": float(f1c.loc[i1, "conf"]), "Best_F1": float(f1c.loc[i1, "F1"]),
                                  "OptThr_F2": float(f1c.loc[i2, "conf"]), "Best_F2": float(f1c.loc[i2, "F2"])}
        fig, ax = plt.subplots(figsize=(6, 6)); ax.plot(pr["recall"], pr["precision"])
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"PR ({name})"); fig.tight_layout(); fig.savefig(pdir / f"pr_curve_{name}.png", dpi=300); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(f1c["conf"], f1c["F1"], label="F1"); ax.plot(f1c["conf"], f1c["F2"], label="F2")
        ax.set_xlabel("confidence"); ax.set_title(f"F1/F2 vs conf ({name})"); ax.legend(); fig.tight_layout()
        fig.savefig(pdir / f"f1_vs_conf_{name}.png", dpi=300); plt.close(fig)
    except Exception as e:
        print(f"[eval] {name} PR/F1 curves failed: {e}")
    # 6) calibration ECE/MCE/Brier
    print(f"[eval] {name}: [6/6] calibration (ECE/MCE/Brier)...", flush=True)
    try:
        cf, ok = _calibration_pairs(model, img_dir, lbl_dir)
        cal, ece, mce, brier = calibration_table(cf, ok)
        cal.to_csv(mdir / f"calibration_{name}.csv", index=False)
        head["calibration"] = {"ECE": ece, "MCE": mce, "Brier": brier}
    except Exception as e:
        print(f"[eval] {name} calibration failed: {e}")
    # per-size recall bar plot
    try:
        fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(psz["bin"], psz["recall"]); ax.set_ylim(0, 1)
        ax.set_ylabel("Recall"); ax.set_title(f"per-size recall ({name})"); fig.tight_layout()
        fig.savefig(pdir / f"per_size_recall_{name}.png", dpi=300); plt.close(fig)
    except Exception:
        pass
    print(f"[eval] {name}: mAP50={head.get('mAP50')} mAP50-95={head.get('mAP50-95')} "
          f"COCO-AP={head.get('AP')} F2={head.get('F2_mean')} vt_recall={head['per_size_recall'].get('very_tiny')}")
    return head

# --- safety net: abort the moment training loss goes non-finite (NaN/Inf) for 2 epochs running.
# The TF32-on / lr0=0.001 run diverged to NaN ~epoch 75 then wasted ~30 epochs before stopping; this
# halts immediately (no wasted time) and best.pt -- the last healthy epoch -- is preserved. The 2-in-a-
# row rule avoids stopping on a single recoverable blip. Fully guarded: it can never break training.
_nan_streak = {"n": 0}
def _abort_on_nan(trainer):
    try:
        import math as _math
        t = getattr(trainer, "tloss", None)
        if t is None:
            t = getattr(trainer, "loss", None)
        if t is None:
            return
        v = float(t.detach().float().mean()) if hasattr(t, "detach") else float(t)
        if not _math.isfinite(v):
            _nan_streak["n"] += 1
            print(f"[guard] non-finite training loss ({v}) at epoch {getattr(trainer, 'epoch', '?')} "
                  f"(streak {_nan_streak['n']}/2)", flush=True)
            if _nan_streak["n"] >= 2:
                print("[guard] 2 consecutive non-finite epochs -> STOPPING now; best.pt holds the last "
                      "healthy model. (If this ever fires, lower LR0 further or set amp=False.)", flush=True)
                trainer.stop = True
        else:
            _nan_streak["n"] = 0
    except Exception:
        pass  # a guard must NEVER break training

def main():
    print("="*78 + "\nJOINT C2A + SARD TRAINING -- the DEPLOYABLE CBAM+P2 model\n" + "="*78)
    if not torch.cuda.is_available():
        print("[warn] CUDA not available -- training will be extremely slow.")

    c2a = find_c2a_root(); sard = find_sard_root()
    c2a_tr_i, c2a_tr_l, _ = pick_split(c2a, ["train"])
    c2a_va_i, c2a_va_l, c2a_va_s = pick_split(c2a, ["val", "valid"])
    c2a_te_i, c2a_te_l, c2a_te_s = pick_split(c2a, ["test", "val"])
    sard_tr_i, sard_tr_l, _ = pick_split(sard, ["train"])
    sard_va_i, sard_va_l, sard_va_s = pick_split(sard, ["valid", "val", "test"])
    sard_te_i, sard_te_l, sard_te_s = pick_split(sard, ["test", "valid", "val"])
    print(f"[data] C2A  = {c2a}")
    print(f"[data] SARD = {sard}")
    for d in (sard_tr_l, sard_va_l, sard_te_l):
        collapse_labels_to_person(d)

    # resume detection (failproof: long run on a load-shedding machine)
    resume_dir = None if SMOKE_TEST else find_incomplete_joint_run()
    if resume_dir is not None:
        rdir = resume_dir; run_id = rdir.name; resuming = True
        print(f"[resume] incomplete joint run found -> {run_id} (resuming from last.pt)")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{ts}_{MODEL_TAG}" + ("_SMOKE" if SMOKE_TEST else "")
        rdir = RUNS_DIR / run_id; resuming = False
    (rdir / "metrics").mkdir(parents=True, exist_ok=True)
    (rdir / "weights").mkdir(parents=True, exist_ok=True)

    # ---- joint image lists ----
    ov = 1 if SMOKE_TEST else SARD_OVERSAMPLE
    n_tr = write_image_list(rdir / "joint_train.txt", [(c2a_tr_i, 1), (sard_tr_i, ov)])
    n_va = write_image_list(rdir / "joint_val.txt",   [(c2a_va_i, 1), (sard_va_i, ov)])
    print(f"[data] joint TRAIN list: {n_tr} imgs (C2A {len(_list_images(c2a_tr_i))} + "
          f"SARD {len(_list_images(sard_tr_i))}x{ov}) | joint VAL: {n_va}")
    data_yaml = rdir / "joint_data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": "", "train": str(rdir / "joint_train.txt").replace("\\", "/"),
        "val": str(rdir / "joint_val.txt").replace("\\", "/"),
        "names": {0: "person"}, "nc": 1}, sort_keys=False))

    # ---- build / load model (resume-aware) ----
    ultra_last = rdir / "ultra" / "weights" / "last.pt"
    do_resume = resuming and ultra_last.is_file()
    if do_resume and not _ckpt_loadable(ultra_last):
        good = _newest_healthy_joint_ckpt(rdir)
        if good is not None:
            print(f"[resume] last.pt CORRUPT (interrupted write) -- restoring from {good.name}")
            try:
                shutil.copy2(ultra_last, ultra_last.parent / "last.corrupt.pt")
            except Exception:
                pass
            shutil.copy2(good, ultra_last)
        else:
            print("[resume] last.pt CORRUPT and no healthy fallback -- starting FRESH")
            do_resume = False

    if do_resume:
        print(f"[resume] resuming joint training FROM {ultra_last}")
        model = YOLO(str(ultra_last))
        model.add_callback("on_train_epoch_end", _abort_on_nan)   # NaN safety net
        try:
            model.train(resume=True)   # Ultralytics restores epoch/optimizer/args from the ckpt
        except Exception as e:
            print(f"[resume] Ultralytics resume failed ({e}) -- restarting fresh from C2A best")
            do_resume = False

    if not do_resume:
        if INIT_FROM_C2A_BEST:
            best = discover_c2a_cbam_p2_best()
            if best is None:
                print("[model] C2A CBAM+P2 best.pt NOT found -- fresh build from yolo11m.pt")
                yml = rdir / "cbam_p2head.yaml"; yml.write_text(CBAM_P2HEAD_YAML.format(nc=NC))
                model = YOLO(str(yml)); model.load(PRETRAINED_WEIGHTS)
            else:
                print(f"[model] init from C2A-trained CBAM+P2: {best}")
                model = YOLO(str(best))
        else:
            yml = rdir / "cbam_p2head.yaml"; yml.write_text(CBAM_P2HEAD_YAML.format(nc=NC))
            model = YOLO(str(yml)); model.load(PRETRAINED_WEIGHTS)
            print("[model] fresh CBAM+P2 from yolo11m.pt")

        kw = dict(data=str(data_yaml), epochs=(2 if SMOKE_TEST else NUM_EPOCHS),
                  imgsz=IMG_SIZE, batch=BATCH_SIZE, nbs=NOMINAL_BATCH, patience=PATIENCE,
                  optimizer=OPTIMIZER, lr0=LR0, lrf=LRF, cos_lr=COS_LR,
                  device=0, workers=NUM_WORKERS, cache=CACHE, amp=True, seed=SEED, deterministic=True,
                  project=str(rdir), name="ultra", exist_ok=True, plots=True, save=True, verbose=True,
                  save_period=25, resume=False)   # save_period=25 -> periodic ckpts for recovery
        if SMOKE_TEST:
            kw["fraction"] = SMOKE_FRACTION
        model.add_callback("on_train_epoch_end", _abort_on_nan)   # NaN safety net (abort on divergence)
        last_err = None
        for bs in OOM_RETRY_BATCHES:
            kw["batch"] = bs
            try:
                torch.cuda.empty_cache(); gc.collect()
                print(f"[train] batch={bs}, nbs={NOMINAL_BATCH}, epochs={kw['epochs']}")
                model.train(**kw)
                last_err = None; break
            except torch.cuda.OutOfMemoryError as e:
                last_err = e; print(f"[train] OOM at batch={bs} -- shrinking"); torch.cuda.empty_cache(); gc.collect()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    last_err = e; print(f"[train] OOM(RuntimeError) at batch={bs}"); torch.cuda.empty_cache(); gc.collect()
                else:
                    raise
        if last_err is not None:
            raise RuntimeError(f"OOM at every batch in {OOM_RETRY_BATCHES}: {last_err}")

    best_pt = rdir / "ultra" / "weights" / "best.pt"
    if best_pt.is_file():
        shutil.copy2(best_pt, rdir / "weights" / "best.pt")
    print(f"[train] done. deployable weights -> {rdir/'weights'/'best.pt'}")

    # ---- dual eval (the deployable claim: good on BOTH) ----
    model = YOLO(str(best_pt if best_pt.is_file() else (rdir / "ultra" / "weights" / "last.pt")))
    if torch.cuda.is_available():
        model.to("cuda:0")
    c2a_coco = None
    for cj in (c2a / c2a_te_s / f"{c2a_te_s}_annotations.json", c2a_te_l.parent / f"{c2a_te_s}_annotations.json"):
        if cj.is_file():
            c2a_coco = cj; break
    sard_coco = build_coco_gt_from_yolo(sard_te_i, sard_te_l, rdir / f"sard_{sard_te_s}_coco_gt.json")

    res_c2a = evaluate_on(model, "c2a", c2a_te_i, c2a_te_l, c2a, c2a_te_s, c2a_coco, rdir)
    res_sard = evaluate_on(model, "sard", sard_te_i, sard_te_l, sard, sard_te_s, sard_coco, rdir)

    # ---- efficiency (Sec 6, computed ONCE -- dataset-invariant) ----
    eff = architecture_report(model)
    eff.update(latency_profile(model))
    wp = rdir / "weights" / "best.pt"
    if wp.is_file():
        eff["weights_size_MB"] = round(wp.stat().st_size / 1024**2, 3)
    print(f"[eff] params={eff.get('params_total_M')}M gflops={eff.get('gflops')} "
          f"latency_mean={round(eff.get('latency_mean_ms', float('nan')),2)}ms fps@640={round(eff.get('fps_at_640', float('nan')),1)}")

    # ---- env.json (Sec 6 reproducibility) ----
    import platform
    def _ver(m):
        try:
            import importlib; return getattr(importlib.import_module(m), "__version__", None)
        except Exception:
            return None
    env = {"run_id": run_id, "model_tag": MODEL_TAG, "python": platform.python_version(),
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "ultralytics": _ver("ultralytics"), "numpy": _ver("numpy"),
           "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
           "gpu_physical_id": GPU_ID, "gpu_auto_select": GPU_AUTO_SELECT,
           "hyperparams": {"epochs": NUM_EPOCHS, "patience": PATIENCE, "imgsz": IMG_SIZE,
                           "batch": BATCH_SIZE, "nbs": NOMINAL_BATCH, "optimizer": OPTIMIZER,
                           "lr0": LR0, "lrf": LRF, "cos_lr": COS_LR, "sard_oversample": ov,
                           "init_from_c2a_best": INIT_FROM_C2A_BEST, "seed": SEED},
           "c2a_root": str(c2a), "sard_root": str(sard),
           "timestamp": datetime.now().isoformat(timespec="seconds")}
    (rdir / "env.json").write_text(json.dumps(env, indent=2, default=str))

    summary = {
        "run_id": run_id, "model_tag": MODEL_TAG, "smoke": SMOKE_TEST,
        "init_from_c2a_best": INIT_FROM_C2A_BEST, "sard_oversample": ov,
        "deployable_weights": str(rdir / "weights" / "best.pt"),
        "c2a_test": res_c2a, "sard_test": res_sard, "efficiency": eff,
        "reference_C2A_only_model": {
            "c2a_test_mAP50": 0.853, "sard_zeroshot_mAP50": 0.006,
            "note": "C2A-only CBAM+P2: strong on C2A, ~0 on SARD (the model this one replaces)."},
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    (rdir / "metrics" / "deployable_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    # manifest (Sec 6)
    (rdir / "manifest.json").write_text(json.dumps({
        "run_id": run_id, "model_tag": MODEL_TAG, "weights": str(rdir / "weights" / "best.pt"),
        "summary_pointer": "metrics/deployable_summary.json", "env_pointer": "env.json",
        "completed": datetime.now().isoformat(timespec="seconds")}, indent=2))
    print("\n" + "="*78)
    print("DEPLOYABLE MODEL RESULT (one model, both domains):")
    print(f"  C2A-test : mAP50={res_c2a.get('mAP50')}  (C2A-only model was ~0.853)")
    print(f"  SARD-test: mAP50={res_sard.get('mAP50')}  (C2A-only zero-shot was ~0.006)")
    print(f"  weights  : {rdir/'weights'/'best.pt'}")
    print("="*78)
    print("[done] If SARD-test is now well above zero AND C2A-test stays ~0.85, you have a")
    print("       single deployable model strong on both domains. If SARD-test is still low,")
    print("       raise SARD_OVERSAMPLE (3 -> 5 -> 8) and re-run.")

if __name__ == "__main__":
    main()

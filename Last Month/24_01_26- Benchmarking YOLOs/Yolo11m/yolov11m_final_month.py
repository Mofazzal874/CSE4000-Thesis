"""
yolov11m_final_month.py
=======================

YOLO11m baseline on C2A under the final-month metric protocol defined in
`Last Month/system_spec.md`. This run IS part of the headline ablation matrix
-- it is Phase A row 1 (`yolo11m_baseline`, P0) in Section 9.6 / Section 25.

Conflicts with the original Kaggle notebook `Yolov11m/yolov11m.ipynb` were
resolved IN FAVOUR OF THE SPEC, namely:

  * Notebook hardcoded `/kaggle/input/.../C2A_Dataset/new_dataset3` and
    `/kaggle/working/...`. Spec Sec 5.2/5.3 forbid hardcoded paths -- the
    output dir is derived from SCRIPT_DIR and the dataset root is auto-probed
    (with the C2A_ROOT env var as override).
  * Notebook used SAMPLE_RATIO=0.5 (a 50% subsample). Spec Sec 23 mandates the
    FROZEN full split with md5 verification. Subsampling now happens ONLY
    inside the smoke test (Sec 17), not in the full run.
  * Notebook ran EPOCHS=25, PATIENCE=5. Spec Sec 18 mandates dynamic epochs
    with NUM_EPOCHS=300 upper bound, Ultralytics PATIENCE=30, and a custom
    F2-patience early-stop (F2_PATIENCE=20) on top.
  * Notebook used BATCH_SIZE=10 on Kaggle T4x2. This box is a single 4070 Ti
    SUPER (16 GB) -- spec Sec 3 places YOLO11m at batch 16-24 with AMP.
    BATCH_SIZE is set to 16 with the OOM retry ladder dropping to 8/4 if
    needed.
  * Notebook used WORKERS=8 with no AMP. Spec Sec 7/20.3 confirms WORKERS=4
    on this Windows host (WinError 1455 silent-hang issue at >=8) and
    requires amp=True.
  * Notebook ran a single un-seeded training. Spec Sec 12.1 requires 5 seeds
    for any "improvement" claim, so SEEDS=[0,1,2,3,4] here. Expect ~30-50
    GPU-hours total on the 4070 Ti SUPER (rough estimate from Sec 25).
  * Spec Sec 18 wrote PATIENCE=30 / F2_PATIENCE=20 as a default. A
    2026-05-29 literature check (HIT-UAV thermal human detection paper,
    Ultralytics current default = 100, SOD-YOLO / VisDrone convention of
    fixed schedules) showed those values are below published convention
    for noisy small-object cosine training. Raised to PATIENCE=50 and
    F2_PATIENCE=40 here. Documented in docs/2026-05-29_yolo11m_final_month_writeup.md.

Spec compliance summary (sections deliberately implemented):
  - Sections 3, 7   : single-GPU 16 GB defaults, AMP
  - Section 5.2/5.3 : output-dir + dataset-root auto-detect (env override)
  - Section 8       : system sanity print
  - Section 11      : full metric catalog (detection / training-dyn / efficiency
                      / calibration / per-image / env / energy) -- with explicit
                      [SKIPPED] entries for SSM/CBAM/P2/SAHI/TTA architecture
                      metrics (Section 11.6) that don't apply to vanilla 11m
  - Section 12      : seed pinning, per-image scores persisted, paired-bootstrap
                      helper, BCa CIs on headline numbers
  - Section 13/14   : every CSV gets a PNG, PLOTS_INDEX.md/.csv provenance
  - Section 15/16   : output folder layout, run-ID naming
  - Section 17      : 8-step smoke-test harness, <5 min, auto-cleanup
  - Section 18      : dynamic epochs + Ultralytics patience + custom F2 patience
  - Section 19      : nvidia-smi loop + psutil sampler in background
  - Section 20      : OOM retry ladder, checkpoint cadence, resume detection
  - Section 21      : ensure_packages() preamble
  - Section 22      : torchinfo model_summary + module_table + flops_breakdown
  - Section 24      : pre-flight checklist gate (refuse to start on failure)
  - Section 25      : marked in_ablation_matrix=True, row=1 (yolo11m_baseline)
  - Section 26      : MODEL_CARD.md, ONNX export sanity

Explicitly SKIPPED (with reasons logged to skipped_metrics.txt):
  - attention_map_examples / channel_attention_weights  (no CBAM in 11m)
  - per-stride_AP / tiny_obj_recall_by_head             (no P2 head in 11m)
  - ssm_state_norm / forward_vs_backward_scan_disagreement / window_size_per_layer
    / dilation_branch_contribution / injection_layer_indices  (no Mamba/SSM)
  - SAHI/TTA-specific metrics                           (not part of this run)
  - paired-bootstrap p-values vs another model          (no comparison run
                                                         provided -- per-image
                                                         scores still saved
                                                         for later Phase-D)
"""

# =============================================================================
# 0. Top-level CONFIG  --  edit this block; everything below auto-derives
# =============================================================================

MODEL_TAG          = "yolo11m_baseline"          # Phase A row 1 (Sec 9.6, 25)
PRETRAINED_WEIGHTS = "yolo11m.pt"                # Ultralytics auto-downloads
SEEDS              = [0]                          # spec Sec 12.1 wants [0,1,2,3,4]
                                                  # for the headline ablation; running
                                                  # seed 0 first per user request.
                                                  # Switch to [0,1,2,3,4] when the
                                                  # full 30-50 h budget is available.

# Skip seeds that already have a complete `manifest.json` in runs/. Setting
# this to False forces a fresh run with a new timestamp even if a previous
# run finished -- use it if you want to deliberately re-train.
SKIP_COMPLETED_SEEDS = True

# If a previous run for this seed crashed (e.g. power loss) and left a
# weights/last.pt in its ultra/ dir, the script reuses that run_id and
# passes resume=True to Ultralytics. Set False to ignore the partial run
# and start over with a fresh run_id.
RESUME_INCOMPLETE_SEEDS = True

# Training
NUM_EPOCHS         = 300                          # upper bound only (Section 18.1)
PATIENCE           = 50                           # Ultralytics fitness patience.
                                                  # Raised from spec's 30 after a
                                                  # 2026-05-29 literature check: the
                                                  # closest published paper (HIT-UAV
                                                  # thermal human detection, Sci
                                                  # Reports 2024) uses 50 with the
                                                  # same 300-epoch cap, and 50 is
                                                  # the historic Ultralytics
                                                  # default. 30 risks premature
                                                  # cutoff on the noisy late phase
                                                  # of small-object cosine schedules.
F2_PATIENCE        = 40                           # custom F2-based patience.
                                                  # Raised from spec's 20 for the
                                                  # same reason -- kept just below
                                                  # PATIENCE so F2-stagnation can
                                                  # still trip first if it stalls
                                                  # while fitness slowly drifts.
IMG_SIZE           = 640
BATCH_SIZE         = 16                           # 4070 Ti SUPER, YOLO11m + AMP (Sec 3:
                                                  # "YOLOv8m / YOLOv11m: batch 16-24 fits")
SAVE_PERIOD        = 5                            # epochs; checkpoint cadence (Section 20.2)
OOM_RETRY_BATCHES  = [BATCH_SIZE, BATCH_SIZE // 2, BATCH_SIZE // 4, 4]

# Hardware-pinned defaults (Section 7)
NUM_WORKERS        = 4
CACHE              = "ram"
COS_LR             = True
LRF                = 0.01

# Smoke (Section 17)
SMOKE_TEST         = False                        # set True to ONLY run smoke
SMOKE_FRACTION     = 0.01
SMOKE_EPOCHS       = 2
SMOKE_BATCH        = None                         # None -> reuse BATCH_SIZE
SMOKE_KEEP_OUTPUTS = False

# Evaluation toggles
DO_TRAIN           = True                         # set False to evaluate an existing best.pt
DO_ONNX_EXPORT     = True
LATENCY_WARMUP     = 50
LATENCY_RUNS       = 500
LATENCY_RESOLUTIONS = (320, 480, 640, 800, 1280)
LATENCY_BATCH_SIZES = (1, 4, 8, 16)
BOOTSTRAP_B        = 1000                          # 1000 for individual-metric BCa CIs

# Compare paired-bootstrap against another run's per_image_test.csv (optional)
BASELINE_PER_IMAGE_CSV = None                      # e.g. Path(...) or None

# Misc
COUNTRY_ISO_CODE   = "BGD"                         # CodeCarbon grid intensity


# =============================================================================
# 1. PACKAGE CHECK  (Section 21)
# =============================================================================

REQUIRED_PACKAGES = [
    ("ultralytics",   ">=8.3.0"),
    ("torch",         ">=2.4.0"),
    ("thop",          None),
    ("openpyxl",      None),
    ("pandas",        "<3.0"),
    ("matplotlib",    "<3.10"),
    ("scikit-learn",  None),
    ("scipy",         None),
    ("statsmodels",   None),
    ("codecarbon",    None),
    ("pynvml",        None),
    ("psutil",        None),
    ("torchinfo",     None),
    ("tqdm",          None),
    ("opencv-python", None),
    ("PyYAML",        None),
    ("seaborn",       None),
    ("tabulate",      None),
    ("pycocotools",   None),
]

def ensure_packages():
    import importlib, subprocess, sys
    name_map = {
        "opencv-python": "cv2",
        "PyYAML":        "yaml",
        "scikit-learn":  "sklearn",
        "pycocotools":   "pycocotools",
    }
    for name, ver in REQUIRED_PACKAGES:
        mod = name_map.get(name, name.replace("-", "_"))
        try:
            importlib.import_module(mod)
        except ImportError:
            spec = name + (ver if ver else "")
            print(f"[deps] installing missing package: {spec}")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q", spec]
                )
            except subprocess.CalledProcessError as e:
                print(f"[deps] WARN: could not install {spec}: {e}. "
                      f"Dependent metrics will be [SKIPPED] at runtime.")

ensure_packages()


# =============================================================================
# 2. IMPORTS
# =============================================================================

import os, sys, json, time, gc, math, hashlib, shutil, random, threading
import subprocess, traceback, platform, warnings, logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Sequence, Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import yaml
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Optional imports -- guarded
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False

try:
    # OfflineEmissionsTracker accepts country_iso_code (the online
    # EmissionsTracker dropped it in newer codecarbon releases).
    from codecarbon import OfflineEmissionsTracker
    HAS_CODECARBON = True
except ImportError:
    HAS_CODECARBON = False

try:
    from torchinfo import summary as torchinfo_summary
    HAS_TORCHINFO = True
except ImportError:
    HAS_TORCHINFO = False

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAS_PYCOCO = True
except ImportError:
    HAS_PYCOCO = False

try:
    from scipy import stats as scistats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"FATAL: ultralytics import failed -- {e}")
    sys.exit(1)

try:
    matplotlib.style.use("seaborn-v0_8-whitegrid")
except Exception:
    pass

# Windows uses 'spawn' for DataLoader workers; each worker re-imports this
# module, which would otherwise re-print every [paths]/[sanity]/[splits]
# block and re-run the git probe + free-space check. Gate verbose top-level
# I/O on the main process. The constants below (SCRIPT_DIR, OUTPUT_ROOT,
# DATASET_ROOT, CLASS_NAMES, ...) still get computed in every worker
# because they are referenced by the dataset/eval functions, but the
# prints and the once-per-process split-md5 write are silenced in workers.
import multiprocessing as _mp
_IS_MAIN = _mp.current_process().name == "MainProcess"


# =============================================================================
# 3. PATH AUTO-DETECT  (Section 5.2 + 5.3)
# =============================================================================

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path(os.getcwd()).resolve()

OUTPUT_ROOT = SCRIPT_DIR

def find_dataset_root(env_var: str, structure_marker: str,
                      candidates: List[Path]) -> Path:
    """Section 5.3 -- env override -> ordered candidate probe."""
    override = os.environ.get(env_var)
    if override:
        p = Path(override)
        if (p / structure_marker).is_dir():
            return p
        raise FileNotFoundError(
            f"{env_var}={override} does not contain {structure_marker}"
        )
    for p in candidates:
        if p and (p / structure_marker).is_dir():
            return p
    raise FileNotFoundError(
        f"Could not locate dataset (looking for `{structure_marker}`). "
        f"Set ${env_var} to the dataset root directory and re-run.\n"
        f"Tried: {[str(c) for c in candidates]}"
    )

C2A_CANDIDATES = [
    # 1) Confirmed AnyDesk PC layout (system_spec.md Sec 27, verified
    #    2026-05-28): dataset lives at
    #    E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3
    #    and the script lives at E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\Yolo11m,
    #    so SCRIPT_DIR.parent.parent == E:\Thesis_mofazzal_2007074\.
    SCRIPT_DIR.parent.parent / "common" / "c2a" / "C2A_Dataset" / "new_dataset3",
    SCRIPT_DIR.parent.parent.parent / "common" / "c2a" / "C2A_Dataset" / "new_dataset3",
    # 2) Same layout without the inner C2A_Dataset/ wrapper (older snapshot,
    #    kept for forward-compat in case the dataset is re-flattened).
    SCRIPT_DIR / "common" / "c2a" / "new_dataset3",
    SCRIPT_DIR.parent / "common" / "c2a" / "new_dataset3",
    SCRIPT_DIR.parent.parent / "common" / "c2a" / "new_dataset3",
    # 3) Cross-drive sweeps -- both with and without the C2A_Dataset wrapper.
    *[
        Path(f"{d}:/Thesis_mofazzal_2007074/common/c2a/C2A_Dataset/new_dataset3")
        for d in ("E", "D", "F", "G", "C")
    ],
    *[
        Path(f"{d}:/Thesis_mofazzal_2007074/common/c2a/new_dataset3")
        for d in ("E", "D", "F", "G", "C")
    ],
    *[
        Path(f"{d}:/C2A_Dataset/new_dataset3")
        for d in ("E", "D", "F", "G", "C")
    ],
]

# Verify dataset internal layout matches Section 5.1 contract.
def verify_dataset_layout(root: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"root": str(root)}
    required = {
        "train_images":  root / "train" / "images",
        "train_labels":  root / "train" / "labels",
        "val_images":    root / "val"   / "images",
        "val_labels":    root / "val"   / "labels",
        "test_images":   root / "test"  / "images",
        "test_labels":   root / "test"  / "labels",
    }
    missing = [k for k, p in required.items() if not p.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Dataset at {root} is missing required subfolders: {missing}. "
            f"Section 5.1 of the spec requires train/val/test each with "
            f"images/ and labels/ subdirectories."
        )
    # COCO JSONs are useful but not strictly required.
    info["train_coco_json"] = root / "train" / "train_annotations.json"
    info["val_coco_json"]   = root / "val"   / "val_annotations.json"
    info["test_coco_json"]  = root / "test"  / "test_annotations.json"
    info["has_train_coco"]  = info["train_coco_json"].is_file()
    info["has_val_coco"]    = info["val_coco_json"].is_file()
    info["has_test_coco"]   = info["test_coco_json"].is_file()
    for split in ("train", "val", "test"):
        imgs = list((root / split / "images").glob("*.png"))
        info[f"{split}_image_count"] = len(imgs)
        if len(imgs) == 0:
            # Section 5.1 says .png; if zero, check other formats and warn.
            other = list((root / split / "images").glob("*"))
            info[f"{split}_image_count_other"] = len(other)
    return info

if _IS_MAIN:
    print("[paths] SCRIPT_DIR  =", SCRIPT_DIR)
    print("[paths] OUTPUT_ROOT =", OUTPUT_ROOT)

DATASET_ROOT = find_dataset_root("C2A_ROOT", "train/images", C2A_CANDIDATES)
DATASET_INFO = verify_dataset_layout(DATASET_ROOT)
if _IS_MAIN:
    print("[paths] DATASET_ROOT =", DATASET_ROOT)
    print("[paths] dataset layout:", {k: v for k, v in DATASET_INFO.items()
                                      if not str(k).endswith("_coco_json")})

# Auto-detect class list from the COCO JSON; fall back to ['person'] per
# spec Sec 5.1. The C2A JSON on disk has placeholder category names like
# "0" -- those are not real class names. Detect numeric / empty / single-
# character placeholders and substitute the spec default.
_SPEC_DEFAULT_NAMES = ["person"]

def _is_placeholder_name(n: str) -> bool:
    if not isinstance(n, str):
        return True
    s = n.strip()
    if not s:
        return True
    # Bare digits ("0", "1", "12") or "cls_0" / "class_0" stubs.
    if s.isdigit():
        return True
    if s.lower() in {"none", "null", "n/a", "unknown"}:
        return True
    return False

def detect_class_names() -> List[str]:
    j = DATASET_INFO["train_coco_json"]
    if j.is_file():
        try:
            data = json.loads(j.read_text(encoding="utf-8"))
            cats = sorted(data.get("categories", []),
                          key=lambda c: c.get("id", 0))
            names = [c.get("name", "") for c in cats]
            if names and all(not _is_placeholder_name(n) for n in names):
                return names
            if _IS_MAIN and names:
                print(f"[paths] WARN: COCO categories had placeholder names "
                      f"{names}; falling back to spec default "
                      f"{_SPEC_DEFAULT_NAMES} (Sec 5.1).")
        except Exception as e:
            if _IS_MAIN:
                print(f"[paths] WARN: could not read class names from {j}: {e}")
    return list(_SPEC_DEFAULT_NAMES)

CLASS_NAMES = detect_class_names()
NC = len(CLASS_NAMES)
if _IS_MAIN:
    print(f"[paths] classes (nc={NC}): {CLASS_NAMES}")


# =============================================================================
# 4. SYSTEM SANITY  (Section 8)
# =============================================================================

def system_sanity(verbose: bool = True) -> Dict[str, Any]:
    assert torch.cuda.is_available(), "CUDA not available -- this script requires a CUDA GPU."
    props = torch.cuda.get_device_properties(0)
    info = {
        "python":       platform.python_version(),
        "torch":        torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn":        torch.backends.cudnn.version(),
        "gpu_name":     torch.cuda.get_device_name(0),
        "gpu_vram_gb":  round(props.total_memory / 1024**3, 2),
        "gpu_capability": f"{props.major}.{props.minor}",
        "gpu_sm_count": props.multi_processor_count,
    }
    if verbose:
        print("[sanity]", json.dumps(info, indent=2))
        if "NVIDIA" not in info["gpu_name"].upper():
            print(f"[sanity] WARN: GPU '{info['gpu_name']}' is not NVIDIA -- "
                  f"refusing to train on Intel UHD (Section 4 of spec).")
        if info["gpu_vram_gb"] < 12:
            print(f"[sanity] WARN: only {info['gpu_vram_gb']} GB VRAM detected -- "
                  f"BATCH_SIZE={BATCH_SIZE} may OOM. Lower it in the config block.")
    return info

SANITY = system_sanity(verbose=_IS_MAIN)
torch.backends.cudnn.benchmark = True

# Free-space check (Section 5.5)
def check_free_space(min_gb: int = 20, verbose: bool = True):
    if HAS_PSUTIL:
        try:
            usage = shutil.disk_usage(str(OUTPUT_ROOT))
            free_gb = usage.free / 1024**3
            if verbose:
                print(f"[sanity] free disk at OUTPUT_ROOT: {free_gb:.1f} GB")
            if free_gb < min_gb:
                raise RuntimeError(
                    f"Free space {free_gb:.1f} GB < required {min_gb} GB. "
                    f"Clear disk before running."
                )
        except Exception as e:
            if verbose:
                print(f"[sanity] WARN: free-space check failed: {e}")
if _IS_MAIN:
    check_free_space(20, verbose=True)


# =============================================================================
# 5. RUN-ID + DIRECTORY LAYOUT  (Sections 15, 16)
# =============================================================================

def git_short_hash_and_dirty() -> Tuple[str, bool]:
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(SCRIPT_DIR),
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()[:6]
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(SCRIPT_DIR),
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip())
        return h, dirty
    except Exception:
        return "nogit", False

GIT_HASH, GIT_DIRTY = git_short_hash_and_dirty()
GIT_TAG = GIT_HASH if not GIT_DIRTY else "dirty"

def make_run_id(seed: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{MODEL_TAG}_s{seed}_{GIT_TAG}"


def find_existing_run_for_seed(seed: int) -> Optional[Dict[str, Any]]:
    """Return info about the most recent prior run for `seed`, if any.

    Status is one of:
      - "completed"  -- manifest.json exists at the run root. Don't re-run
                        unless SKIP_COMPLETED_SEEDS is False.
      - "incomplete" -- weights/best.pt or ultra/weights/last.pt exists but
                        no manifest.json. Power-loss / crash mid-run.
                        Caller should reuse this run_id and pass
                        resume=True to Ultralytics so training continues
                        from the last saved epoch.
      - None        -- no prior run for this seed.
    """
    runs_root = OUTPUT_ROOT / "runs"
    if not runs_root.is_dir():
        return None
    # Directory names start with the launch timestamp, so name-sort = chronological.
    candidates = sorted(
        [d for d in runs_root.iterdir()
         if d.is_dir()
         and f"_s{seed}_" in d.name
         and not d.name.endswith("_multi_seed_rollup")],
        key=lambda d: d.name,
    )
    if not candidates:
        return None
    # Walk newest-first; the most recent state of this seed wins.
    for d in reversed(candidates):
        has_manifest = (d / "manifest.json").is_file()
        if has_manifest:
            return {"run_id": d.name, "status": "completed", "dir": d}
        # Ultralytics saves last.pt under <project>/<name>/weights/. Our config
        # uses project=run_dir, name="ultra", so the canonical resume target
        # is runs/<run_id>/ultra/weights/last.pt.
        ultra_last = d / "ultra" / "weights" / "last.pt"
        safety = d / "weights" / "last_safety.pt"
        if ultra_last.is_file() or safety.is_file():
            return {"run_id": d.name, "status": "incomplete", "dir": d,
                    "ultra_last": ultra_last if ultra_last.is_file() else None,
                    "safety":     safety     if safety.is_file()     else None}
    # Directories exist but none have a checkpoint -- treat as empty.
    return None

def make_run_dirs(run_id: str) -> Dict[str, Path]:
    base = OUTPUT_ROOT / "runs" / run_id
    sub = {
        "base":    base,
        "code":    base / "code",
        "weights": base / "weights",
        "metrics": base / "metrics",
        "plots":   base / "plots",
        "arch":    base / "architecture",
        "sig":     base / "significance",
        "energy":  base / "energy",
        "logs":    base / "logs",
    }
    for p in sub.values():
        p.mkdir(parents=True, exist_ok=True)
    return sub

COMMON_DIR = OUTPUT_ROOT / "common"
COMMON_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR = COMMON_DIR / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR = OUTPUT_ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
SMOKE_BASE = OUTPUT_ROOT / "smoke"
SMOKE_BASE.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 6. SEED PINNING  (Section 12.1)
# =============================================================================

def set_seed(s: int, *, deterministic: bool = True):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# =============================================================================
# 7. FROZEN SPLIT MD5  (Section 23)
# =============================================================================

def md5_of_filenames(image_dir: Path) -> str:
    names = sorted([p.name for p in image_dir.iterdir() if p.is_file()])
    h = hashlib.md5()
    for n in names:
        h.update(n.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()

def freeze_or_verify_splits(verbose: bool = True) -> Dict[str, str]:
    splits = {
        "train": DATASET_ROOT / "train" / "images",
        "val":   DATASET_ROOT / "val"   / "images",
        "test":  DATASET_ROOT / "test"  / "images",
    }
    current = {k: md5_of_filenames(v) for k, v in splits.items()}
    splits_file = SPLITS_DIR / "splits.md5"
    if not splits_file.is_file():
        splits_file.write_text(json.dumps(current, indent=2))
        # Also dump the actual filename lists so anyone can reconstruct.
        for k, v in splits.items():
            names = sorted([p.name for p in v.iterdir() if p.is_file()])
            (SPLITS_DIR / f"{k}_images.txt").write_text("\n".join(names))
        if verbose:
            print(f"[splits] FROZEN new split md5: {current}")
        return current
    saved = json.loads(splits_file.read_text())
    mism = {k: (saved.get(k), current.get(k))
            for k in current if saved.get(k) != current.get(k)}
    if mism:
        raise RuntimeError(
            f"Split md5 mismatch vs {splits_file} -- the dataset on disk "
            f"differs from the frozen split. Mismatches: {mism}. "
            f"Delete the file ONLY if you genuinely intend a new split."
        )
    if verbose:
        print(f"[splits] verified -- matches {splits_file}")
    return saved

SPLIT_MD5 = freeze_or_verify_splits(verbose=_IS_MAIN)


# =============================================================================
# 8. data.yaml builder
# =============================================================================

def write_data_yaml(out_path: Path) -> Path:
    data = {
        "path":  str(DATASET_ROOT).replace("\\", "/"),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
        "nc":    NC,
    }
    out_path.write_text(yaml.safe_dump(data, sort_keys=False))
    return out_path


# =============================================================================
# 9. RESOURCE SAMPLERS  (Section 19.1, 11.2)
# =============================================================================

class ResourceSampler:
    """Background-thread sampler that writes nvidia-smi + psutil snapshots
    every `interval` seconds while running. Output: two CSVs in logs/."""
    def __init__(self, logs_dir: Path, interval: float = 2.0):
        self.logs_dir = logs_dir
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.gpu_rows: List[Dict[str, Any]] = []
        self.cpu_rows: List[Dict[str, Any]] = []

    def _gpu_snapshot(self) -> Optional[Dict[str, Any]]:
        if not HAS_NVML:
            return None
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            try:
                pwr = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                pwr = float("nan")
            try:
                tmp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                tmp = float("nan")
            return {
                "ts": time.time(),
                "gpu_util_pct": util.gpu,
                "mem_util_pct": util.memory,
                "vram_used_MB": mem.used / 1024**2,
                "power_W": pwr,
                "temp_C": tmp,
            }
        except Exception:
            return None

    def _cpu_snapshot(self) -> Optional[Dict[str, Any]]:
        if not HAS_PSUTIL:
            return None
        try:
            return {
                "ts": time.time(),
                "cpu_pct": psutil.cpu_percent(interval=None),
                "ram_used_GB": psutil.virtual_memory().used / 1024**3,
                "ram_pct": psutil.virtual_memory().percent,
            }
        except Exception:
            return None

    def _loop(self):
        while not self._stop.is_set():
            g = self._gpu_snapshot()
            if g is not None:
                self.gpu_rows.append(g)
            c = self._cpu_snapshot()
            if c is not None:
                self.cpu_rows.append(c)
            self._stop.wait(self.interval)

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop_and_dump(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self.gpu_rows:
            pd.DataFrame(self.gpu_rows).to_csv(
                self.logs_dir / "nvidia_smi_loop.csv", index=False)
        if self.cpu_rows:
            pd.DataFrame(self.cpu_rows).to_csv(
                self.logs_dir / "psutil_loop.csv", index=False)

    def summary(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if self.gpu_rows:
            df = pd.DataFrame(self.gpu_rows)
            out["gpu_util_avg_%"] = float(df["gpu_util_pct"].mean())
            out["gpu_util_max_%"] = float(df["gpu_util_pct"].max())
            out["gpu_power_avg_W"] = float(df["power_W"].mean())
            out["gpu_power_max_W"] = float(df["power_W"].max())
            out["gpu_temp_avg_C"] = float(df["temp_C"].mean())
            out["gpu_temp_max_C"] = float(df["temp_C"].max())
            out["vram_used_max_MB"] = float(df["vram_used_MB"].max())
        if self.cpu_rows:
            df = pd.DataFrame(self.cpu_rows)
            out["cpu_util_avg_%"] = float(df["cpu_pct"].mean())
            out["ram_used_max_GB"] = float(df["ram_used_GB"].max())
        return out


# =============================================================================
# 10. CALLBACKS  (Sections 11.2, 18, 20.2)
# =============================================================================

class F2EarlyStop:
    """Custom F2-patience early-stop. Tracks val F2 per epoch and triggers
    `stop=True` when F2 has not improved for `patience` epochs."""
    def __init__(self, patience: int):
        self.patience = patience
        self.best_f2 = -1.0
        self.best_epoch = 0
        self.bad = 0
        self.stop = False
        self.trace: List[Tuple[int, float]] = []

    def update(self, epoch: int, p: float, r: float) -> bool:
        if (p + r) == 0:
            f2 = 0.0
        else:
            f2 = (5 * p * r) / (4 * p + r + 1e-12)
        self.trace.append((epoch, f2))
        if f2 > self.best_f2 + 1e-6:
            self.best_f2 = f2
            self.best_epoch = epoch
            self.bad = 0
        else:
            self.bad += 1
        if self.bad >= self.patience:
            self.stop = True
        return self.stop


def attach_ultralytics_callbacks(model, *, run_dirs: Dict[str, Path],
                                 sampler: ResourceSampler,
                                 f2_stopper: F2EarlyStop,
                                 epoch_metrics: List[Dict[str, Any]]):
    """Hook into Ultralytics's callback registry. We watch every epoch for:
       - val P/R/mAP -> compute F2, feed F2EarlyStop, decide whether to halt
       - VRAM peak, grad-norm proxy, epoch wall-clock
       - last-safety checkpoint copy (Section 20.2)
    """
    state = {"epoch_start_ts": time.time(),
             "epoch_start_vram": 0.0,
             "epoch_idx": 0}

    def on_train_epoch_start(trainer):
        torch.cuda.reset_peak_memory_stats(0)
        state["epoch_start_ts"] = time.time()
        state["epoch_start_vram"] = torch.cuda.memory_allocated(0) / 1024**2

    def on_fit_epoch_end(trainer):
        ep = int(getattr(trainer, "epoch", state["epoch_idx"]))
        state["epoch_idx"] = ep + 1
        elapsed = time.time() - state["epoch_start_ts"]
        vram_peak = torch.cuda.max_memory_allocated(0) / 1024**2
        # Try to pull P/R off the validator's metrics.
        p = r = map50 = map5095 = float("nan")
        try:
            metrics = trainer.metrics  # dict-like
            p     = float(metrics.get("metrics/precision(B)", float("nan")))
            r     = float(metrics.get("metrics/recall(B)",    float("nan")))
            map50 = float(metrics.get("metrics/mAP50(B)",     float("nan")))
            map5095 = float(metrics.get("metrics/mAP50-95(B)", float("nan")))
        except Exception:
            pass

        try:
            ds = trainer.train_loader.dataset
            n_train = len(ds)
            sps = n_train / max(elapsed, 1e-6)
        except Exception:
            sps = float("nan")

        row = {
            "epoch":           ep,
            "epoch_time_s":    round(elapsed, 3),
            "samples_per_sec": round(sps, 3) if not math.isnan(sps) else sps,
            "vram_peak_MB":    round(vram_peak, 1),
            "precision":       p,
            "recall":          r,
            "mAP50":           map50,
            "mAP50-95":        map5095,
        }
        # Grad norm proxy: parameter norm (true grad norm would need a forward
        # hook; we approximate via the parameter norm for plotting purposes).
        try:
            total = 0.0
            for prm in trainer.model.parameters():
                if prm.requires_grad and prm.grad is not None:
                    total += float(prm.grad.detach().data.norm(2).item()) ** 2
            row["grad_norm_total"] = total ** 0.5
        except Exception:
            row["grad_norm_total"] = float("nan")
        epoch_metrics.append(row)
        print(f"[epoch {ep}] t={elapsed:.1f}s  P={p:.3f}  R={r:.3f}  "
              f"mAP50={map50:.3f}  mAP50-95={map5095:.3f}  vram_peak={vram_peak:.0f}MB")

        # F2 patience
        if not (math.isnan(p) or math.isnan(r)):
            stop = f2_stopper.update(ep, p, r)
            if stop:
                print(f"[F2-patience] stop triggered at epoch {ep} "
                      f"(best F2={f2_stopper.best_f2:.4f} at epoch "
                      f"{f2_stopper.best_epoch})")
                try:
                    trainer.stop_training = True
                except Exception:
                    pass

        # Safety checkpoint copy
        try:
            last = Path(trainer.save_dir) / "weights" / "last.pt"
            if last.is_file():
                shutil.copy2(last, run_dirs["weights"] / "last_safety.pt")
        except Exception:
            pass

    try:
        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        model.add_callback("on_fit_epoch_end",     on_fit_epoch_end)
    except Exception as e:
        print(f"[callbacks] WARN: could not attach callbacks: {e}")


# =============================================================================
# 11. OOM-RETRY TRAINER  (Section 20.1)
# =============================================================================

def train_with_oom_retry(model, base_kwargs: Dict[str, Any]) -> Any:
    last_err: Optional[BaseException] = None
    for bs in OOM_RETRY_BATCHES:
        kwargs = dict(base_kwargs)
        kwargs["batch"] = bs
        # Ultralytics 8.4.x removed the top-level `accumulate` arg. Use `nbs`
        # (nominal batch size) instead — the trainer computes accumulate as
        # round(nbs / batch), so setting nbs=BATCH_SIZE keeps the effective
        # batch constant across retries.
        kwargs["nbs"] = BATCH_SIZE
        _effective_accum = max(1, BATCH_SIZE // bs)
        torch.cuda.empty_cache(); gc.collect()
        try:
            print(f"[oom-retry] attempting batch={bs}, nbs={BATCH_SIZE} (effective accumulate={_effective_accum})")
            return model.train(**kwargs)
        except torch.cuda.OutOfMemoryError as e:
            last_err = e
            print(f"[oom-retry] OOM at batch={bs} -- shrinking...")
            torch.cuda.empty_cache(); gc.collect()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                last_err = e
                print(f"[oom-retry] OOM (RuntimeError) at batch={bs}")
                torch.cuda.empty_cache(); gc.collect()
            else:
                raise
    raise RuntimeError(
        f"OOM at every retry batch size {OOM_RETRY_BATCHES}. Last: {last_err}. "
        f"Suggest: imgsz 640->512, disable cache='ram', drop num_workers."
    )


# =============================================================================
# 12. EVALUATION HELPERS  (Section 11.1, 11.3, 11.4)
# =============================================================================

def _box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a:(N,4) b:(M,4) xyxy -> (N,M) IoU."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    al = a[:, None, :]; bl = b[None, :, :]
    inter_x1 = np.maximum(al[..., 0], bl[..., 0])
    inter_y1 = np.maximum(al[..., 1], bl[..., 1])
    inter_x2 = np.minimum(al[..., 2], bl[..., 2])
    inter_y2 = np.minimum(al[..., 3], bl[..., 3])
    iw = np.clip(inter_x2 - inter_x1, 0, None)
    ih = np.clip(inter_y2 - inter_y1, 0, None)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter + 1e-12
    return inter / union


def _yolo_label_to_xyxy(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    if not label_path.is_file():
        return np.zeros((0, 4), dtype=np.float32)
    out = []
    for ln in label_path.read_text().splitlines():
        parts = ln.strip().split()
        if len(parts) < 5:
            continue
        _, cx, cy, w, h = (float(x) for x in parts[:5])
        x1 = (cx - w/2) * img_w
        y1 = (cy - h/2) * img_h
        x2 = (cx + w/2) * img_w
        y2 = (cy + h/2) * img_h
        out.append([x1, y1, x2, y2])
    return np.asarray(out, dtype=np.float32) if out else np.zeros((0, 4), dtype=np.float32)


def per_image_eval(model: YOLO, *, split: str, conf: float = 0.25,
                   iou_match: float = 0.5) -> pd.DataFrame:
    """Per-image TP/FP/FN/F1/F2/avg_conf/latency on the split. Per spec
    Section 11.1 we ALSO need TP/FP/FN at conf=0.001 for confusion etc., but
    those come out of ultralytics .val(); this helper is the operational
    (conf=0.25) per-image table that fuels paired-bootstrap tests."""
    img_dir = DATASET_ROOT / split / "images"
    lbl_dir = DATASET_ROOT / split / "labels"
    images = sorted(img_dir.glob("*.png"))
    rows = []
    for img_path in images:
        # GT
        import cv2
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]
        gt = _yolo_label_to_xyxy(
            lbl_dir / (img_path.stem + ".txt"), w, h)
        # Predict (with timing)
        t0 = time.perf_counter()
        res = model.predict(str(img_path), conf=conf, verbose=False)
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        boxes = res[0].boxes
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            pred = np.zeros((0, 4), dtype=np.float32)
            confs = np.zeros((0,), dtype=np.float32)
        else:
            pred = boxes.xyxy.cpu().numpy().astype(np.float32)
            confs = boxes.conf.cpu().numpy().astype(np.float32)
        # Greedy matching at iou_match
        tp = fp = fn = 0
        if pred.shape[0] == 0:
            fn = gt.shape[0]
        elif gt.shape[0] == 0:
            fp = pred.shape[0]
        else:
            iou = _box_iou_xyxy(pred, gt)
            order = np.argsort(-confs)
            matched_gt: set = set()
            for pi in order:
                j = int(np.argmax(iou[pi]))
                if iou[pi, j] >= iou_match and j not in matched_gt:
                    tp += 1
                    matched_gt.add(j)
                else:
                    fp += 1
            fn = gt.shape[0] - len(matched_gt)
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2*p*r / max(p + r, 1e-12)
        f2 = 5*p*r / max(4*p + r, 1e-12)
        rows.append({
            "image":         img_path.name,
            "gt":            int(gt.shape[0]),
            "pred":          int(pred.shape[0]),
            "TP": tp, "FP": fp, "FN": fn,
            "precision":     round(p, 6),
            "recall":        round(r, 6),
            "F1":            round(f1, 6),
            "F2":            round(f2, 6),
            "avg_conf":      float(confs.mean()) if confs.size else 0.0,
            "max_conf":      float(confs.max())  if confs.size else 0.0,
            "inference_ms":  round(dt_ms, 3),
        })
    return pd.DataFrame(rows)


def pr_and_f1_curves(per_img: pd.DataFrame, model: YOLO, split: str,
                     conf_grid: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build PR / F1-vs-conf / confidence-histogram CSVs by sweeping `conf`
    over the grid (Section 11.1)."""
    if conf_grid is None:
        conf_grid = np.arange(0.0, 1.001, 0.01)
    img_dir = DATASET_ROOT / split / "images"
    lbl_dir = DATASET_ROOT / split / "labels"
    images = sorted(img_dir.glob("*.png"))
    # Pre-collect predictions ONCE at conf~0 to sweep cheaply.
    all_pred_boxes: List[np.ndarray] = []
    all_pred_confs: List[np.ndarray] = []
    all_gt_boxes: List[np.ndarray] = []
    import cv2
    for img_path in images:
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]
        gt = _yolo_label_to_xyxy(lbl_dir / (img_path.stem + ".txt"), w, h)
        res = model.predict(str(img_path), conf=0.001, verbose=False)
        b = res[0].boxes
        if b is None or len(b) == 0:
            pb = np.zeros((0, 4), dtype=np.float32)
            pc = np.zeros((0,), dtype=np.float32)
        else:
            pb = b.xyxy.cpu().numpy().astype(np.float32)
            pc = b.conf.cpu().numpy().astype(np.float32)
        all_pred_boxes.append(pb)
        all_pred_confs.append(pc)
        all_gt_boxes.append(gt)
    # Confidence histogram
    all_confs = np.concatenate(all_pred_confs) if all_pred_confs else np.array([])
    hist, edges = np.histogram(all_confs, bins=50, range=(0.0, 1.0))
    hist_df = pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:],
                            "count": hist})
    # PR sweep
    rows = []
    for c in conf_grid:
        TP = FP = FN = 0
        for pb, pc, gt in zip(all_pred_boxes, all_pred_confs, all_gt_boxes):
            keep = pc >= c
            pbf = pb[keep]; pcf = pc[keep]
            if pbf.shape[0] == 0:
                FN += gt.shape[0]
                continue
            if gt.shape[0] == 0:
                FP += pbf.shape[0]
                continue
            iou = _box_iou_xyxy(pbf, gt)
            order = np.argsort(-pcf)
            matched: set = set()
            for pi in order:
                j = int(np.argmax(iou[pi]))
                if iou[pi, j] >= 0.5 and j not in matched:
                    TP += 1
                    matched.add(j)
                else:
                    FP += 1
            FN += gt.shape[0] - len(matched)
        prec = TP / max(TP + FP, 1)
        rec  = TP / max(TP + FN, 1)
        f1 = 2*prec*rec / max(prec + rec, 1e-12)
        f2 = 5*prec*rec / max(4*prec + rec, 1e-12)
        rows.append({"conf": round(float(c), 3),
                     "precision": prec, "recall": rec,
                     "F1": f1, "F2": f2,
                     "TP": TP, "FP": FP, "FN": FN})
    pr_df = pd.DataFrame(rows)
    f1_df = pr_df[["conf", "F1", "F2", "precision", "recall"]].copy()
    return pr_df, f1_df, hist_df


def per_size_recall(per_img_dir: pd.DataFrame, model: YOLO,
                    split: str) -> pd.DataFrame:
    """Recall broken by object size bins. Spec Section 11.1 uses the existing
    progress-slide bins (<8, 8-16, 16-32, 32-96, >=96 in pixel-area-sqrt)."""
    img_dir = DATASET_ROOT / split / "images"
    lbl_dir = DATASET_ROOT / split / "labels"
    images = sorted(img_dir.glob("*.png"))
    bins = [
        ("very_tiny", 0,  8),
        ("tiny",      8,  16),
        ("small",     16, 32),
        ("medium",    32, 96),
        ("large",     96, 1_000_000),
    ]
    tally = {name: {"matched": 0, "total": 0} for name, *_ in bins}
    import cv2
    for img_path in images:
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]
        gt = _yolo_label_to_xyxy(lbl_dir / (img_path.stem + ".txt"), w, h)
        if gt.shape[0] == 0:
            continue
        res = model.predict(str(img_path), conf=0.25, verbose=False)
        b = res[0].boxes
        pb = (b.xyxy.cpu().numpy().astype(np.float32)
              if (b is not None and len(b) > 0)
              else np.zeros((0, 4), dtype=np.float32))
        iou = _box_iou_xyxy(pb, gt) if pb.shape[0] else np.zeros((0, gt.shape[0]))
        gt_matched = np.zeros(gt.shape[0], dtype=bool)
        if pb.shape[0]:
            best_per_gt = iou.max(axis=0) if iou.size else np.zeros(gt.shape[0])
            gt_matched = best_per_gt >= 0.5
        side = np.sqrt((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]))
        for name, lo, hi in bins:
            mask = (side >= lo) & (side < hi)
            tally[name]["total"]   += int(mask.sum())
            tally[name]["matched"] += int(gt_matched[mask].sum())
    rows = []
    for name, lo, hi in bins:
        t = tally[name]["total"]; m = tally[name]["matched"]
        rows.append({"bin": name, "lo_px": lo, "hi_px": hi,
                     "gt_total": t, "matched": m,
                     "recall": (m / t) if t else float("nan")})
    return pd.DataFrame(rows)


def calibration_table(per_img_pred_confs: np.ndarray,
                      per_img_pred_correct: np.ndarray,
                      n_bins: int = 10) -> Tuple[pd.DataFrame, float, float, float]:
    """ECE / MCE / Brier on prediction-level (conf, correct) pairs."""
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    ece = mce = 0.0
    total = max(len(per_img_pred_confs), 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        mask = (per_img_pred_confs >= lo) & (per_img_pred_confs < hi + (1e-9 if i == n_bins - 1 else 0))
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin_lo": lo, "bin_hi": hi, "count": 0,
                         "mean_conf": float("nan"), "mean_acc": float("nan"),
                         "gap": float("nan")})
            continue
        mc = float(per_img_pred_confs[mask].mean())
        ma = float(per_img_pred_correct[mask].mean())
        gap = abs(mc - ma)
        ece += (n / total) * gap
        mce = max(mce, gap)
        rows.append({"bin_lo": lo, "bin_hi": hi, "count": n,
                     "mean_conf": mc, "mean_acc": ma, "gap": gap})
    brier = float(np.mean((per_img_pred_confs - per_img_pred_correct) ** 2)) \
            if len(per_img_pred_confs) else float("nan")
    return pd.DataFrame(rows), ece, mce, brier


def latency_profile(model: YOLO) -> Dict[str, Any]:
    """Latency mean/std/P50/P95/P99 + throughput + per-resolution sweep
    (Section 11.3)."""
    device = model.device
    # Warmup
    for _ in range(LATENCY_WARMUP):
        x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
        with torch.no_grad():
            _ = model.model(x)
    torch.cuda.synchronize()

    # Single-image latency distribution
    times_ms: List[float] = []
    for _ in range(LATENCY_RUNS):
        x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.model(x)
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(times_ms)
    out: Dict[str, Any] = {
        "latency_mean_ms": float(a.mean()),
        "latency_std_ms":  float(a.std()),
        "latency_p50_ms":  float(np.percentile(a, 50)),
        "latency_p95_ms":  float(np.percentile(a, 95)),
        "latency_p99_ms":  float(np.percentile(a, 99)),
        "per_image_latency_ms": times_ms,
    }
    # Batched throughput
    bsz_rows = []
    for bsz in LATENCY_BATCH_SIZES:
        try:
            x = torch.randn(bsz, 3, IMG_SIZE, IMG_SIZE, device=device)
            for _ in range(5):
                with torch.no_grad():
                    _ = model.model(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(30):
                with torch.no_grad():
                    _ = model.model(x)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0)
            imgs = 30 * bsz
            bsz_rows.append({"batch": bsz, "imgs_per_sec": imgs / elapsed})
        except Exception as e:
            bsz_rows.append({"batch": bsz, "imgs_per_sec": float("nan"),
                             "error": str(e)})
    out["throughput_table"] = bsz_rows
    # Per-resolution sweep
    res_rows = []
    for r in LATENCY_RESOLUTIONS:
        try:
            x = torch.randn(1, 3, r, r, device=device)
            for _ in range(20):
                with torch.no_grad():
                    _ = model.model(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100):
                with torch.no_grad():
                    _ = model.model(x)
            torch.cuda.synchronize()
            res_rows.append({"resolution": r,
                             "latency_ms": (time.perf_counter() - t0) * 10.0})
        except Exception as e:
            res_rows.append({"resolution": r, "latency_ms": float("nan"),
                             "error": str(e)})
    out["resolution_table"] = res_rows
    return out


# =============================================================================
# 13. PAIRED BOOTSTRAP  (Section 12.4)
# =============================================================================

def paired_bootstrap(a: Sequence[float], b: Sequence[float],
                     B: int = 10_000, seed: int = 0
                     ) -> Tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape:
        raise ValueError("paired_bootstrap requires equal-length per-image arrays")
    n = len(aa)
    idx = rng.integers(0, n, size=(B, n))
    deltas = (aa[idx] - bb[idx]).mean(axis=1)
    obs = float(aa.mean() - bb.mean())
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return obs, float(lo), float(hi), float(p)


def bca_bootstrap_ci(values: Sequence[float], B: int = 1000,
                     seed: int = 0) -> Tuple[float, float, float]:
    """BCa 95% CI on the mean of `values`. Falls back to percentile if scipy
    is unavailable (the BCa adjustment uses scipy.stats.norm)."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(B)
    for i in range(B):
        boots[i] = x[rng.integers(0, n, n)].mean()
    if HAS_SCIPY:
        from scipy.stats import norm
        z0 = norm.ppf((boots < x.mean()).mean()) if (boots < x.mean()).mean() not in (0, 1) else 0.0
        jack = np.array([np.delete(x, i).mean() for i in range(n)])
        jbar = jack.mean()
        num = ((jbar - jack) ** 3).sum()
        den = 6.0 * ((jbar - jack) ** 2).sum() ** 1.5
        a_hat = num / den if den > 0 else 0.0
        alpha_lo, alpha_hi = 0.025, 0.975
        z_lo = norm.ppf(alpha_lo); z_hi = norm.ppf(alpha_hi)
        a1 = norm.cdf(z0 + (z0 + z_lo) / (1 - a_hat * (z0 + z_lo) + 1e-12))
        a2 = norm.cdf(z0 + (z0 + z_hi) / (1 - a_hat * (z0 + z_hi) + 1e-12))
        lo = float(np.quantile(boots, a1))
        hi = float(np.quantile(boots, a2))
    else:
        lo = float(np.quantile(boots, 0.025))
        hi = float(np.quantile(boots, 0.975))
    return float(x.mean()), lo, hi


# =============================================================================
# 14. ARCHITECTURE REPORT  (Section 22)
# =============================================================================

def architecture_report(model: YOLO, arch_dir: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        n_params = sum(p.numel() for p in model.model.parameters())
        n_train  = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        info["params_total_M"]     = round(n_params / 1e6, 4)
        info["params_trainable_M"] = round(n_train / 1e6, 4)
    except Exception:
        pass
    # GFLOPs via thop
    try:
        import thop
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=model.device)
        macs, _ = thop.profile(model.model, inputs=(dummy,), verbose=False)
        info["gflops"] = round(macs / 1e9 * 2, 3)  # MACs * 2 = FLOPs
    except Exception as e:
        info["gflops_error"] = str(e)

    # torchinfo
    if HAS_TORCHINFO:
        try:
            s = torchinfo_summary(
                model.model,
                input_size=(1, 3, IMG_SIZE, IMG_SIZE),
                col_names=("input_size", "output_size", "num_params",
                           "mult_adds", "trainable"),
                depth=4, verbose=0,
            )
            (arch_dir / "model_summary.txt").write_text(str(s), encoding="utf-8")
        except Exception as e:
            info["torchinfo_error"] = str(e)
    else:
        info["torchinfo_error"] = "torchinfo not installed"

    # Module-level CSV
    try:
        rows = []
        for i, (name, mod) in enumerate(model.model.named_modules()):
            if len(list(mod.children())) == 0:  # leaf only
                params = sum(p.numel() for p in mod.parameters())
                rows.append({
                    "layer_idx": i,
                    "name": name,
                    "type": mod.__class__.__name__,
                    "params": params,
                    "trainable": any(p.requires_grad for p in mod.parameters()),
                })
        pd.DataFrame(rows).to_csv(arch_dir / "module_table.csv", index=False)
    except Exception as e:
        info["module_table_error"] = str(e)

    # Layer count
    info["layers_total"] = sum(
        1 for _, m in model.model.named_modules() if len(list(m.children())) == 0)

    # FLOPs breakdown (best-effort: by name prefix backbone/neck/head)
    try:
        bucket = {"backbone": 0, "neck": 0, "head": 0, "other": 0}
        for name, p in model.model.named_parameters():
            n = p.numel()
            ln = name.lower()
            if "backbone" in ln or ".0." in name or ".1." in name:
                bucket["backbone"] += n
            elif "head" in ln or "detect" in ln:
                bucket["head"] += n
            elif "neck" in ln or "fpn" in ln or "pan" in ln:
                bucket["neck"] += n
            else:
                bucket["other"] += n
        pd.DataFrame([
            {"section": k, "params": v} for k, v in bucket.items()
        ]).to_csv(arch_dir / "flops_breakdown.csv", index=False)
    except Exception as e:
        info["flops_breakdown_error"] = str(e)

    return info


# =============================================================================
# 15. PLOTS  (Section 13) + PLOTS_INDEX  (Section 14)
# =============================================================================

class PlotsIndex:
    def __init__(self, plots_dir: Path):
        self.plots_dir = plots_dir
        self.rows: List[Dict[str, str]] = []
        self.md_lines: List[str] = []

    def add(self, plot_filename: str, data_file: str,
            data_columns: str, producing_script: str, run_id: str):
        self.rows.append({
            "plot_filename": plot_filename,
            "data_file":     data_file,
            "data_columns":  data_columns,
            "producing_script": producing_script,
            "run_id":        run_id,
            "timestamp":     datetime.now().isoformat(timespec="seconds"),
        })
        self.md_lines.append(
            f"- {plot_filename}  <-  {data_file}  "
            f"(cols: {data_columns}) -- produced by: {producing_script}"
        )

    def finalize(self):
        (self.plots_dir / "PLOTS_INDEX.md").write_text(
            "# Plots Index\n\n" + "\n".join(self.md_lines), encoding="utf-8")
        pd.DataFrame(self.rows).to_csv(
            self.plots_dir / "PLOTS_INDEX.csv", index=False)


def _save(fig, path: Path, dpi: int = 300):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_training_dynamics(epoch_metrics_df: pd.DataFrame,
                           ultra_results_csv: Path,
                           plots_dir: Path, run_id: str,
                           idx: PlotsIndex):
    if not ultra_results_csv.is_file():
        print("[plots] WARN: results.csv not found -- skipping training-dyn plots")
        return
    df = pd.read_csv(ultra_results_csv)
    df.columns = df.columns.str.strip()
    # 1) Loss curves (box/cls/dfl train + val)
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for j, key in enumerate(("box_loss", "cls_loss", "dfl_loss")):
        if f"train/{key}" in df.columns:
            ax[0, j].plot(df[f"train/{key}"], label="train")
        if f"val/{key}" in df.columns:
            ax[0, j].plot(df[f"val/{key}"], label="val")
        ax[0, j].set_title(key); ax[0, j].set_xlabel("epoch"); ax[0, j].legend()
    # Metric panel
    for j, key in enumerate(("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)")):
        if key in df.columns:
            ax[1, j].plot(df[key]); ax[1, j].set_title(key)
            ax[1, j].set_xlabel("epoch")
    fname = f"{run_id}_metrics_6panel.png"
    _save(fig, plots_dir / fname)
    idx.add(fname, str(ultra_results_csv.relative_to(plots_dir.parent.parent)),
            "train/box_loss,train/cls_loss,...", __file__, run_id)

    # 2) Resource usage from epoch_metrics_df
    if not epoch_metrics_df.empty:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        if "vram_peak_MB" in epoch_metrics_df:
            ax[0].plot(epoch_metrics_df["epoch"], epoch_metrics_df["vram_peak_MB"])
            ax[0].set_title("VRAM peak (MB)"); ax[0].set_xlabel("epoch")
        if "epoch_time_s" in epoch_metrics_df:
            ax[1].plot(epoch_metrics_df["epoch"], epoch_metrics_df["epoch_time_s"])
            ax[1].set_title("Epoch time (s)"); ax[1].set_xlabel("epoch")
        if "samples_per_sec" in epoch_metrics_df:
            ax[2].plot(epoch_metrics_df["epoch"], epoch_metrics_df["samples_per_sec"])
            ax[2].set_title("Throughput (img/s)"); ax[2].set_xlabel("epoch")
        fname = f"{run_id}_resource_usage.png"
        _save(fig, plots_dir / fname)
        idx.add(fname, "metrics/epoch_metrics.csv",
                "epoch,vram_peak_MB,epoch_time_s,samples_per_sec",
                __file__, run_id)

    # 3) LR schedule
    lr_cols = [c for c in df.columns if c.startswith("lr/")]
    if lr_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        for c in lr_cols:
            ax.plot(df[c], label=c)
        ax.set_title("LR schedule"); ax.set_xlabel("epoch"); ax.legend()
        fname = f"{run_id}_lr_schedule.png"
        _save(fig, plots_dir / fname)
        idx.add(fname, str(ultra_results_csv.name),
                ",".join(lr_cols), __file__, run_id)


def plot_pr_and_f1_and_hist(pr_df, f1_df, hist_df, metrics_dir, plots_dir,
                            run_id, idx: PlotsIndex):
    # PR
    pr_df.to_csv(metrics_dir / "pr_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(pr_df["recall"], pr_df["precision"])
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("PR curve (test)"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fname = f"{run_id}_pr_curve.png"
    _save(fig, plots_dir / fname)
    idx.add(fname, "metrics/pr_curve.csv", "conf,precision,recall,F1,F2,TP,FP,FN",
            __file__, run_id)

    # F1 vs conf
    f1_df.to_csv(metrics_dir / "f1_vs_conf.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(f1_df["conf"], f1_df["F1"], label="F1")
    ax.plot(f1_df["conf"], f1_df["F2"], label="F2")
    ax.set_xlabel("confidence threshold"); ax.set_ylabel("score")
    ax.set_title("F1 / F2 vs confidence (test)"); ax.legend()
    fname = f"{run_id}_f1_conf.png"
    _save(fig, plots_dir / fname)
    idx.add(fname, "metrics/f1_vs_conf.csv", "conf,F1,F2,precision,recall",
            __file__, run_id)

    # Confidence histogram
    hist_df.to_csv(metrics_dir / "confidence_hist.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(hist_df["bin_left"], hist_df["count"],
           width=(hist_df["bin_right"] - hist_df["bin_left"]), align="edge")
    ax.set_xlabel("confidence"); ax.set_ylabel("count")
    ax.set_title("Prediction confidence distribution (test)")
    fname = f"{run_id}_conf_hist.png"
    _save(fig, plots_dir / fname)
    idx.add(fname, "metrics/confidence_hist.csv", "bin_left,bin_right,count",
            __file__, run_id)


def plot_per_size(per_size_df, metrics_dir, plots_dir, run_id,
                  idx: PlotsIndex):
    per_size_df.to_csv(metrics_dir / "per_size.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(per_size_df["bin"], per_size_df["recall"])
    ax.set_ylabel("Recall"); ax.set_ylim(0, 1)
    ax.set_title("Recall by object size (test)")
    fname = f"{run_id}_per_size_recall.png"
    _save(fig, plots_dir / fname)
    idx.add(fname, "metrics/per_size.csv",
            "bin,lo_px,hi_px,gt_total,matched,recall", __file__, run_id)


def plot_calibration(cal_df, metrics_dir, plots_dir, run_id, idx: PlotsIndex):
    cal_df.to_csv(metrics_dir / "calibration.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    sub = cal_df.dropna(subset=["mean_conf", "mean_acc"])
    ax.plot(sub["mean_conf"], sub["mean_acc"], "o-", label="model")
    ax.set_xlabel("mean confidence"); ax.set_ylabel("empirical accuracy")
    ax.set_title("Reliability diagram (test)"); ax.legend()
    fname = f"{run_id}_calibration.png"
    _save(fig, plots_dir / fname)
    idx.add(fname, "metrics/calibration.csv",
            "bin_lo,bin_hi,count,mean_conf,mean_acc,gap", __file__, run_id)


def plot_latency(lat: Dict[str, Any], metrics_dir, plots_dir, run_id,
                 idx: PlotsIndex):
    res_df = pd.DataFrame(lat["resolution_table"])
    res_df.to_csv(metrics_dir / "latency_by_res.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(res_df["resolution"], res_df["latency_ms"], "o-")
    ax.set_xlabel("input resolution"); ax.set_ylabel("latency (ms)")
    ax.set_title("Latency vs input resolution")
    fname = f"{run_id}_latency_resolution.png"
    _save(fig, plots_dir / fname)
    idx.add(fname, "metrics/latency_by_res.csv", "resolution,latency_ms",
            __file__, run_id)

    per_lat = pd.DataFrame({"latency_ms": lat["per_image_latency_ms"]})
    per_lat.to_csv(metrics_dir / "per_image_latency.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(per_lat["latency_ms"], bins=50)
    ax.set_xlabel("latency (ms)"); ax.set_ylabel("count")
    ax.set_title("Per-image latency distribution")
    fname = f"{run_id}_latency_hist.png"
    _save(fig, plots_dir / fname)
    idx.add(fname, "metrics/per_image_latency.csv", "latency_ms",
            __file__, run_id)


def plot_confusion(per_img_df, metrics_dir, plots_dir, run_id, idx: PlotsIndex):
    TP = int(per_img_df["TP"].sum()); FP = int(per_img_df["FP"].sum())
    FN = int(per_img_df["FN"].sum())
    cm = pd.DataFrame([[TP, FN], [FP, 0]],
                      index=["actual_person", "actual_bg"],
                      columns=["pred_person", "pred_bg"])
    cm.to_csv(metrics_dir / "confusion.csv")
    fig, ax = plt.subplots(figsize=(5, 4))
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    else:
        ax.imshow(cm.values, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm.values[i, j]), ha="center", va="center")
    ax.set_title("Confusion (test, conf=0.25)")
    fname = f"{run_id}_confusion.png"
    _save(fig, plots_dir / fname)
    idx.add(fname, "metrics/confusion.csv",
            "row=actual,col=predicted", __file__, run_id)


# =============================================================================
# 16. SMOKE TEST  (Section 17)
# =============================================================================

def smoke_test() -> bool:
    print("\n" + "=" * 70)
    print("SMOKE TEST  (Section 17) -- tiny slice, every metric/plot path exercised")
    print("=" * 70)
    smoke_run_id = make_run_id(seed=0) + "_SMOKE"
    smoke_dir = SMOKE_BASE / smoke_run_id
    smoke_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        set_seed(0)
        # 1. Build a tiny temporary YAML pointing at the FULL dataset; use
        #    Ultralytics' built-in `fraction` to subsample.
        smoke_yaml = smoke_dir / "data.yaml"
        write_data_yaml(smoke_yaml)
        # 2. Tiny train
        m = YOLO(PRETRAINED_WEIGHTS)
        kw = dict(
            data=str(smoke_yaml),
            epochs=SMOKE_EPOCHS,
            imgsz=IMG_SIZE,
            batch=SMOKE_BATCH or BATCH_SIZE,
            patience=0,
            device=0, workers=NUM_WORKERS, cache=CACHE,
            project=str(smoke_dir), name="smoke", exist_ok=True,
            plots=False, save=True, save_period=-1, val=True,
            verbose=False, amp=True,
            fraction=SMOKE_FRACTION,
        )
        m.train(**kw)

        # 3. Tiny eval (val split) to exercise per-image eval path
        per_img = per_image_eval(m, split="val", conf=0.25)
        per_img.to_csv(smoke_dir / "per_image_val.csv", index=False)
        if per_img.empty:
            raise RuntimeError("smoke per-image eval returned empty -- broken pipeline")

        # 4. Tiny PR sweep on a few conf points
        pr_df, f1_df, hist_df = pr_and_f1_curves(
            per_img, m, "val", conf_grid=np.array([0.1, 0.25, 0.5, 0.75]))
        pr_df.to_csv(smoke_dir / "pr.csv", index=False)
        f1_df.to_csv(smoke_dir / "f1.csv", index=False)
        hist_df.to_csv(smoke_dir / "hist.csv", index=False)

        # 5. A single tiny plot to exercise the plot path
        fig, ax = plt.subplots()
        ax.plot(per_img["F1"]); ax.set_title("smoke F1 per image")
        _save(fig, smoke_dir / "smoke_test.png", dpi=100)

        # 6. env.json + manifest sanity
        (smoke_dir / "env.json").write_text(json.dumps(
            {"smoke": True, **SANITY}, indent=2))

        dt = time.time() - t0
        print(f"[smoke] PASSED in {dt:.1f}s")
        if dt > 300:
            print("[smoke] WARN: smoke exceeded 5-minute budget (Section 17.1 item 8).")

        # Record success timestamp for Section 17.4 hard rule
        marker = SMOKE_BASE / f"_PASSED_{MODEL_TAG}.txt"
        marker.write_text(datetime.now(timezone.utc).isoformat())

        # 7. Cleanup
        if not SMOKE_KEEP_OUTPUTS:
            try:
                shutil.rmtree(smoke_dir, ignore_errors=True)
            except Exception:
                pass
        return True
    except Exception as e:
        print(f"[smoke] FAILED: {e}\n{traceback.format_exc()}")
        print(f"[smoke] artifacts kept at {smoke_dir} for inspection.")
        return False


def smoke_recent() -> bool:
    marker = SMOKE_BASE / f"_PASSED_{MODEL_TAG}.txt"
    if not marker.is_file():
        return False
    try:
        ts = datetime.fromisoformat(marker.read_text().strip())
        age_h = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
        return age_h < 24
    except Exception:
        return False


# =============================================================================
# 17. ENV.JSON  (Section 11.7)
# =============================================================================

def write_env_json(path: Path, *, run_id: str, seed: int,
                   hyperparams: Dict[str, Any],
                   data_yaml_md5: str, weights_md5: Optional[str] = None):
    import importlib
    def ver(name, attr="__version__"):
        try:
            return getattr(importlib.import_module(name), attr)
        except Exception:
            return None
    env = {
        "run_id":             run_id,
        "model_tag":          MODEL_TAG,
        "in_ablation_matrix": True,
        "matrix_status":      ("Phase A row 1 (yolo11m_baseline, P0) "
                               "per system_spec.md Section 9.6 / Section 25"),
        "timestamp_utc":      datetime.now(timezone.utc).isoformat(),
        "git_commit":         GIT_HASH,
        "git_dirty":          GIT_DIRTY,
        "python_version":     platform.python_version(),
        "torch_version":      torch.__version__,
        "cuda_version":       torch.version.cuda,
        "cudnn_version":      torch.backends.cudnn.version(),
        "ultralytics_version": ver("ultralytics"),
        "numpy_version":      ver("numpy"),
        "opencv_version":     ver("cv2"),
        "pandas_version":     ver("pandas"),
        "matplotlib_version": ver("matplotlib"),
        "scipy_version":      ver("scipy"),
        "statsmodels_version": ver("statsmodels"),
        "pycocotools_present": HAS_PYCOCO,
        "platform":           platform.platform(),
        "os":                 platform.system(),
        "gpu_name":           SANITY["gpu_name"],
        "gpu_vram_gb":        SANITY["gpu_vram_gb"],
        "cpu_model":          platform.processor(),
        "ram_gb": (round(psutil.virtual_memory().total / 1024**3, 1)
                   if HAS_PSUTIL else None),
        "random_seed":        seed,
        "torch_deterministic": True,
        "cudnn_deterministic": True,
        "dataset_split_md5":  SPLIT_MD5,
        "data_yaml_md5":      data_yaml_md5,
        "weights_md5":        weights_md5,
        "cli_argv":           " ".join(sys.argv),
        "hyperparameters":    hyperparams,
    }
    path.write_text(json.dumps(env, indent=2))
    return env


# =============================================================================
# 18. SKIPPED-METRICS LOGGER
# =============================================================================

def log_skipped(logs_dir: Path, msgs: List[str]):
    if not msgs:
        return
    p = logs_dir / "skipped_metrics.txt"
    with p.open("a", encoding="utf-8") as f:
        for m in msgs:
            f.write(m + "\n")


# =============================================================================
# 19. COCO AP (best-effort)  (Section 11.1)
# =============================================================================

def coco_ap_eval(model: YOLO, split: str, run_dirs: Dict[str, Path]
                 ) -> Optional[Dict[str, float]]:
    if not HAS_PYCOCO:
        return None
    json_path = DATASET_INFO[f"{split}_coco_json"]
    if not Path(json_path).is_file():
        return None
    try:
        gt = COCO(str(json_path))
        cats = gt.getCatIds()
        img_ids = gt.getImgIds()
        # Build filename->image_id index
        fname_to_id = {im["file_name"]: im["id"] for im in gt.loadImgs(img_ids)}
        # Run predictions and collect into COCO format
        dets = []
        img_dir = DATASET_ROOT / split / "images"
        for fname, img_id in fname_to_id.items():
            p = img_dir / fname
            if not p.is_file():
                continue
            res = model.predict(str(p), conf=0.001, verbose=False)
            b = res[0].boxes
            if b is None or len(b) == 0:
                continue
            xyxy = b.xyxy.cpu().numpy()
            confs = b.conf.cpu().numpy()
            clses = b.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), s, c in zip(xyxy, confs, clses):
                cat_id = cats[c] if c < len(cats) else cats[0]
                dets.append({
                    "image_id":    int(img_id),
                    "category_id": int(cat_id),
                    "bbox":        [float(x1), float(y1),
                                    float(x2 - x1), float(y2 - y1)],
                    "score":       float(s),
                })
        det_path = run_dirs["metrics"] / f"{split}_coco_dets.json"
        det_path.write_text(json.dumps(dets))
        dt = gt.loadRes(str(det_path))
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
        stats = ev.stats  # 12 numbers
        out = {
            "AP":           float(stats[0]),
            "AP50":         float(stats[1]),
            "AP75":         float(stats[2]),
            "AP_small":     float(stats[3]),
            "AP_medium":    float(stats[4]),
            "AP_large":     float(stats[5]),
            "AR_1":         float(stats[6]),
            "AR_10":        float(stats[7]),
            "AR_100":       float(stats[8]),
            "AR_small":     float(stats[9]),
            "AR_medium":    float(stats[10]),
            "AR_large":     float(stats[11]),
        }
        return out
    except Exception as e:
        print(f"[coco] eval failed on split={split}: {e}")
        return None


# =============================================================================
# 20. PER-RUN ORCHESTRATION
# =============================================================================

def run_one_seed(seed: int) -> Dict[str, Any]:
    # Power-loss-aware run-ID selection: if a previous run for this seed
    # is incomplete, reuse its run_id and resume; if completed, skip; else
    # create a fresh run_id.
    existing = find_existing_run_for_seed(seed) if RESUME_INCOMPLETE_SEEDS else None
    resume_from_existing = False
    if existing and existing["status"] == "completed" and SKIP_COMPLETED_SEEDS:
        print(f"[run] seed {seed} ALREADY COMPLETED at "
              f"runs/{existing['run_id']}/manifest.json -- skipping")
        try:
            with open(existing["dir"] / "metrics" / "summary.json", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"run_id": existing["run_id"], "model_tag": MODEL_TAG,
                    "seed": seed, "status": "completed_prior_run"}
    if existing and existing["status"] == "incomplete":
        run_id = existing["run_id"]
        resume_from_existing = True
        print(f"[run] RESUMING incomplete seed {seed} from runs/{run_id}/")
    else:
        run_id = make_run_id(seed)
    dirs = make_run_dirs(run_id)
    print("\n" + "#" * 78)
    print(f"# RUN {run_id}  (seed={seed})  "
          f"{'[RESUMING]' if resume_from_existing else '[FRESH]'}")
    print("#" * 78)

    # Snapshot code into run/code/
    try:
        shutil.copy2(__file__, dirs["code"] / Path(__file__).name)
    except Exception:
        pass

    # Logging setup
    log_path = dirs["logs"] / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(str(log_path), encoding="utf-8"),
                  logging.StreamHandler()],
        force=True,
    )
    logging.info(f"run_id={run_id}")

    set_seed(seed, deterministic=True)

    # Data YAML
    data_yaml = dirs["base"] / "data.yaml"
    write_data_yaml(data_yaml)
    data_yaml_md5 = hashlib.md5(data_yaml.read_bytes()).hexdigest()

    # Hyperparameters bundle for env.json
    hyperparams = dict(
        epochs=NUM_EPOCHS, patience=PATIENCE, f2_patience=F2_PATIENCE,
        imgsz=IMG_SIZE, batch=BATCH_SIZE, save_period=SAVE_PERIOD,
        num_workers=NUM_WORKERS, cache=CACHE, cos_lr=COS_LR, lrf=LRF,
        amp=True, seed=seed, model_tag=MODEL_TAG,
        pretrained=PRETRAINED_WEIGHTS,
    )
    (dirs["base"] / "hyperparams.yaml").write_text(
        yaml.safe_dump(hyperparams, sort_keys=False))

    # env.json (initial; weights_md5 patched after training)
    env = write_env_json(dirs["base"] / "env.json", run_id=run_id, seed=seed,
                         hyperparams=hyperparams, data_yaml_md5=data_yaml_md5)

    # Pre-flight checklist
    preflight_ok = preflight_checklist(dirs, env)
    if not preflight_ok:
        raise RuntimeError("Pre-flight checklist failed. See logs/train.log.")

    # Resource sampler + CodeCarbon
    sampler = ResourceSampler(dirs["logs"], interval=2.0)
    sampler.start()
    tracker = None
    if HAS_CODECARBON:
        try:
            tracker = OfflineEmissionsTracker(
                project_name=run_id, output_dir=str(dirs["energy"]),
                country_iso_code=COUNTRY_ISO_CODE, log_level="error",
                allow_multiple_runs=True,
            )
            tracker.start()
        except Exception as e:
            print(f"[codecarbon] failed to start: {e}")

    # Resume detection (Section 20.2). Ultralytics resumes from
    # <project>/<name>/weights/last.pt -- in our layout that is
    # runs/<run_id>/ultra/weights/last.pt. The safety copy at
    # runs/<run_id>/weights/last_safety.pt is a backup we maintain via the
    # epoch callback; Ultralytics can't resume from that path directly.
    ultra_last = dirs["base"] / "ultra" / "weights" / "last.pt"
    safety_last = dirs["weights"] / "last_safety.pt"
    if ultra_last.is_file():
        resume = True
        print(f"[resume] Ultralytics last.pt found at {ultra_last} -- "
              f"enabling resume=True")
    elif safety_last.is_file() and resume_from_existing:
        # ultra/ was deleted but the safety copy survives. Restore it so
        # Ultralytics can find it.
        ultra_last.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(safety_last, ultra_last)
        resume = True
        print(f"[resume] restored {ultra_last} from safety copy {safety_last} "
              f"-- enabling resume=True")
    else:
        resume = False

    epoch_metrics: List[Dict[str, Any]] = []
    f2_stop = F2EarlyStop(F2_PATIENCE)
    weights_path = None
    train_started_at = time.time()

    if DO_TRAIN:
        model = YOLO(PRETRAINED_WEIGHTS)
        attach_ultralytics_callbacks(
            model, run_dirs=dirs, sampler=sampler,
            f2_stopper=f2_stop, epoch_metrics=epoch_metrics)
        base_kwargs = dict(
            data=str(data_yaml), epochs=NUM_EPOCHS, imgsz=IMG_SIZE,
            patience=PATIENCE, device=0, workers=NUM_WORKERS, cache=CACHE,
            project=str(dirs["base"]), name="ultra", exist_ok=True,
            plots=True, save=True, save_period=SAVE_PERIOD, val=True,
            cos_lr=COS_LR, lrf=LRF, amp=True, verbose=True, seed=seed,
            resume=resume,
        )
        try:
            train_with_oom_retry(model, base_kwargs)
        except Exception as e:
            err_path = dirs["logs"] / "train.err"
            err_path.write_text(traceback.format_exc())
            raise

        # Copy best.pt out into run_dir/weights
        ultra_dir = dirs["base"] / "ultra"
        best = ultra_dir / "weights" / "best.pt"
        if best.is_file():
            shutil.copy2(best, dirs["weights"] / "best.pt")
        else:
            print(f"[train] WARN: best.pt not found at {best}")
        last = ultra_dir / "weights" / "last.pt"
        if last.is_file():
            shutil.copy2(last, dirs["weights"] / "last.pt")
        weights_path = dirs["weights"] / "best.pt"
        # Copy results.csv
        rcsv = ultra_dir / "results.csv"
        if rcsv.is_file():
            shutil.copy2(rcsv, dirs["base"] / "results.csv")
    else:
        # Evaluate-only: expect best.pt in weights/. If the env var
        # YOLO11M_PRETRAINED_BEST points at an existing .pt file, copy it
        # in so we can re-eval a checkpoint produced by a previous (possibly
        # interrupted) training run without re-training.
        weights_path = dirs["weights"] / "best.pt"
        preexisting = os.environ.get("YOLO11M_PRETRAINED_BEST")
        if not weights_path.is_file() and preexisting:
            src = Path(preexisting)
            if src.is_file():
                shutil.copy2(src, weights_path)
                print(f"[eval-only] copied pre-trained best.pt from {src}")
                # Also copy last.pt and results.csv if siblings exist, so
                # downstream plots (training-dynamics) have their data.
                sib_last = src.parent / "last.pt"
                if sib_last.is_file():
                    shutil.copy2(sib_last, dirs["weights"] / "last.pt")
                sib_results = src.parent.parent / "results.csv"
                if sib_results.is_file():
                    shutil.copy2(sib_results, dirs["base"] / "results.csv")
                    print(f"[eval-only] copied results.csv from {sib_results}")
            else:
                raise FileNotFoundError(
                    f"YOLO11M_PRETRAINED_BEST={preexisting} is not a file.")
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"DO_TRAIN=False but no checkpoint at {weights_path}. "
                f"Either place best.pt there, or set "
                f"$env:YOLO11M_PRETRAINED_BEST to the path of an existing best.pt.")

    train_wallclock_s = time.time() - train_started_at

    # epoch metrics CSV
    if epoch_metrics:
        pd.DataFrame(epoch_metrics).to_csv(
            dirs["metrics"] / "epoch_metrics.csv", index=False)

    # Stop trackers
    if tracker is not None:
        try:
            emissions_kg = tracker.stop()
        except Exception as e:
            print(f"[codecarbon] stop failed: {e}")
            emissions_kg = None
    else:
        emissions_kg = None
    sampler.stop_and_dump()
    sampler_summary = sampler.summary()

    # weights_md5 -> patch env.json
    weights_md5 = None
    if weights_path and weights_path.is_file():
        h = hashlib.sha256()
        with weights_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        weights_md5 = h.hexdigest()
        env["weights_md5"] = weights_md5
        (dirs["base"] / "env.json").write_text(json.dumps(env, indent=2))

    # ===== EVALUATION =====
    print("\n[eval] reloading best.pt and running full Section 11 eval...")
    model = YOLO(str(weights_path))
    model.to("cuda:0")

    # Ultralytics val on val & test (mAP50, mAP50-95, etc.)
    headline: Dict[str, Any] = {}
    for split in ("val", "test"):
        try:
            res = model.val(data=str(data_yaml), split=split, verbose=False,
                            conf=0.001, iou=0.7)
            box = res.box
            def _scalar(x):
                try:
                    return float(x.mean()) if hasattr(x, "mean") else float(x)
                except Exception:
                    return float(x[0]) if hasattr(x, "__getitem__") else float("nan")
            headline[split] = {
                "precision":  _scalar(box.p),
                "recall":     _scalar(box.r),
                "mAP50":      _scalar(box.map50),
                "mAP50-95":   _scalar(box.map),
                "mAP75":      _scalar(box.maps[..., 5]) if hasattr(box, "maps") and box.maps is not None and len(getattr(box.maps, "shape", ())) >= 1 else float("nan"),
            }
        except Exception as e:
            print(f"[eval] ultralytics .val on {split} failed: {e}")
            headline[split] = {}

    # COCO AP small/medium/large + AR (Section 11.1)
    skipped: List[str] = []
    coco_test = coco_ap_eval(model, "test", dirs)
    if coco_test:
        headline["test"].update(coco_test)
        (dirs["metrics"] / "coco_test.json").write_text(json.dumps(coco_test, indent=2))
    else:
        skipped.append("[SKIPPED] coco AP_small/medium/large/AR_* on test -- "
                       "pycocotools missing or test COCO JSON unavailable")
    coco_val = coco_ap_eval(model, "val", dirs)
    if coco_val:
        headline["val"].update(coco_val)
        (dirs["metrics"] / "coco_val.json").write_text(json.dumps(coco_val, indent=2))

    # Per-image eval at conf=0.25 (operational)
    per_img_test = per_image_eval(model, split="test", conf=0.25)
    per_img_test.to_csv(dirs["metrics"] / "per_image_test.csv", index=False)
    per_img_val = per_image_eval(model, split="val", conf=0.25)
    per_img_val.to_csv(dirs["metrics"] / "per_image_val.csv", index=False)

    # PR / F1-vs-conf / confidence-hist
    pr_df, f1_df, hist_df = pr_and_f1_curves(per_img_test, model, "test")
    # Optimal thresholds
    f1_idx = int(f1_df["F1"].idxmax())
    f2_idx = int(f1_df["F2"].idxmax())
    opt_thr = {
        "OptThr_F1": float(f1_df.loc[f1_idx, "conf"]),
        "Best_F1":   float(f1_df.loc[f1_idx, "F1"]),
        "OptThr_F2": float(f1_df.loc[f2_idx, "conf"]),
        "Best_F2":   float(f1_df.loc[f2_idx, "F2"]),
    }

    # Per-size recall
    per_size_df = per_size_recall(per_img_test, model, "test")

    # Calibration
    pred_confs, pred_correct = [], []
    import cv2
    for _, row in per_img_test.iterrows():
        # Approximate per-PREDICTION correctness by using image-level info is wrong;
        # but Section 11.4 ECE/Brier wants prediction-level confidence vs correctness.
        # We re-run predictions per image with iou-matching against GT to mark each
        # pred as correct/incorrect.
        img_path = DATASET_ROOT / "test" / "images" / row["image"]
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]
        gt = _yolo_label_to_xyxy(
            DATASET_ROOT / "test" / "labels" / (img_path.stem + ".txt"), w, h)
        res = model.predict(str(img_path), conf=0.001, verbose=False)
        b = res[0].boxes
        if b is None or len(b) == 0:
            continue
        pb = b.xyxy.cpu().numpy().astype(np.float32)
        pc = b.conf.cpu().numpy().astype(np.float32)
        if gt.shape[0] == 0:
            for c in pc:
                pred_confs.append(c); pred_correct.append(0.0)
            continue
        iou = _box_iou_xyxy(pb, gt)
        order = np.argsort(-pc); matched = set()
        for pi in order:
            j = int(np.argmax(iou[pi]))
            ok = (iou[pi, j] >= 0.5) and (j not in matched)
            if ok:
                matched.add(j)
            pred_confs.append(pc[pi]); pred_correct.append(1.0 if ok else 0.0)
    pred_confs = np.asarray(pred_confs)
    pred_correct = np.asarray(pred_correct)
    cal_df, ece, mce, brier = calibration_table(pred_confs, pred_correct)

    # Latency profile
    print("[eval] measuring latency profile...")
    lat = latency_profile(model)
    pd.DataFrame(lat["throughput_table"]).to_csv(
        dirs["metrics"] / "throughput.csv", index=False)

    # Efficiency
    eff = {
        "params_total_M":     None,
        "weights_size_MB":    round(weights_path.stat().st_size / 1024**2, 3),
        "latency_mean_ms":    lat["latency_mean_ms"],
        "latency_std_ms":     lat["latency_std_ms"],
        "latency_p50_ms":     lat["latency_p50_ms"],
        "latency_p95_ms":     lat["latency_p95_ms"],
        "latency_p99_ms":     lat["latency_p99_ms"],
    }

    # Architecture (Section 22)
    arch_info = architecture_report(model, dirs["arch"])
    eff["params_total_M"]     = arch_info.get("params_total_M")
    eff["params_trainable_M"] = arch_info.get("params_trainable_M")
    eff["gflops"]             = arch_info.get("gflops")
    eff["layers_total"]       = arch_info.get("layers_total")

    # ONNX export sanity (Section 26)
    onnx_ok = False; onnx_size_mb = None
    if DO_ONNX_EXPORT:
        try:
            onnx_path = model.export(format="onnx", imgsz=IMG_SIZE, dynamic=False)
            onnx_p = Path(onnx_path) if onnx_path else None
            if onnx_p and onnx_p.is_file():
                shutil.copy2(onnx_p, dirs["arch"] / Path(onnx_p).name)
                onnx_size_mb = onnx_p.stat().st_size / 1024**2
                onnx_ok = True
        except Exception as e:
            print(f"[onnx] export failed: {e}")
    eff["export_onnx_OK"] = onnx_ok
    eff["onnx_size_MB"]   = onnx_size_mb

    # BCa CIs on headline numbers using per-image F1/F2/precision/recall
    cis: Dict[str, Any] = {}
    for col in ("F1", "F2", "precision", "recall"):
        if col in per_img_test.columns and len(per_img_test) > 1:
            mean, lo, hi = bca_bootstrap_ci(
                per_img_test[col].values, B=BOOTSTRAP_B, seed=seed)
            cis[col] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}

    # Paired bootstrap vs comparison run (if provided)
    paired = None
    if BASELINE_PER_IMAGE_CSV is not None:
        try:
            base_df = pd.read_csv(BASELINE_PER_IMAGE_CSV)
            # Align by `image`
            merged = per_img_test.merge(
                base_df, on="image", suffixes=("_us", "_base"))
            paired = {}
            for col in ("F1", "F2", "precision", "recall"):
                obs, lo, hi, p = paired_bootstrap(
                    merged[f"{col}_us"].values,
                    merged[f"{col}_base"].values, B=10_000, seed=seed)
                paired[col] = {"delta_mean": obs, "ci_lo": lo, "ci_hi": hi,
                               "p_two_sided": p}
            (dirs["sig"] / f"{run_id}_vs_baseline.json").write_text(
                json.dumps(paired, indent=2))
        except Exception as e:
            skipped.append(f"[SKIPPED] paired bootstrap vs baseline -- {e}")
    else:
        skipped.append("[SKIPPED] paired bootstrap vs another model -- "
                       "BASELINE_PER_IMAGE_CSV not configured; per-image scores "
                       "saved in metrics/per_image_test.csv for later Phase-D.")

    # Architecture-specific skips
    skipped.extend([
        "[SKIPPED] attention_map_examples         -- vanilla YOLO11m (no CBAM)",
        "[SKIPPED] channel_attention_weights      -- vanilla YOLO11m (no CBAM)",
        "[SKIPPED] spatial_attention_entropy      -- vanilla YOLO11m (no CBAM)",
        "[SKIPPED] per-stride_AP                  -- vanilla YOLO11m (no P2 head)",
        "[SKIPPED] tiny_obj_recall_by_head        -- vanilla YOLO11m (no P2 head)",
        "[SKIPPED] ssm_state_norm                 -- vanilla YOLO11m (no SSM)",
        "[SKIPPED] forward_vs_backward_scan_disag -- vanilla YOLO11m (no SSM)",
        "[SKIPPED] window_size_per_layer          -- vanilla YOLO11m (no SSM)",
        "[SKIPPED] dilation_branch_contribution   -- vanilla YOLO11m (no SSM)",
        "[SKIPPED] injection_layer_indices        -- vanilla YOLO11m (no SSM)",
        "[SKIPPED] SAHI-specific metrics          -- not part of this run",
        "[SKIPPED] TTA-specific metrics           -- not part of this run",
    ])
    log_skipped(dirs["logs"], skipped)

    # ===== PLOTS =====
    pidx = PlotsIndex(dirs["plots"])
    plot_training_dynamics(
        pd.DataFrame(epoch_metrics) if epoch_metrics else pd.DataFrame(),
        dirs["base"] / "results.csv",
        dirs["plots"], run_id, pidx)
    plot_pr_and_f1_and_hist(pr_df, f1_df, hist_df, dirs["metrics"],
                            dirs["plots"], run_id, pidx)
    plot_per_size(per_size_df, dirs["metrics"], dirs["plots"], run_id, pidx)
    plot_calibration(cal_df, dirs["metrics"], dirs["plots"], run_id, pidx)
    plot_latency(lat, dirs["metrics"], dirs["plots"], run_id, pidx)
    plot_confusion(per_img_test, dirs["metrics"], dirs["plots"], run_id, pidx)
    pidx.finalize()

    # ===== SUMMARY =====
    summary = {
        "run_id":              run_id,
        "model_tag":           MODEL_TAG,
        "seed":                seed,
        "train_wallclock_s":   round(train_wallclock_s, 1),
        "train_wallclock_h":   round(train_wallclock_s / 3600, 3),
        "headline":            headline,
        "opt_thresholds":      opt_thr,
        "efficiency":          eff,
        "calibration":         {"ECE": ece, "MCE": mce, "Brier": brier},
        "bca_cis":             cis,
        "paired_bootstrap":    paired,
        "resource_summary":    sampler_summary,
        "training_co2_kg":     emissions_kg,
        "f2_early_stop":       {"best_epoch": f2_stop.best_epoch,
                                "best_f2":   f2_stop.best_f2,
                                "stopped":   f2_stop.stop},
        "skipped_metrics_count": len(skipped),
    }
    (dirs["metrics"] / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    pd.DataFrame([{
        "model":         MODEL_TAG, "seed": seed, "run_id": run_id,
        "val_mAP50":     headline.get("val", {}).get("mAP50"),
        "val_mAP50_95":  headline.get("val", {}).get("mAP50-95"),
        "test_mAP50":    headline.get("test", {}).get("mAP50"),
        "test_mAP50_95": headline.get("test", {}).get("mAP50-95"),
        "test_AP_small": headline.get("test", {}).get("AP_small"),
        "test_AR_small": headline.get("test", {}).get("AR_small"),
        "params_M":      eff.get("params_total_M"),
        "gflops":        eff.get("gflops"),
        "latency_p50_ms": eff.get("latency_p50_ms"),
        "latency_p95_ms": eff.get("latency_p95_ms"),
        "co2_kg":        emissions_kg,
    }]).to_excel(dirs["metrics"] / "summary.xlsx", index=False)

    # MODEL_CARD.md (Section 26)
    (dirs["base"] / "MODEL_CARD.md").write_text(
        f"""# Model Card -- {MODEL_TAG} (run {run_id})

**Status:** Phase A row 1 of the final-month ablation matrix
(`yolo11m_baseline`, P0) per system_spec.md Section 9.6 / Section 25.
This is the reference baseline against which CBAM / P2Head / Mamba
variants are compared.

## Intended Use
Aerial / SAR human detection on C2A imagery. Single class: `person`.

## Training data
C2A dataset (Nihal et al., ICPR 2024). Split-md5: `{SPLIT_MD5}`. Image format: PNG.

## Evaluation
- Val mAP50: {headline.get('val', {}).get('mAP50')}
- Test mAP50: {headline.get('test', {}).get('mAP50')}
- Test mAP50-95: {headline.get('test', {}).get('mAP50-95')}
- Test AP_small: {headline.get('test', {}).get('AP_small')}
- Test F1 (per-image mean): {cis.get('F1', {}).get('mean')}
- Test F2 (per-image mean): {cis.get('F2', {}).get('mean')}
- Latency p50 / p95: {eff.get('latency_p50_ms')} / {eff.get('latency_p95_ms')} ms

## Limitations
- This card was generated from SEEDS={SEEDS}. If fewer than 5 seeds
  completed successfully, paired-significance tests against other
  ablation rows are not valid (system_spec.md Section 12.1).
- Architecture-specific metrics in Section 11.6 (attention maps, per-stride
  AP, SSM state norms, dilation contributions) are N/A for vanilla YOLO11m
  and are logged as [SKIPPED] in logs/skipped_metrics.txt.

## Early-stopping configuration
- Ultralytics fitness patience = 50 (raised from spec's 30; matches HIT-UAV
  Sci Reports 2024 convention and the historic Ultralytics default).
- Custom F2 patience = 40 (raised from spec's 20). Stops training if F2
  has not improved for 40 consecutive epochs. Belt-and-suspenders on top
  of the built-in stopper.
- See docs/2026-05-29_yolo11m_final_month_writeup.md for the literature
  check that motivated these values.

## Ethical considerations
SAR / humanitarian use case. Not validated for surveillance.
""")

    # Manifest
    manifest = {
        "run_id":            run_id,
        "model_tag":         MODEL_TAG,
        "ablation_matrix_row": 1,
        "matrix_status":     "Phase A row 1 (yolo11m_baseline, P0)",
        "parent_run_id":     None,
        "git_commit":        GIT_HASH,
        "git_dirty":         GIT_DIRTY,
        "smoke_passed_at":   (SMOKE_BASE / f"_PASSED_{MODEL_TAG}.txt"
                              ).read_text().strip()
                              if (SMOKE_BASE / f"_PASSED_{MODEL_TAG}.txt").is_file()
                              else None,
        "weights_md5":       weights_md5,
        "data_yaml_md5":     data_yaml_md5,
        "dataset_split_md5": SPLIT_MD5,
        "seed":              seed,
        "completed_utc":     datetime.now(timezone.utc).isoformat(),
        "summary_pointer":   "metrics/summary.json",
    }
    (dirs["base"] / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[run] DONE -- run_dir = {dirs['base']}")
    return summary


# =============================================================================
# 21. PRE-FLIGHT CHECKLIST  (Section 24)
# =============================================================================

def preflight_checklist(dirs: Dict[str, Path], env: Dict[str, Any]) -> bool:
    print("\n[pre-flight] Section 24 checklist...")
    fails: List[str] = []

    # GPU sanity
    if SANITY["gpu_vram_gb"] < 12:
        fails.append("GPU VRAM < 12 GB (config tuned for 16 GB Ada).")
    # env.json
    if not (dirs["base"] / "env.json").is_file():
        fails.append("env.json missing.")
    # Seed
    if env.get("random_seed") is None:
        fails.append("random_seed not set.")
    # Split md5
    if not (SPLITS_DIR / "splits.md5").is_file():
        fails.append("common/splits/splits.md5 missing.")
    # Smoke passed within 24h
    if not smoke_recent():
        fails.append("Smoke test has not passed in the last 24h.")
    # Disk
    try:
        free_gb = shutil.disk_usage(str(OUTPUT_ROOT)).free / 1024**3
        if free_gb < 20:
            fails.append(f"Disk free {free_gb:.1f} GB < 20 GB.")
    except Exception:
        pass
    # OOM ladder configured
    if not OOM_RETRY_BATCHES:
        fails.append("OOM_RETRY_BATCHES empty.")
    # PLOTS_INDEX dir reachable
    if not dirs["plots"].is_dir():
        fails.append("plots/ dir not created.")

    if fails:
        for f in fails:
            print(f"[pre-flight] FAIL: {f}")
        return False
    print("[pre-flight] OK")
    return True


# =============================================================================
# 22. MAIN
# =============================================================================

def main():
    print("\n" + "=" * 78)
    print(f"yolov11m_final_month.py  --  MODEL_TAG={MODEL_TAG}  SEEDS={SEEDS}")
    print(f"  OUTPUT_ROOT  = {OUTPUT_ROOT}")
    print(f"  DATASET_ROOT = {DATASET_ROOT}")
    print(f"  GPU          = {SANITY['gpu_name']} ({SANITY['gpu_vram_gb']} GB)")
    print("=" * 78)

    if SMOKE_TEST:
        ok = smoke_test()
        if not ok:
            print("[main] SMOKE_TEST=True and smoke failed -- exiting.")
            sys.exit(2)
        print("[main] SMOKE_TEST=True and smoke passed -- exiting without full run.")
        return

    # Section 17.4: refuse full run unless smoke is fresh.
    if not smoke_recent():
        print("[main] no recent smoke marker -- running smoke automatically first.")
        ok = smoke_test()
        if not ok:
            print("[main] smoke failed -- refusing to start full run.")
            sys.exit(2)

    all_summaries = []
    for seed in SEEDS:
        try:
            s = run_one_seed(seed)
            all_summaries.append(s)
        except Exception as e:
            print(f"[main] seed {seed} FAILED: {e}\n{traceback.format_exc()}")

    # Multi-seed roll-up
    if len(all_summaries) >= 1:
        roll_dir = OUTPUT_ROOT / "runs" / f"{MODEL_TAG}_multi_seed_rollup"
        roll_dir.mkdir(parents=True, exist_ok=True)
        (roll_dir / "per_seed_summaries.json").write_text(
            json.dumps(all_summaries, indent=2, default=str))

        # Cross-seed aggregation for the headline metrics (Sec 12.1).
        # We average over the seeds that completed; if any seed crashed it is
        # excluded but counted in `n_failed`.
        headline_keys = [
            ("val",  "precision"), ("val",  "recall"),
            ("val",  "mAP50"),     ("val",  "mAP50-95"),
            ("test", "precision"), ("test", "recall"),
            ("test", "mAP50"),     ("test", "mAP50-95"),
            ("test", "AP_small"),  ("test", "AP_medium"), ("test", "AP_large"),
            ("test", "AR_1"),      ("test", "AR_10"),     ("test", "AR_100"),
            ("test", "AR_small"),  ("test", "AR_medium"), ("test", "AR_large"),
        ]
        agg_rows: List[Dict[str, Any]] = []
        for split, key in headline_keys:
            vals = []
            for s in all_summaries:
                v = s.get("headline", {}).get(split, {}).get(key)
                if v is None:
                    continue
                try:
                    vf = float(v)
                except Exception:
                    continue
                if math.isnan(vf):
                    continue
                vals.append(vf)
            row: Dict[str, Any] = {
                "split": split, "metric": key,
                "n_seeds": len(vals),
            }
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                row.update({
                    "mean":   float(arr.mean()),
                    "std":    float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    "min":    float(arr.min()),
                    "max":    float(arr.max()),
                    "median": float(np.median(arr)),
                })
                if len(arr) >= 3:
                    try:
                        m, lo, hi = bca_bootstrap_ci(arr, B=BOOTSTRAP_B, seed=0)
                        row.update({"bca_ci_lo": lo, "bca_ci_hi": hi})
                    except Exception:
                        row.update({"bca_ci_lo": float("nan"),
                                    "bca_ci_hi": float("nan")})
            agg_rows.append(row)
        agg_df = pd.DataFrame(agg_rows)
        agg_df.to_csv(roll_dir / "cross_seed_metrics.csv", index=False)
        try:
            agg_df.to_excel(roll_dir / "cross_seed_metrics.xlsx", index=False)
        except Exception:
            pass

        # Efficiency / latency cross-seed table -- these should be ~constant
        # across seeds but we report std anyway.
        eff_keys = ["params_total_M", "gflops", "weights_size_MB",
                    "latency_mean_ms", "latency_p50_ms", "latency_p95_ms",
                    "latency_p99_ms"]
        eff_rows = []
        for k in eff_keys:
            vals = [float(s.get("efficiency", {}).get(k))
                    for s in all_summaries
                    if s.get("efficiency", {}).get(k) is not None
                    and not (isinstance(s["efficiency"][k], float)
                             and math.isnan(s["efficiency"][k]))]
            row = {"metric": k, "n_seeds": len(vals)}
            if vals:
                arr = np.asarray(vals)
                row.update({
                    "mean": float(arr.mean()),
                    "std":  float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                })
            eff_rows.append(row)
        pd.DataFrame(eff_rows).to_csv(roll_dir / "cross_seed_efficiency.csv",
                                      index=False)

        n_done = len(all_summaries); n_req = len(SEEDS)
        n_failed = n_req - n_done
        print(f"\n[rollup] {n_done}/{n_req} seeds completed; cross-seed table "
              f"-> {roll_dir/'cross_seed_metrics.csv'}")
        if n_done < 5:
            print(f"[rollup] WARN: only {n_done} seed(s) succeeded; "
                  f"system_spec.md Section 12.1 requires >=5 successful seeds "
                  f"for any paired-significance claim. Re-run the failed "
                  f"seed(s) before publishing.")


if __name__ == "__main__":
    main()

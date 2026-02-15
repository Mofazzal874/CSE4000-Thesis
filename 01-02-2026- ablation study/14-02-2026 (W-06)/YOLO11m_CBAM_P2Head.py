"""
================================================================================
DISASTER HUMAN DETECTION: CBAM + P2 EXTRA DETECTION HEAD ABLATION STUDY
================================================================================

TRAINS ONLY: CBAM+P2-Head (Baseline & CBAM-only loaded from previous run)

THREE-WAY COMPARISON: Baseline vs CBAM-only vs CBAM+P2-Head

PRE-TRAINED MODELS (uploaded to Kaggle input):
  - Baseline:  /kaggle/input/yolo11vm-cbam/runs/detect/yolo11m_baseline/weights/best.pt
  - CBAM-only: /kaggle/input/yolo11vm-cbam/runs/detect/yolo11m_cbam/weights/best.pt

Based on:
  - CBAM: Woo et al. (2018) - "CBAM: Convolutional Block Attention Module"
  - P2 Head: Ultralytics yolo11-p2.yaml (PR #16558)
  - SED-YOLO: Shi et al. (2024) - extra small-object detection head

CHECKPOINT RESUME: Saves checkpoints every 5 epochs for Kaggle 12h limit.
================================================================================
"""

# ============================================================================
# CELL 1: Configuration & Dependencies
# ============================================================================

# ===================== CONTROL FLAGS =====================
TEST_MODE = True           # True = 5% data, 2 epochs | False = full run
RESUME_TRAINING = False    # Set True to resume CBAM+P2 from checkpoint
RESUME_CBAM_P2_PT = ""     # Path to cbam+p2 last.pt (if resuming)
# =========================================================

!pip uninstall ultralytics -y -q 2>/dev/null
!pip install -q -U ultralytics timm thop "pandas<3.0" "matplotlib<3.10" openpyxl scikit-learn
print("✓ Dependencies installed")


# ============================================================================
# CELL 2: Training Config
# ============================================================================
import os, sys, time, yaml, shutil, gc
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn

if TEST_MODE:
    TRAIN_FRACTION, NUM_EPOCHS, PATIENCE = 0.05, 2, 2
    TEST_IMAGES, VAL_IMAGES, SAVE_PERIOD = 10, 20, 1
    print("⚠ TEST MODE: 5% data, 2 epochs")
else:
    TRAIN_FRACTION, NUM_EPOCHS, PATIENCE = 1.0, 70, 10
    TEST_IMAGES, VAL_IMAGES, SAVE_PERIOD = 30, 100, 5
    print("🚀 FULL MODE: 100% data, 70 epochs")

print(f"  Epochs: {NUM_EPOCHS} | Fraction: {TRAIN_FRACTION} | Save every: {SAVE_PERIOD} epochs")

num_gpus = torch.cuda.device_count()
gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"\nGPUs: {num_gpus} | GPU 0: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")

# ===================== PRE-TRAINED MODEL PATHS =====================
# Auto-detect the Kaggle input dataset containing baseline & CBAM models
INPUT_DIR = None
_SEARCH_FILE = os.path.join("runs", "detect", "yolo11m_baseline", "weights", "best.pt")
print("Searching for pre-trained models in /kaggle/input/...")

for root, dirs, files in os.walk("/kaggle/input"):
    candidate = os.path.join(root, _SEARCH_FILE)
    if os.path.isfile(candidate):
        INPUT_DIR = root
        print(f"  ✓ Found dataset at: {INPUT_DIR}")
        break

if INPUT_DIR is None:
    print("  ❌ Could not auto-detect! Listing /kaggle/input/ contents:")
    for root, dirs, files in os.walk("/kaggle/input"):
        level = root.replace("/kaggle/input", "").count(os.sep)
        if level < 4:
            indent = "     " + "  " * level
            print(f"{indent}📁 {os.path.basename(root)}/")
            for f in files[:5]:
                print(f"{indent}  📄 {f}")
    raise FileNotFoundError("Pre-trained models not found! Check your Kaggle input dataset.")

BASELINE_BEST = f"{INPUT_DIR}/runs/detect/yolo11m_baseline/weights/best.pt"
CBAM_BEST = f"{INPUT_DIR}/runs/detect/yolo11m_cbam/weights/best.pt"
BASELINE_RESULTS_CSV = f"{INPUT_DIR}/runs/detect/yolo11m_baseline/results.csv"
CBAM_RESULTS_CSV = f"{INPUT_DIR}/runs/detect/yolo11m_cbam/results.csv"

# Validate
for path, label in [(BASELINE_BEST, "Baseline best.pt"), (CBAM_BEST, "CBAM best.pt"),
                     (BASELINE_RESULTS_CSV, "Baseline results.csv"), (CBAM_RESULTS_CSV, "CBAM results.csv")]:
    if os.path.exists(path):
        print(f"  ✓ {label}")
    else:
        print(f"  ❌ MISSING: {path}")
# ===================================================================


# ============================================================================
# CELL 3: Dataset Configuration
# ============================================================================
dataset_yaml_content = """train: /kaggle/input/c2a-dataset/C2A_Dataset/new_dataset3/train/images
val: /kaggle/input/c2a-dataset/C2A_Dataset/new_dataset3/val/images
test: /kaggle/input/c2a-dataset/C2A_Dataset/new_dataset3/test/images

nc: 1
names: ['person']
"""
with open("c2a.yaml", "w") as f:
    f.write(dataset_yaml_content)
print("✓ Dataset config saved")


# ============================================================================
# CELL 4: Define CBAM Module (Lazy Init - from previous study)
# ============================================================================
import torch
import torch.nn as nn

cbam_code = '''
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """Channel Attention Module"""
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
    """Spatial Attention Module"""
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
    """
    CBAM: Convolutional Block Attention Module with LAZY INITIALIZATION
    Auto-detects input channels from tensor at runtime.
    Handles any arg format from Ultralytics parse_model.
    """
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
            self.reduction = args[2] if isinstance(args[2], int) else 16
            self.kernel_size = args[3] if isinstance(args[3], int) else 7

        self.reduction = kwargs.get("reduction", self.reduction)
        self.kernel_size = kwargs.get("kernel_size", self.kernel_size)
        if self.kernel_size not in (3, 7):
            self.kernel_size = 7

        self._initialized = False
        self.channel_attention = None
        self.spatial_attention = None
        self._channels = None

    def _lazy_init(self, channels, device, dtype):
        self._channels = channels
        self.channel_attention = ChannelAttention(channels, self.reduction)
        self.spatial_attention = SpatialAttention(self.kernel_size)
        self.channel_attention = self.channel_attention.to(device=device, dtype=dtype)
        self.spatial_attention = self.spatial_attention.to(device=device, dtype=dtype)
        self._initialized = True

    def forward(self, x):
        if not self._initialized:
            self._lazy_init(x.size(1), x.device, x.dtype)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
'''

# Save module file for import
with open('/kaggle/working/cbam_module.py', 'w') as f:
    f.write(cbam_code)

# Define in current namespace
exec(cbam_code)

# Test CBAM
print("Testing CBAM...")
cbam_test = CBAM(16, 7)
x = torch.randn(2, 512, 20, 20)
out = cbam_test(x)
assert out.shape == x.shape
print(f"  ✓ CBAM test passed: {x.shape} → {out.shape}")


# ============================================================================
# CELL 5: Register CBAM in Ultralytics
# ============================================================================
if '/kaggle/working' not in sys.path:
    sys.path.insert(0, '/kaggle/working')

from cbam_module import CBAM, ChannelAttention, SpatialAttention
import ultralytics.nn.modules as modules
import ultralytics.nn.tasks as tasks

modules.CBAM = CBAM
modules.ChannelAttention = ChannelAttention
modules.SpatialAttention = SpatialAttention
tasks.CBAM = CBAM
tasks.ChannelAttention = ChannelAttention
tasks.SpatialAttention = SpatialAttention

assert "CBAM" in dir(modules) and "CBAM" in dir(tasks)
print("✓ CBAM registered in ultralytics")


# ============================================================================
# CELL 6: Extract Base YAML & Generate CBAM-only and CBAM+P2 YAMLs
# ============================================================================
from ultralytics import YOLO

# Only need the CBAM+P2 YAML (baseline and CBAM-only are pre-trained)
print("✓ Skipping CBAM-only YAML generation (using pre-trained model)")

# --- YAML 2: CBAM + P2 Head (CBAM in backbone + 4-scale head) ---
cbam_p2_yaml = """# YOLOv11m + CBAM + P2 Extra Detection Head
# CBAM replaces C2PSA in backbone + P2/4 extra detection scale
# Based on: Ultralytics yolo11-p2.yaml (PR #16558) + CBAM (Woo et al., 2018)

nc: 1
scales:
  m: [0.50, 1.00, 512]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]          # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]         # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]  # 2
  - [-1, 1, Conv, [256, 3, 2]]         # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]  # 4
  - [-1, 1, Conv, [512, 3, 2]]         # 5-P4/16
  - [-1, 2, C3k2, [512, True]]         # 6
  - [-1, 1, Conv, [1024, 3, 2]]        # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]        # 8
  - [-1, 1, SPPF, [1024, 5]]           # 9
  - [-1, 1, CBAM, [16, 7]]             # 10 ← CBAM replaces C2PSA

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]          # 11
  - [[-1, 6], 1, Concat, [1]]                            # 12 cat P4
  - [-1, 2, C3k2, [512, False]]                          # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]          # 14
  - [[-1, 4], 1, Concat, [1]]                            # 15 cat P3
  - [-1, 2, C3k2, [256, False]]                          # 16 (P3/8)

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]          # 17 NEW
  - [[-1, 2], 1, Concat, [1]]                            # 18 cat P2
  - [-1, 2, C3k2, [128, False]]                          # 19 (P2/4-xsmall)

  - [-1, 1, Conv, [128, 3, 2]]                           # 20
  - [[-1, 16], 1, Concat, [1]]                           # 21
  - [-1, 2, C3k2, [256, False]]                          # 22 (P3/8)

  - [-1, 1, Conv, [256, 3, 2]]                           # 23
  - [[-1, 13], 1, Concat, [1]]                           # 24
  - [-1, 2, C3k2, [512, False]]                          # 25 (P4/16)

  - [-1, 1, Conv, [512, 3, 2]]                           # 26
  - [[-1, 10], 1, Concat, [1]]                           # 27 cat P5
  - [-1, 2, C3k2, [1024, True]]                          # 28 (P5/32)

  - [[19, 22, 25, 28], 1, Detect, [nc]]                  # 29 Detect(P2,P3,P4,P5)
"""

with open("yolov11m_cbam_p2head.yaml", "w") as f:
    f.write(cbam_p2_yaml)

# Verify
with open("yolov11m_cbam_p2head.yaml", "r") as f:
    p2_cfg = yaml.safe_load(f)
detect = p2_cfg["head"][-1]
assert len(detect[0]) == 4, f"Expected 4 scales, got {len(detect[0])}"
# Verify CBAM is in backbone
has_cbam = any(layer[2] == "CBAM" for layer in p2_cfg["backbone"] if len(layer) >= 3)
assert has_cbam, "CBAM not found in backbone!"
print(f"✓ CBAM+P2 YAML: CBAM in backbone + {len(detect[0])}-scale detection")


# ============================================================================
# CELL 7: Load Pre-trained Baseline & CBAM (SKIP TRAINING)
# ============================================================================
from ultralytics import YOLO
import torch

print("=" * 80)
print("PHASE 1/3: LOADING PRE-TRAINED BASELINE (from uploaded input)")
print("=" * 80)
assert os.path.exists(BASELINE_BEST), f"Baseline model not found: {BASELINE_BEST}"
print(f"✓ Baseline loaded from: {BASELINE_BEST}")

print("\n" + "=" * 80)
print("PHASE 2/3: LOADING PRE-TRAINED CBAM (from uploaded input)")
print("=" * 80)
assert os.path.exists(CBAM_BEST), f"CBAM model not found: {CBAM_BEST}"
print(f"✓ CBAM loaded from: {CBAM_BEST}")


# ============================================================================
# CELL 9: Train CBAM + P2 Head YOLOv11m
# ============================================================================
from ultralytics import YOLO

print("\n" + "=" * 80)
print("PHASE 3/3: TRAINING CBAM + P2-HEAD YOLOv11m")
print("=" * 80)

# P2 head uses more memory (stride-4 feature maps are 4x larger spatially)
# batch=8 is safe for T4 (14.6GB), batch=16 will OOM
p2_batch = 8 if gpu_mem >= 14 else 4

if RESUME_TRAINING and RESUME_CBAM_P2_PT:
    print(f"⚡ RESUMING CBAM+P2 from: {RESUME_CBAM_P2_PT}")
    cbam_p2_model = YOLO(RESUME_CBAM_P2_PT)
    cbam_p2_model.train(resume=True)
else:
    cbam_p2_model = YOLO("yolov11m_cbam_p2head.yaml")
    cbam_p2_model.load("yolo11m.pt")
    print("✓ CBAM+P2 architecture loaded + pretrained weights")
    cbam_p2_model.info()

    cbam_p2_model.train(
        data="c2a.yaml", epochs=NUM_EPOCHS, imgsz=640, batch=p2_batch,
        device=0, name="yolo11m_cbam_p2head", patience=PATIENCE,
        save=True, save_period=SAVE_PERIOD, verbose=True, plots=True,
        fraction=TRAIN_FRACTION, amp=True, cache=False, workers=4, exist_ok=True,
    )

print("✓ CBAM+P2 training complete")
del cbam_p2_model; torch.cuda.empty_cache(); gc.collect()


# ============================================================================
# CELL 10: Output Setup & Training Curves with Total Loss
# ============================================================================
import os, pandas as pd, matplotlib.pyplot as plt, numpy as np

BASE_DIR = "/kaggle/working"
EXCEL_DIR = f"{BASE_DIR}/excel_reports"
PLOT_DIR = f"{BASE_DIR}/plots"
REPORT_DIR = f"{BASE_DIR}/benchmark_reports"
for d in [EXCEL_DIR, PLOT_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# Baseline & CBAM results from uploaded input, CBAM+P2 from local training
BASELINE_RUN = f"{INPUT_DIR}/runs/detect/yolo11m_baseline"
CBAM_RUN = f"{INPUT_DIR}/runs/detect/yolo11m_cbam"
CBAM_P2_RUN = "runs/detect/yolo11m_cbam_p2head"

# Load training CSVs
runs = {
    "baseline": pd.read_csv(BASELINE_RESULTS_CSV),
    "cbam": pd.read_csv(CBAM_RESULTS_CSV),
    "cbam_p2": pd.read_csv(f"{CBAM_P2_RUN}/results.csv"),
}

for tag, df in runs.items():
    df.columns = df.columns.str.strip()
    # Add total losses
    train_loss_cols = [c for c in df.columns if "train" in c and "loss" in c]
    val_loss_cols = [c for c in df.columns if "val" in c and "loss" in c]
    df["train/total_loss"] = df[train_loss_cols].sum(axis=1)
    df["val/total_loss"] = df[val_loss_cols].sum(axis=1)
    df.to_excel(f"{EXCEL_DIR}/yolo11m_{tag}_training.xlsx", index=False)
    print(f"  {tag}: {len(df)} epochs, cols: {train_loss_cols}")

# --- Loss plots ---
def plot_losses(df, tag):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    ax = axes[0]
    for k in ["box", "cls", "dfl"]:
        tc, vc = f"train/{k}_loss", f"val/{k}_loss"
        if tc in df.columns:
            ax.plot(df["epoch"], df[tc], label=f"Train {k}", alpha=0.7)
            ax.plot(df["epoch"], df[vc], label=f"Val {k}", linestyle='--', alpha=0.7)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title(f"{tag} - Individual Losses")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(df["epoch"], df["train/total_loss"], label="Train Total", color='blue', linewidth=2)
    ax.plot(df["epoch"], df["val/total_loss"], label="Val Total", color='red', linewidth=2, linestyle='--')
    ax.set_xlabel("Epoch"); ax.set_ylabel("Total Loss"); ax.set_title(f"{tag} - Total Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/{tag}_losses.png", dpi=300); plt.close()

for tag, df in runs.items():
    plot_losses(df, f"yolo11m_{tag}")

# 3-way total loss overlay
plt.figure(figsize=(14, 6))
colors = {'baseline': '#2196F3', 'cbam': '#4CAF50', 'cbam_p2': '#FF5722'}
for tag, df in runs.items():
    plt.plot(df["epoch"], df["val/total_loss"], label=f"{tag} Val Total",
             linewidth=2, color=colors[tag])
plt.xlabel("Epoch"); plt.ylabel("Total Loss"); plt.title("Total Loss: 3-Way Comparison")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/total_loss_3way_comparison.png", dpi=300); plt.close()

# Metrics plots
def plot_metrics(df, tag):
    plt.figure(figsize=(14, 6))
    for col in ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
        if col in df.columns:
            plt.plot(df["epoch"], df[col], label=col.replace("metrics/","").replace("(B)",""),
                     marker='o', markersize=3)
    plt.xlabel("Epoch"); plt.ylabel("Value"); plt.title(f"{tag} - Validation Metrics")
    plt.legend(); plt.grid(True, alpha=0.3); plt.ylim([0, 1.05]); plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{tag}_metrics.png", dpi=300); plt.close()

for tag, df in runs.items():
    plot_metrics(df, f"yolo11m_{tag}")

print("✓ Training curves with total loss exported")


# ============================================================================
# CELL 11: Model Complexity Comparison
# ============================================================================
from ultralytics import YOLO

models_loaded = {
    "Baseline": YOLO(BASELINE_BEST),
    "CBAM": YOLO(CBAM_BEST),
    "CBAM+P2": YOLO(f"{CBAM_P2_RUN}/weights/best.pt"),
}

def get_model_info(model, name):
    n_params = sum(p.numel() for p in model.model.parameters())
    n_layers = len(list(model.model.modules()))
    try:
        from thop import profile
        dummy = torch.randn(1, 3, 640, 640).to(next(model.model.parameters()).device)
        flops, _ = profile(model.model, inputs=(dummy,), verbose=False)
        gflops = flops / 1e9
    except Exception:
        gflops = 0.0
    return {"Model": name, "Parameters": n_params, "Layers": n_layers, "GFLOPs": round(gflops, 2)}

complexity_data = [get_model_info(m, n) for n, m in models_loaded.items()]
complexity_df = pd.DataFrame(complexity_data)
base_params = complexity_df.iloc[0]["Parameters"]
complexity_df["Δ Params (%)"] = ((complexity_df["Parameters"] / base_params) - 1) * 100
complexity_df.to_excel(f"{EXCEL_DIR}/model_complexity_3way.xlsx", index=False)
print(complexity_df.to_string(index=False))


# ============================================================================
# CELL 12: Official Ultralytics Val Metrics
# ============================================================================
print("\n" + "=" * 80)
print("OFFICIAL VALIDATION METRICS")
print("=" * 80)

def run_official_val(model, model_name):
    metrics = model.val(data="c2a.yaml", split="test", verbose=False)
    result = {
        "Model": model_name,
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "Precision": float(metrics.box.mp),
        "Recall": float(metrics.box.mr),
    }
    print(f"  {model_name}: mAP50={result['mAP50']:.4f} mAP50-95={result['mAP50-95']:.4f}")
    return result

official_results = {n: run_official_val(m, n) for n, m in models_loaded.items()}
official_df = pd.DataFrame(official_results.values())
official_df.to_excel(f"{EXCEL_DIR}/official_val_metrics_3way.xlsx", index=False)


# ============================================================================
# CELL 13: Helper Functions
# ============================================================================
import cv2
from ultralytics.utils.metrics import box_iou

def get_image_list(img_dir):
    return sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])

def parse_yolo_label(label_path, img_w, img_h):
    if not os.path.exists(label_path):
        return torch.empty((0, 4), dtype=torch.float32)
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            _, xc, yc, w, h = map(float, parts[:5])
            xc *= img_w; yc *= img_h; w *= img_w; h *= img_h
            boxes.append([xc-w/2, yc-h/2, xc+w/2, yc+h/2])
    return torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)

def match_predictions_to_gt(pred_boxes, gt_boxes, iou_thresh=0.5):
    if len(pred_boxes) == 0: return 0, 0, len(gt_boxes)
    if len(gt_boxes) == 0: return 0, len(pred_boxes), 0
    pred_boxes, gt_boxes = pred_boxes.float(), gt_boxes.float()
    ious = box_iou(pred_boxes, gt_boxes)
    matched_gt = set()
    tp = 0
    for pred_idx in range(len(pred_boxes)):
        if ious.shape[1] == 0: break
        max_iou, max_idx = ious[pred_idx].max(dim=0)
        if max_iou >= iou_thresh and max_idx.item() not in matched_gt:
            tp += 1; matched_gt.add(max_idx.item())
    return tp, len(pred_boxes) - tp, len(gt_boxes) - tp

def categorize_by_size(boxes):
    if len(boxes) == 0:
        return {k: torch.zeros(0, dtype=torch.bool) for k in
                ['very_tiny', 'tiny', 'small', 'medium', 'large']}
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return {
        'very_tiny': areas < 8**2,
        'tiny': (areas >= 8**2) & (areas < 16**2),
        'small': (areas >= 16**2) & (areas < 32**2),
        'medium': (areas >= 32**2) & (areas < 96**2),
        'large': areas >= 96**2
    }

print("✓ Helper functions loaded")


# ============================================================================
# CELL 14: Comprehensive Evaluation Function
# ============================================================================
from tqdm import tqdm

def evaluate_model_comprehensive(model, model_name, img_dir, lbl_dir, split_name="test",
                                  n_images=None, conf=0.25):
    print(f"\n{'='*80}\nEVALUATING: {model_name} on {split_name.upper()}\n{'='*80}")
    image_list = get_image_list(img_dir)
    if n_images:
        image_list = image_list[:min(n_images, len(image_list))]

    results = []
    inference_times = []
    total_tp, total_fp, total_fn = 0, 0, 0
    size_stats = {k: {'tp': 0, 'fn': 0, 'gt_count': 0}
                  for k in ['very_tiny', 'tiny', 'small', 'medium', 'large']}

    for img_file in tqdm(image_list, desc=f"{model_name}-{split_name}"):
        img_path = f"{img_dir}/{img_file}"
        lbl_path = f"{lbl_dir}/{img_file.rsplit('.', 1)[0]}.txt"
        img = cv2.imread(img_path)
        if img is None: continue
        img_h, img_w = img.shape[:2]
        gt_boxes = parse_yolo_label(lbl_path, img_w, img_h)

        start = time.time()
        preds = model.predict(img_path, conf=conf, verbose=False)
        infer_ms = (time.time() - start) * 1000
        inference_times.append(infer_ms)

        if len(preds[0].boxes) > 0:
            pred_boxes = preds[0].boxes.xyxy.cpu().float()
            avg_conf = float(preds[0].boxes.conf.cpu().float().mean())
        else:
            pred_boxes = torch.empty((0, 4), dtype=torch.float32)
            avg_conf = 0.0

        tp, fp, fn = match_predictions_to_gt(pred_boxes, gt_boxes, 0.5)
        total_tp += tp; total_fp += fp; total_fn += fn

        if len(gt_boxes) > 0:
            for size_cat, mask in categorize_by_size(gt_boxes).items():
                n_gt = int(mask.sum())
                size_stats[size_cat]['gt_count'] += n_gt
                if n_gt > 0:
                    size_tp, _, size_fn = match_predictions_to_gt(pred_boxes, gt_boxes[mask], 0.5)
                    size_stats[size_cat]['tp'] += size_tp
                    size_stats[size_cat]['fn'] += size_fn

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        f2 = 5 * precision * recall / (4 * precision + recall + 1e-9)

        results.append({
            'Image': img_file, 'GT_Boxes': len(gt_boxes), 'Pred_Boxes': len(pred_boxes),
            'TP': tp, 'FP': fp, 'FN': fn, 'Precision': precision, 'Recall': recall,
            'F1': f1, 'F2': f2, 'Avg_Confidence': avg_conf, 'Inference_ms': infer_ms
        })

    df = pd.DataFrame(results)
    df.to_excel(f"{EXCEL_DIR}/{model_name}_{split_name}_detailed.xlsx", index=False)

    op = total_tp / (total_tp + total_fp + 1e-9)
    orr = total_tp / (total_tp + total_fn + 1e-9)
    of1 = 2*op*orr/(op+orr+1e-9)
    of2 = 5*op*orr/(4*op+orr+1e-9)

    size_recalls = {}
    for cat, stats in size_stats.items():
        total = stats['tp'] + stats['fn']
        size_recalls[f"{cat}_recall"] = stats['tp'] / total if total > 0 else 0.0
        size_recalls[f"{cat}_gt_count"] = stats['gt_count']

    summary = {
        'Model': model_name, 'Split': split_name, 'Images': len(image_list),
        'Total_GT': total_tp + total_fn, 'Total_Pred': total_tp + total_fp,
        'TP': total_tp, 'FP': total_fp, 'FN': total_fn,
        'Precision': op, 'Recall': orr, 'F1': of1, 'F2': of2,
        **size_recalls,
        'Avg_Inference_ms': np.mean(inference_times),
        'Std_Inference_ms': np.std(inference_times),
        'P95_Inference_ms': np.percentile(inference_times, 95),
    }

    print(f"SUMMARY: P={op:.4f} R={orr:.4f} F1={of1:.4f}")
    print(f"  Very Tiny(<8²): {size_recalls.get('very_tiny_recall',0):.4f} "
          f"(n={size_recalls.get('very_tiny_gt_count',0)})")
    return df, summary, inference_times

print("✓ Evaluation function loaded")


# ============================================================================
# CELL 15: Visualization & Advanced Metrics
# ============================================================================

def visualize_predictions(model, img_dir, lbl_dir, model_name, split="test", n_images=15, conf=0.25):
    images = get_image_list(img_dir)[:n_images]
    cols = 5; rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    for i, img_file in enumerate(images):
        img_path = f"{img_dir}/{img_file}"
        lbl_path = f"{lbl_dir}/{img_file.rsplit('.', 1)[0]}.txt"
        gt_count = len(open(lbl_path).readlines()) if os.path.exists(lbl_path) else 0
        preds = model.predict(img_path, conf=conf, verbose=False)
        axes[i].imshow(cv2.cvtColor(preds[0].plot(), cv2.COLOR_BGR2RGB))
        axes[i].axis("off")
        pc = len(preds[0].boxes)
        c = 'green' if pc == gt_count else 'orange' if abs(pc-gt_count) <= 1 else 'red'
        axes[i].set_title(f"GT:{gt_count}|Pred:{pc}", fontsize=10, color=c, fontweight='bold')
    for j in range(len(images), len(axes)): axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{model_name}_{split}_predictions.png", dpi=150, bbox_inches='tight')
    plt.close()

def analyze_confidence_calibration(results_df, model_name, split_name, n_bins=10):
    dfp = results_df[results_df['Pred_Boxes'] > 0].copy()
    if len(dfp) == 0: return None, None
    bins = np.linspace(0, 1, n_bins + 1); cal_data = []
    for i in range(n_bins):
        m = (dfp['Avg_Confidence'] >= bins[i]) & (dfp['Avg_Confidence'] < bins[i+1])
        if m.sum() > 0:
            cal_data.append({'Bin': f"{bins[i]:.2f}-{bins[i+1]:.2f}",
                'Avg_Conf': dfp.loc[m, 'Avg_Confidence'].mean(),
                'Avg_Prec': dfp.loc[m, 'Precision'].mean(),
                'Count': int(m.sum()),
                'Gap': abs(dfp.loc[m,'Avg_Confidence'].mean() - dfp.loc[m,'Precision'].mean())})
    if not cal_data: return None, None
    cdf = pd.DataFrame(cal_data)
    ece = float(np.average(cdf['Gap'], weights=cdf['Count']))
    cdf.to_excel(f"{EXCEL_DIR}/{model_name}_{split_name}_calibration.xlsx", index=False)
    plt.figure(figsize=(10, 6))
    plt.plot([0,1],[0,1],'k--',label='Perfect')
    plt.scatter(cdf['Avg_Conf'], cdf['Avg_Prec'], s=cdf['Count']*10, alpha=0.6)
    plt.xlabel('Confidence'); plt.ylabel('Precision')
    plt.title(f'{model_name} Calibration (ECE={ece:.4f})')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{model_name}_{split_name}_calibration.png", dpi=150); plt.close()
    return cdf, ece

def analyze_failure_modes(results_df, model_name, split_name):
    dangerous = results_df[(results_df['Avg_Confidence'] > 0.7) & (results_df['Recall'] < 0.5)]
    high_fn = results_df[results_df['FN'] > 3]
    high_fp = results_df[results_df['FP'] > 3]
    if len(dangerous) > 0:
        dangerous.to_excel(f"{EXCEL_DIR}/{model_name}_{split_name}_dangerous.xlsx", index=False)
    print(f"  Dangerous:{len(dangerous)} HighFN:{len(high_fn)} HighFP:{len(high_fp)}")
    return {'Dangerous': len(dangerous), 'High_FN': len(high_fn), 'High_FP': len(high_fp)}

def benchmark_speed(model, sample_img, model_name):
    results = []
    for sz in [320, 480, 640, 800]:
        # Warmup
        _ = model.predict(sample_img, imgsz=sz, verbose=False)
        actual_times = []
        for _ in range(10):
            s = time.time()
            _ = model.predict(sample_img, imgsz=sz, verbose=False)
            actual_times.append((time.time()-s)*1000)
        results.append({'Resolution': sz, 'Avg_ms': np.mean(actual_times), 'FPS': 1000/np.mean(actual_times)})
    sdf = pd.DataFrame(results)
    sdf.to_excel(f"{EXCEL_DIR}/{model_name}_speed.xlsx", index=False)
    print(sdf.to_string(index=False))
    return sdf

def generate_3way_comparison(summaries, split_name):
    """Generate 3-way ablation comparison table"""
    metrics = [
        ("Precision", "Precision"), ("Recall", "Recall"), ("F1", "F1"), ("F2", "F2"),
        ("Very Tiny Recall", "very_tiny_recall"), ("Tiny Recall", "tiny_recall"),
        ("Small Recall", "small_recall"), ("Medium Recall", "medium_recall"),
        ("Avg Inference(ms)", "Avg_Inference_ms"),
    ]
    rows = []
    base = summaries[0]
    for display, key in metrics:
        row = {'Metric': display}
        for s in summaries:
            row[s['Model']] = round(float(s.get(key, 0)), 4)
        # Deltas vs baseline
        for s in summaries[1:]:
            bv, sv = float(base.get(key, 0)), float(s.get(key, 0))
            row[f"Δ {s['Model']}"] = round(sv - bv, 4)
        rows.append(row)
    cdf = pd.DataFrame(rows)
    cdf.to_excel(f"{EXCEL_DIR}/3way_comparison_{split_name}.xlsx", index=False)
    print(f"\n{'='*80}\n3-WAY COMPARISON - {split_name.upper()}\n{'='*80}")
    print(cdf.to_string(index=False))
    return cdf

def write_training_report(tag, df):
    best_idx = df["metrics/mAP50(B)"].idxmax() if "metrics/mAP50(B)" in df.columns else len(df)-1
    best = df.loc[best_idx]
    p, r = float(best.get("metrics/precision(B)", 0)), float(best.get("metrics/recall(B)", 0))
    report = f"""
{'='*80}
TRAINING SUMMARY: {tag}
{'='*80}
Best Epoch:     {int(best['epoch'])}
Precision:      {p:.4f}
Recall:         {r:.4f}
F1:             {2*p*r/(p+r+1e-9):.4f}
mAP@0.5:        {float(best.get('metrics/mAP50(B)', 0)):.4f}
mAP@0.5:0.95:   {float(best.get('metrics/mAP50-95(B)', 0)):.4f}
Total Val Loss: {float(best.get('val/total_loss', 0)):.4f}
{'='*80}
"""
    with open(f"{REPORT_DIR}/{tag}_training_summary.txt", "w") as f:
        f.write(report)

for tag, df in runs.items():
    write_training_report(f"yolo11m_{tag}", df)
print("✓ All functions loaded, training reports saved")


# ============================================================================
# CELL 16: Test Set Evaluation (3-way)
# ============================================================================
DATASET_ROOT = "/kaggle/input/c2a-dataset/C2A_Dataset/new_dataset3"
TEST_IMG_DIR = f"{DATASET_ROOT}/test/images"
TEST_LBL_DIR = f"{DATASET_ROOT}/test/labels"
VAL_IMG_DIR = f"{DATASET_ROOT}/val/images"
VAL_LBL_DIR = f"{DATASET_ROOT}/val/labels"

print("\n" + "=" * 80 + f"\nTEST SET EVALUATION ({TEST_IMAGES} images)\n" + "=" * 80)

test_summaries = []
test_dfs = {}
for name, model in models_loaded.items():
    safe = name.lower().replace("+", "_")
    df, summary, _ = evaluate_model_comprehensive(
        model, f"yolo11m_{safe}", TEST_IMG_DIR, TEST_LBL_DIR, "test", n_images=TEST_IMAGES)
    test_dfs[name] = df
    test_summaries.append(summary)
    visualize_predictions(model, TEST_IMG_DIR, TEST_LBL_DIR, f"yolo11m_{safe}", "test",
                          min(15, TEST_IMAGES))

test_comparison = generate_3way_comparison(test_summaries, "test")


# ============================================================================
# CELL 17: Validation Set Evaluation (3-way)
# ============================================================================
print("\n" + "=" * 80 + f"\nVALIDATION SET EVALUATION ({VAL_IMAGES} images)\n" + "=" * 80)

val_summaries = []
val_dfs = {}
for name, model in models_loaded.items():
    safe = name.lower().replace("+", "_")
    df, summary, _ = evaluate_model_comprehensive(
        model, f"yolo11m_{safe}", VAL_IMG_DIR, VAL_LBL_DIR, "val", n_images=VAL_IMAGES)
    val_dfs[name] = df
    val_summaries.append(summary)
    visualize_predictions(model, VAL_IMG_DIR, VAL_LBL_DIR, f"yolo11m_{safe}", "val",
                          min(15, VAL_IMAGES))

val_comparison = generate_3way_comparison(val_summaries, "val")


# ============================================================================
# CELL 18: Advanced Metrics (3-way)
# ============================================================================
print("\n" + "=" * 80 + "\nADVANCED METRICS\n" + "=" * 80)

test_images_list = get_image_list(TEST_IMG_DIR)
sample_img = f"{TEST_IMG_DIR}/{test_images_list[0]}"

ece_results = {}
for name, model in models_loaded.items():
    safe = name.lower().replace("+", "_")
    print(f"\n--- {name} ---")
    print("Calibration:")
    _, ece = analyze_confidence_calibration(test_dfs[name], f"yolo11m_{safe}", "test")
    ece_results[name] = ece
    print("Failures:")
    analyze_failure_modes(test_dfs[name], f"yolo11m_{safe}", "test")
    print("Speed:")
    benchmark_speed(model, sample_img, f"yolo11m_{safe}")


# ============================================================================
# CELL 19: Per-Size Recall Bar Chart (3-way)
# ============================================================================
size_cats = ['very_tiny', 'tiny', 'small', 'medium', 'large']
size_labels = ['Very Tiny\n(<8²px)', 'Tiny\n(8-16px)', 'Small\n(16-32px)', 'Medium\n(32-96px)', 'Large\n(>96px)']

x = np.arange(len(size_cats))
width = 0.25
colors_list = ['#2196F3', '#4CAF50', '#FF5722']

fig, ax = plt.subplots(figsize=(16, 7))
for idx, summary in enumerate(test_summaries):
    recalls = [summary.get(f"{c}_recall", 0) for c in size_cats]
    ax.bar(x + (idx - 1) * width, recalls, width, label=summary['Model'],
           color=colors_list[idx], alpha=0.8)

# Count annotations
counts = [test_summaries[0].get(f"{c}_gt_count", 0) for c in size_cats]
for i, c in enumerate(counts):
    ax.text(i, max(s.get(f"{size_cats[i]}_recall", 0) for s in test_summaries) + 0.02,
            f'n={c}', ha='center', fontsize=9, color='gray')

ax.set_xlabel('Object Size', fontsize=12); ax.set_ylabel('Recall', fontsize=12)
ax.set_title('Per-Size Recall: 3-Way Comparison (Test Set)', fontsize=14, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(size_labels)
ax.legend(fontsize=11); ax.set_ylim([0, 1.15]); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/per_size_recall_3way.png", dpi=300); plt.close()
print("✓ Per-size recall 3-way comparison saved")


# ============================================================================
# CELL 20: Master Report
# ============================================================================
print("\n" + "=" * 80 + "\nFINAL SUMMARY\n" + "=" * 80)

all_summaries_df = pd.DataFrame(test_summaries + val_summaries)
all_summaries_df.to_excel(f"{EXCEL_DIR}/complete_3way_summary.xlsx", index=False)

# Best epochs
best_epochs = {}
for tag, df in runs.items():
    best_epochs[tag] = int(df.loc[df["metrics/mAP50(B)"].idxmax(), "epoch"])

ts, vs = test_summaries, val_summaries
of = official_results

master = f"""
{'='*80}
DISASTER HUMAN DETECTION: CBAM + P2 HEAD ABLATION STUDY (3-WAY)
{'='*80}

CONFIG: Test={TEST_MODE} | Epochs={NUM_EPOCHS} | Fraction={TRAIN_FRACTION*100:.0f}% | ImgSz=640

MODEL COMPLEXITY:
  Baseline:  {complexity_data[0]['Parameters']:>12,} params | {complexity_data[0]['GFLOPs']:>6.1f} GFLOPs
  CBAM:      {complexity_data[1]['Parameters']:>12,} params | {complexity_data[1]['GFLOPs']:>6.1f} GFLOPs ({complexity_df.iloc[1]['Δ Params (%)']:+.1f}%)
  CBAM+P2:   {complexity_data[2]['Parameters']:>12,} params | {complexity_data[2]['GFLOPs']:>6.1f} GFLOPs ({complexity_df.iloc[2]['Δ Params (%)']:+.1f}%)

OFFICIAL mAP (test split):
  Baseline: mAP50={of['Baseline']['mAP50']:.4f} mAP50-95={of['Baseline']['mAP50-95']:.4f}
  CBAM:     mAP50={of['CBAM']['mAP50']:.4f} mAP50-95={of['CBAM']['mAP50-95']:.4f}
  CBAM+P2:  mAP50={of['CBAM+P2']['mAP50']:.4f} mAP50-95={of['CBAM+P2']['mAP50-95']:.4f}

TEST SET (custom eval):
  Baseline: P={ts[0]['Precision']:.4f} R={ts[0]['Recall']:.4f} F1={ts[0]['F1']:.4f}
  CBAM:     P={ts[1]['Precision']:.4f} R={ts[1]['Recall']:.4f} F1={ts[1]['F1']:.4f}
  CBAM+P2:  P={ts[2]['Precision']:.4f} R={ts[2]['Recall']:.4f} F1={ts[2]['F1']:.4f}

SMALL OBJECT RECALL (TEST):
  Very Tiny (<8²px):  Base={ts[0].get('very_tiny_recall',0):.4f} | CBAM={ts[1].get('very_tiny_recall',0):.4f} | CBAM+P2={ts[2].get('very_tiny_recall',0):.4f}
  Tiny (8-16px):      Base={ts[0].get('tiny_recall',0):.4f} | CBAM={ts[1].get('tiny_recall',0):.4f} | CBAM+P2={ts[2].get('tiny_recall',0):.4f}
  Small (16-32px):    Base={ts[0].get('small_recall',0):.4f} | CBAM={ts[1].get('small_recall',0):.4f} | CBAM+P2={ts[2].get('small_recall',0):.4f}

VALIDATION SET:
  Baseline: P={vs[0]['Precision']:.4f} R={vs[0]['Recall']:.4f} F1={vs[0]['F1']:.4f}
  CBAM:     P={vs[1]['Precision']:.4f} R={vs[1]['Recall']:.4f} F1={vs[1]['F1']:.4f}
  CBAM+P2:  P={vs[2]['Precision']:.4f} R={vs[2]['Recall']:.4f} F1={vs[2]['F1']:.4f}

CALIBRATION (ECE):
  Baseline: {ece_results.get('Baseline', 'N/A')}
  CBAM:     {ece_results.get('CBAM', 'N/A')}
  CBAM+P2:  {ece_results.get('CBAM+P2', 'N/A')}

BEST MODEL: {max(ts, key=lambda s: s.get('very_tiny_recall', 0))['Model']} (very-tiny recall priority)

OUTPUTS: {EXCEL_DIR} | {PLOT_DIR} | {REPORT_DIR}
{'='*80}
"""

with open(f"{REPORT_DIR}/MASTER_SUMMARY_CBAM_P2HEAD.txt", "w") as f:
    f.write(master)
print(master)


# ============================================================================
# CELL 21: Package Results
# ============================================================================
from IPython.display import FileLink
!zip -r /kaggle/working/cbam_p2head_results.zip {EXCEL_DIR} {PLOT_DIR} {REPORT_DIR}
FileLink("/kaggle/working/cbam_p2head_results.zip")

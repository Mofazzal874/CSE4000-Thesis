"""
================================================================================
DISASTER HUMAN DETECTION: P2 EXTRA DETECTION HEAD ABLATION STUDY
================================================================================

MODIFICATION: Adds a 4th detection scale (P2/4) to YOLO11m for improved
small-object detection (<10px humans in aerial disaster imagery).

Architecture: Standard YOLO11m predicts on P3/P4/P5 (3 scales).
This adds P2/4 scale → 160×160 feature maps → detects targets ≥4×4px.

Based on: Ultralytics yolo11-p2.yaml (PR #16558)
Reference: SED-YOLO (Shi et al., 2024) - extra small-object detection head

CHECKPOINT RESUME: Saves checkpoints every 5 epochs for Kaggle 12h limit.
To resume: Set RESUME_TRAINING = True and point to last.pt

================================================================================
"""

# ============================================================================
# CELL 1: Configuration & Dependencies
# ============================================================================

# ===================== CONTROL FLAGS =====================
TEST_MODE = True          # True = 5% data, 2 epochs (dry run)
                          # False = 100% data, 70 epochs (full run)

RESUME_TRAINING = False   # Set True to resume from checkpoint
RESUME_BASELINE_PT = ""   # Path to baseline last.pt (if resuming)
RESUME_P2HEAD_PT = ""     # Path to p2head last.pt (if resuming)
# =========================================================

!pip uninstall ultralytics -y -q 2>/dev/null
!pip install -q -U ultralytics timm thop "pandas<3.0" "matplotlib<3.10" openpyxl scikit-learn
print("✓ Dependencies installed")


# ============================================================================
# CELL 2: Derive Training Config from TEST_MODE
# ============================================================================
import os
import sys
import time
import yaml
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

if TEST_MODE:
    TRAIN_FRACTION = 0.05
    NUM_EPOCHS = 2
    PATIENCE = 2
    TEST_IMAGES = 10
    VAL_IMAGES = 20
    SAVE_PERIOD = 1
    print("⚠ TEST MODE: 5% data, 2 epochs, minimal evaluation")
else:
    TRAIN_FRACTION = 1.0
    NUM_EPOCHS = 70
    PATIENCE = 10
    TEST_IMAGES = 30
    VAL_IMAGES = 100
    SAVE_PERIOD = 5    # checkpoint every 5 epochs for Kaggle 12h limit
    print("🚀 FULL MODE: 100% data, 70 epochs")

print(f"  Epochs: {NUM_EPOCHS} | Fraction: {TRAIN_FRACTION} | Patience: {PATIENCE}")
print(f"  Checkpoint save period: every {SAVE_PERIOD} epochs")

# GPU info
num_gpus = torch.cuda.device_count()
print(f"\nGPUs: {num_gpus}")
for i in range(num_gpus):
    gpu = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {gpu.name} ({gpu.total_memory / 1024**3:.1f} GB)")


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
# CELL 4: Extract & Generate P2-Head YAML
# ============================================================================
from ultralytics import YOLO

# Load pretrained model and extract base YAML
model = YOLO("yolo11m.pt")
model.info()

base_cfg = model.model.yaml
with open("yolov11m_original.yaml", "w") as f:
    yaml.dump(base_cfg, f)
print("✓ Extracted base YOLOv11m YAML")

# --- Generate P2-Head YAML ---
# The official yolo11-p2.yaml from Ultralytics PR #16558
# Backbone is IDENTICAL to standard yolo11.yaml
# Head adds: Upsample → Concat(P2) → C3k2[128] for P2/4 scale

p2_yaml_content = """# YOLOv11m + P2 Extra Detection Head for Small Objects
# Based on: ultralytics/ultralytics PR #16558 (yolo11-p2.yaml)
# Modification: 4-scale detection (P2/P3/P4/P5) instead of 3-scale (P3/P4/P5)

nc: 1
scales:
  m: [0.50, 1.00, 512]

# Backbone (identical to standard yolo11m)
backbone:
  - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]  # 2
  - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]  # 4
  - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16
  - [-1, 2, C3k2, [512, True]]      # 6
  - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]     # 8
  - [-1, 1, SPPF, [1024, 5]]        # 9
  - [-1, 2, C2PSA, [1024]]          # 10

# Head with P2 extra detection scale
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]          # 11
  - [[-1, 6], 1, Concat, [1]]                            # 12 cat backbone P4
  - [-1, 2, C3k2, [512, False]]                          # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]          # 14
  - [[-1, 4], 1, Concat, [1]]                            # 15 cat backbone P3
  - [-1, 2, C3k2, [256, False]]                          # 16 (P3/8-small)

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]          # 17 NEW: upsample to P2
  - [[-1, 2], 1, Concat, [1]]                            # 18 NEW: cat backbone P2
  - [-1, 2, C3k2, [128, False]]                          # 19 NEW: P2/4-xsmall

  - [-1, 1, Conv, [128, 3, 2]]                           # 20
  - [[-1, 16], 1, Concat, [1]]                           # 21 cat head P3
  - [-1, 2, C3k2, [256, False]]                          # 22 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]                           # 23
  - [[-1, 13], 1, Concat, [1]]                           # 24 cat head P4
  - [-1, 2, C3k2, [512, False]]                          # 25 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]                           # 26
  - [[-1, 10], 1, Concat, [1]]                           # 27 cat head P5
  - [-1, 2, C3k2, [1024, True]]                          # 28 (P5/32-large)

  - [[19, 22, 25, 28], 1, Detect, [nc]]                  # 29 Detect(P2, P3, P4, P5)
"""

with open("yolov11m_p2head.yaml", "w") as f:
    f.write(p2_yaml_content)

print("✓ Generated P2-Head YAML (4-scale detection: P2/P3/P4/P5)")
print("  NEW layers: 17 (Upsample), 18 (Concat P2), 19 (C3k2 P2/4-xsmall)")
print("  Detect: [19, 22, 25, 28] = P2, P3, P4, P5")

# Verify YAML loads correctly
with open("yolov11m_p2head.yaml", "r") as f:
    p2_cfg = yaml.safe_load(f)
detect_layer = p2_cfg["head"][-1]
assert len(detect_layer[0]) == 4, f"Expected 4 detection scales, got {len(detect_layer[0])}"
print(f"  ✓ Detect layer verified: {detect_layer[0]} → 4 scales")


# ============================================================================
# CELL 5: Train Baseline YOLOv11m
# ============================================================================
from ultralytics import YOLO
import torch

print("=" * 80)
print("PHASE 1: TRAINING BASELINE YOLOv11m (3-scale P3/P4/P5)")
print("=" * 80)

# Determine batch size based on GPU memory
gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
baseline_batch = 16 if gpu_mem >= 15 else 8
print(f"GPU memory: {gpu_mem:.1f} GB → baseline batch={baseline_batch}")

if RESUME_TRAINING and RESUME_BASELINE_PT:
    print(f"⚡ RESUMING baseline from: {RESUME_BASELINE_PT}")
    baseline = YOLO(RESUME_BASELINE_PT)
    baseline.train(resume=True)
else:
    baseline = YOLO("yolo11m.pt")

    # Multi-GPU for baseline (no custom modules)
    if num_gpus >= 2:
        device_config = [0, 1]
        baseline_batch = baseline_batch * 2
        print(f"✓ Multi-GPU: {device_config}, batch={baseline_batch}")
    else:
        device_config = 0
        print(f"✓ Single GPU, batch={baseline_batch}")

    baseline.train(
        data="c2a.yaml",
        epochs=NUM_EPOCHS,
        imgsz=640,
        batch=baseline_batch,
        device=device_config,
        name="yolo11m_baseline",
        patience=PATIENCE,
        save=True,
        save_period=SAVE_PERIOD,  # Checkpoint for Kaggle 12h limit
        verbose=True,
        plots=True,
        fraction=TRAIN_FRACTION,
        amp=True,           # Mixed precision to save memory
        cache=False,         # Don't cache to avoid OOM on large datasets
        workers=4,
        exist_ok=True,       # Allow resuming into same directory
    )

print("✓ Baseline training complete")

# Save baseline best.pt to a reusable location (no retraining needed later)
baseline_best_src = "runs/detect/yolo11m_baseline/weights/best.pt"
baseline_best_dst = "/kaggle/working/yolo11m_baseline_best.pt"
if os.path.exists(baseline_best_src):
    shutil.copy2(baseline_best_src, baseline_best_dst)
    print(f"✓ Baseline model saved to: {baseline_best_dst}")
    print(f"  → Reuse later: YOLO('{baseline_best_dst}')")

# Free GPU memory before next training
del baseline
torch.cuda.empty_cache()
import gc; gc.collect()
print("✓ GPU memory cleared")


# ============================================================================
# CELL 6: Train P2-Head YOLOv11m
# ============================================================================
from ultralytics import YOLO
import torch

print("\n" + "=" * 80)
print("PHASE 2: TRAINING P2-HEAD YOLOv11m (4-scale P2/P3/P4/P5)")
print("=" * 80)

# P2 head uses more memory → reduce batch size
p2_batch = 8 if gpu_mem >= 15 else 4
print(f"P2 head batch size: {p2_batch} (reduced due to extra P2 scale)")

if RESUME_TRAINING and RESUME_P2HEAD_PT:
    print(f"⚡ RESUMING P2-head from: {RESUME_P2HEAD_PT}")
    p2_model = YOLO(RESUME_P2HEAD_PT)
    p2_model.train(resume=True)
else:
    p2_model = YOLO("yolov11m_p2head.yaml")

    # Transfer pretrained weights (matching layers only)
    p2_model.load("yolo11m.pt")
    print("✓ P2-Head architecture loaded")
    print("✓ Pretrained weights transferred (matching layers)")

    # Print model comparison
    p2_model.info()

    # FORCE SINGLE GPU for P2 model (safer with modified architecture)
    device_config = 0

    p2_model.train(
        data="c2a.yaml",
        epochs=NUM_EPOCHS,
        imgsz=640,
        batch=p2_batch,
        device=device_config,
        name="yolo11m_p2head",
        patience=PATIENCE,
        save=True,
        save_period=SAVE_PERIOD,
        verbose=True,
        plots=True,
        fraction=TRAIN_FRACTION,
        amp=True,
        cache=False,
        workers=4,
        exist_ok=True,
    )

print("✓ P2-Head training complete")

del p2_model
torch.cuda.empty_cache()
gc.collect()
print("✓ GPU memory cleared")


# ============================================================================
# CELL 7: Output Directories Setup
# ============================================================================
import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "/kaggle/working"
EXCEL_DIR = f"{BASE_DIR}/excel_reports"
PLOT_DIR = f"{BASE_DIR}/plots"
REPORT_DIR = f"{BASE_DIR}/benchmark_reports"

for d in [EXCEL_DIR, PLOT_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

BASELINE_RUN = "runs/detect/yolo11m_baseline"
P2HEAD_RUN = "runs/detect/yolo11m_p2head"

print("✓ Output directories ready")


# ============================================================================
# CELL 8: Training Curves with TOTAL LOSS
# ============================================================================
baseline_df = pd.read_csv(f"{BASELINE_RUN}/results.csv")
p2head_df = pd.read_csv(f"{P2HEAD_RUN}/results.csv")

# Strip whitespace from column names (Ultralytics quirk)
baseline_df.columns = baseline_df.columns.str.strip()
p2head_df.columns = p2head_df.columns.str.strip()

# --- ADD TOTAL LOSS (box + cls + dfl) ---
for df, tag in [(baseline_df, "baseline"), (p2head_df, "p2head")]:
    loss_cols_train = [c for c in df.columns if "train" in c and "loss" in c]
    loss_cols_val = [c for c in df.columns if "val" in c and "loss" in c]
    df["train/total_loss"] = df[loss_cols_train].sum(axis=1)
    df["val/total_loss"] = df[loss_cols_val].sum(axis=1)
    print(f"  {tag} train loss columns: {loss_cols_train}")

# Save with total loss added
baseline_df.to_excel(f"{EXCEL_DIR}/yolo11m_baseline_training.xlsx", index=False)
p2head_df.to_excel(f"{EXCEL_DIR}/yolo11m_p2head_training.xlsx", index=False)

def plot_losses(df, tag):
    """Plot individual + total losses"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Individual losses
    ax = axes[0]
    for k in ["box", "cls", "dfl"]:
        tc = f"train/{k}_loss"
        vc = f"val/{k}_loss"
        if tc in df.columns:
            ax.plot(df["epoch"], df[tc], label=f"Train {k}", alpha=0.7)
            ax.plot(df["epoch"], df[vc], label=f"Val {k}", linestyle='--', alpha=0.7)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title(f"{tag} - Individual Losses")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Total loss
    ax = axes[1]
    ax.plot(df["epoch"], df["train/total_loss"], label="Train Total", color='blue', linewidth=2)
    ax.plot(df["epoch"], df["val/total_loss"], label="Val Total", color='red', linewidth=2, linestyle='--')
    ax.set_xlabel("Epoch"); ax.set_ylabel("Total Loss"); ax.set_title(f"{tag} - Total Loss (box+cls+dfl)")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{tag}_losses.png", dpi=300)
    plt.close()

plot_losses(baseline_df, "yolo11m_baseline")
plot_losses(p2head_df, "yolo11m_p2head")

# --- Overlay total loss comparison ---
plt.figure(figsize=(12, 6))
plt.plot(baseline_df["epoch"], baseline_df["val/total_loss"], label="Baseline Val Total", linewidth=2)
plt.plot(p2head_df["epoch"], p2head_df["val/total_loss"], label="P2-Head Val Total", linewidth=2)
plt.xlabel("Epoch"); plt.ylabel("Total Loss"); plt.title("Total Loss Comparison: Baseline vs P2-Head")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/total_loss_comparison.png", dpi=300)
plt.close()

def plot_metrics(df, tag):
    plt.figure(figsize=(14, 6))
    for col in ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
        if col in df.columns:
            label = col.replace("metrics/", "").replace("(B)", "")
            plt.plot(df["epoch"], df[col], label=label, marker='o', markersize=3)
    plt.xlabel("Epoch"); plt.ylabel("Value"); plt.title(f"{tag} - Validation Metrics")
    plt.legend(); plt.grid(True, alpha=0.3); plt.ylim([0, 1.05]); plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{tag}_metrics.png", dpi=300)
    plt.close()

plot_metrics(baseline_df, "yolo11m_baseline")
plot_metrics(p2head_df, "yolo11m_p2head")

print("✓ Training curves (with total loss) exported")


# ============================================================================
# CELL 9: Model Complexity Comparison
# ============================================================================
from ultralytics import YOLO

baseline_model = YOLO(f"{BASELINE_RUN}/weights/best.pt")
p2head_model = YOLO(f"{P2HEAD_RUN}/weights/best.pt")

def get_model_info(model, name):
    """Extract model complexity metrics"""
    n_params = sum(p.numel() for p in model.model.parameters())
    n_layers = len(list(model.model.modules()))
    # GFLOPs from model info
    try:
        from thop import profile
        dummy = torch.randn(1, 3, 640, 640).to(next(model.model.parameters()).device)
        flops, _ = profile(model.model, inputs=(dummy,), verbose=False)
        gflops = flops / 1e9
    except Exception:
        gflops = 0.0
    return {"Model": name, "Parameters": n_params, "Layers": n_layers, "GFLOPs": round(gflops, 2)}

baseline_info = get_model_info(baseline_model, "YOLOv11m (Baseline)")
p2head_info = get_model_info(p2head_model, "YOLOv11m + P2-Head")

complexity_df = pd.DataFrame([baseline_info, p2head_info])
complexity_df["Δ Params"] = complexity_df["Parameters"] - complexity_df["Parameters"].iloc[0]
complexity_df["Δ Params (%)"] = ((complexity_df["Parameters"] / complexity_df["Parameters"].iloc[0]) - 1) * 100
complexity_df.to_excel(f"{EXCEL_DIR}/model_complexity.xlsx", index=False)
print(complexity_df.to_string(index=False))
print("✓ Model complexity comparison saved")


# ============================================================================
# CELL 10: Ultralytics Val() for Official mAP Metrics
# ============================================================================
print("\n" + "=" * 80)
print("OFFICIAL ULTRALYTICS VALIDATION (mAP@0.5, mAP@0.5:0.95)")
print("=" * 80)

def run_official_val(model, model_name):
    """Run Ultralytics val() and extract all official metrics"""
    metrics = model.val(data="c2a.yaml", split="test", verbose=False)
    result = {
        "Model": model_name,
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "Precision": float(metrics.box.mp),
        "Recall": float(metrics.box.mr),
    }
    # Try to get per-size AP if available
    try:
        if hasattr(metrics.box, 'maps'):
            result["mAP50_per_class"] = metrics.box.maps.tolist()
    except Exception:
        pass
    print(f"  {model_name}: mAP50={result['mAP50']:.4f} mAP50-95={result['mAP50-95']:.4f}")
    return result

baseline_official = run_official_val(baseline_model, "Baseline")
p2head_official = run_official_val(p2head_model, "P2-Head")

official_df = pd.DataFrame([baseline_official, p2head_official])
official_df.to_excel(f"{EXCEL_DIR}/official_val_metrics.xlsx", index=False)
print("✓ Official validation metrics saved")


# ============================================================================
# CELL 11: Helper Functions for Custom Evaluation
# ============================================================================
import cv2
from ultralytics.utils.metrics import box_iou

def get_image_list(img_dir):
    return sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])

def parse_yolo_label(label_path, img_w, img_h):
    if not os.path.exists(label_path):
        return torch.empty((0, 4))
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, xc, yc, w, h = map(float, parts[:5])
            xc *= img_w; yc *= img_h; w *= img_w; h *= img_h
            boxes.append([xc - w/2, yc - h/2, xc + w/2, yc + h/2])
    return torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)

def match_predictions_to_gt(pred_boxes, gt_boxes, iou_thresh=0.5):
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    # Ensure float tensors for IoU computation
    pred_boxes = pred_boxes.float()
    gt_boxes = gt_boxes.float()
    ious = box_iou(pred_boxes, gt_boxes)
    matched_gt = set()
    tp = 0
    for pred_idx in range(len(pred_boxes)):
        if ious.shape[1] == 0:
            break
        max_iou, max_idx = ious[pred_idx].max(dim=0)
        if max_iou >= iou_thresh and max_idx.item() not in matched_gt:
            tp += 1
            matched_gt.add(max_idx.item())
    return tp, len(pred_boxes) - tp, len(gt_boxes) - tp

def categorize_by_size(boxes):
    """Enhanced 5-tier size categorization for sub-10px detection focus"""
    if len(boxes) == 0:
        return {k: torch.zeros(0, dtype=torch.bool) for k in
                ['very_tiny', 'tiny', 'small', 'medium', 'large']}
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return {
        'very_tiny': areas < 8**2,          # <64 px² (sub-8px, hardest)
        'tiny':      (areas >= 8**2) & (areas < 16**2),   # 8-16px
        'small':     (areas >= 16**2) & (areas < 32**2),   # COCO "small"
        'medium':    (areas >= 32**2) & (areas < 96**2),   # COCO "medium"
        'large':     areas >= 96**2                         # COCO "large"
    }

print("✓ Helper functions loaded (with 5-tier size categories)")


# ============================================================================
# CELL 12: Comprehensive Evaluation Function
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
        if img is None:
            continue
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

        # Per-size analysis
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

    overall_precision = total_tp / (total_tp + total_fp + 1e-9)
    overall_recall = total_tp / (total_tp + total_fn + 1e-9)
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall + 1e-9)
    overall_f2 = 5 * overall_precision * overall_recall / (4 * overall_precision + overall_recall + 1e-9)

    size_recalls = {}
    for cat, stats in size_stats.items():
        total = stats['tp'] + stats['fn']
        size_recalls[f"{cat}_recall"] = stats['tp'] / total if total > 0 else 0.0
        size_recalls[f"{cat}_gt_count"] = stats['gt_count']

    summary = {
        'Model': model_name, 'Split': split_name, 'Images': len(image_list),
        'Total_GT': total_tp + total_fn, 'Total_Pred': total_tp + total_fp,
        'TP': total_tp, 'FP': total_fp, 'FN': total_fn,
        'Precision': overall_precision, 'Recall': overall_recall,
        'F1': overall_f1, 'F2': overall_f2, **size_recalls,
        'Avg_Inference_ms': np.mean(inference_times),
        'Std_Inference_ms': np.std(inference_times),
        'P95_Inference_ms': np.percentile(inference_times, 95),
    }

    print(f"\nSUMMARY: P={overall_precision:.4f} R={overall_recall:.4f} F1={overall_f1:.4f}")
    print(f"  Very Tiny Recall (<8²px): {size_recalls.get('very_tiny_recall', 0):.4f} "
          f"({size_recalls.get('very_tiny_gt_count', 0)} objects)")
    print(f"  Tiny Recall (8-16px):     {size_recalls.get('tiny_recall', 0):.4f} "
          f"({size_recalls.get('tiny_gt_count', 0)} objects)")
    print(f"  Small Recall (16-32px):   {size_recalls.get('small_recall', 0):.4f} "
          f"({size_recalls.get('small_gt_count', 0)} objects)")

    return df, summary, inference_times

print("✓ Evaluation function loaded")


# ============================================================================
# CELL 13: Visualization & Advanced Metrics Functions
# ============================================================================

def visualize_predictions(model, img_dir, lbl_dir, model_name, split="test", n_images=15, conf=0.25):
    images = get_image_list(img_dir)[:n_images]
    cols = 5
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, img_file in enumerate(images):
        img_path = f"{img_dir}/{img_file}"
        lbl_path = f"{lbl_dir}/{img_file.rsplit('.', 1)[0]}.txt"
        gt_count = len(open(lbl_path).readlines()) if os.path.exists(lbl_path) else 0
        preds = model.predict(img_path, conf=conf, verbose=False)
        img_plot = preds[0].plot()
        axes[i].imshow(cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB))
        axes[i].axis("off")
        pred_count = len(preds[0].boxes)
        color = 'green' if pred_count == gt_count else 'orange' if abs(pred_count - gt_count) <= 1 else 'red'
        axes[i].set_title(f"GT:{gt_count} | Pred:{pred_count}", fontsize=10, color=color, fontweight='bold')

    for j in range(len(images), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{model_name}_{split}_predictions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Visualization: {model_name}_{split}_predictions.png")

def analyze_confidence_calibration(results_df, model_name, split_name, n_bins=10):
    df_with_preds = results_df[results_df['Pred_Boxes'] > 0].copy()
    if len(df_with_preds) == 0:
        return None, None
    bins = np.linspace(0, 1, n_bins + 1)
    calibration_data = []
    for i in range(n_bins):
        bin_mask = (df_with_preds['Avg_Confidence'] >= bins[i]) & (df_with_preds['Avg_Confidence'] < bins[i+1])
        if bin_mask.sum() > 0:
            calibration_data.append({
                'Bin': f"{bins[i]:.2f}-{bins[i+1]:.2f}",
                'Avg_Conf': df_with_preds.loc[bin_mask, 'Avg_Confidence'].mean(),
                'Avg_Prec': df_with_preds.loc[bin_mask, 'Precision'].mean(),
                'Count': int(bin_mask.sum()),
                'Gap': abs(df_with_preds.loc[bin_mask, 'Avg_Confidence'].mean() -
                          df_with_preds.loc[bin_mask, 'Precision'].mean())
            })
    if not calibration_data:
        return None, None
    cal_df = pd.DataFrame(calibration_data)
    ece = float(np.average(cal_df['Gap'], weights=cal_df['Count']))
    cal_df.to_excel(f"{EXCEL_DIR}/{model_name}_{split_name}_calibration.xlsx", index=False)
    plt.figure(figsize=(10, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
    plt.scatter(cal_df['Avg_Conf'], cal_df['Avg_Prec'], s=cal_df['Count']*10, alpha=0.6)
    plt.xlabel('Confidence'); plt.ylabel('Precision')
    plt.title(f'{model_name} Calibration (ECE={ece:.4f})')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{model_name}_{split_name}_calibration.png", dpi=150)
    plt.close()
    print(f"  ECE: {ece:.4f}")
    return cal_df, ece

def analyze_failure_modes(results_df, model_name, split_name):
    dangerous = results_df[(results_df['Avg_Confidence'] > 0.7) & (results_df['Recall'] < 0.5)]
    high_fn = results_df[results_df['FN'] > 3]
    high_fp = results_df[results_df['FP'] > 3]
    if len(dangerous) > 0:
        dangerous.to_excel(f"{EXCEL_DIR}/{model_name}_{split_name}_dangerous_errors.xlsx", index=False)
    print(f"  Dangerous errors: {len(dangerous)}, High FN: {len(high_fn)}, High FP: {len(high_fp)}")
    return {'Dangerous': len(dangerous), 'High_FN': len(high_fn), 'High_FP': len(high_fp)}

def benchmark_speed_vs_resolution(model, sample_img_path, model_name):
    results = []
    for imgsz in [320, 480, 640, 800]:
        times = []
        for i in range(10):
            start = time.time()
            _ = model.predict(sample_img_path, imgsz=imgsz, verbose=False)
            times.append((time.time() - start) * 1000)
        results.append({'Resolution': imgsz, 'Avg_ms': np.mean(times), 'FPS': 1000/np.mean(times)})
    speed_df = pd.DataFrame(results)
    speed_df.to_excel(f"{EXCEL_DIR}/{model_name}_speed_vs_resolution.xlsx", index=False)
    print(speed_df.to_string(index=False))
    return speed_df

def generate_ablation_comparison(baseline_summary, variant_summary, variant_name, split_name):
    metrics = [
        ("Precision", "Precision"), ("Recall", "Recall"), ("F1", "F1"), ("F2", "F2"),
        ("Very Tiny Recall (<8²px)", "very_tiny_recall"), ("Tiny Recall (8-16px)", "tiny_recall"),
        ("Small Recall (16-32px)", "small_recall"), ("Medium Recall", "medium_recall"),
        ("Large Recall", "large_recall"),
        ("Avg Inference (ms)", "Avg_Inference_ms"), ("P95 Inference (ms)", "P95_Inference_ms"),
    ]
    comparison_data = []
    for display_name, key in metrics:
        baseline_val = baseline_summary.get(key, 0.0)
        variant_val = variant_summary.get(key, 0.0)
        comparison_data.append({
            'Metric': display_name,
            'Baseline': round(float(baseline_val), 4),
            variant_name: round(float(variant_val), 4),
            'Δ': round(float(variant_val - baseline_val), 4),
            'Δ (%)': round(float((variant_val - baseline_val) / (baseline_val + 1e-9) * 100), 2)
        })
    comparison_df = pd.DataFrame(comparison_data)
    safe_name = variant_name.replace(" ", "_").replace("+", "_")
    comparison_df.to_excel(f"{EXCEL_DIR}/ablation_{safe_name}_vs_baseline_{split_name}.xlsx", index=False)
    print(f"\n{'='*80}\nABLATION: {variant_name} vs Baseline - {split_name.upper()}\n{'='*80}")
    print(comparison_df.to_string(index=False))
    return comparison_df

def write_training_report(tag, df):
    best_idx = df["metrics/mAP50(B)"].idxmax() if "metrics/mAP50(B)" in df.columns else len(df)-1
    best = df.loc[best_idx]
    p = float(best.get("metrics/precision(B)", 0))
    r = float(best.get("metrics/recall(B)", 0))
    f1 = 2*p*r/(p+r+1e-9)
    total_loss = float(best.get("val/total_loss", 0))
    report = f"""
{'='*80}
TRAINING SUMMARY: {tag}
{'='*80}
Best Epoch:       {int(best['epoch'])}
Precision:        {p:.4f}
Recall:           {r:.4f}
F1:               {f1:.4f}
mAP@0.5:          {float(best.get('metrics/mAP50(B)', 0)):.4f}
mAP@0.5:0.95:     {float(best.get('metrics/mAP50-95(B)', 0)):.4f}
Val Total Loss:   {total_loss:.4f}
{'='*80}
"""
    with open(f"{REPORT_DIR}/{tag}_training_summary.txt", "w") as f:
        f.write(report)
    print(f"✓ Report: {tag}_training_summary.txt")

write_training_report("yolo11m_baseline", baseline_df)
write_training_report("yolo11m_p2head", p2head_df)
print("✓ Training reports and functions loaded")


# ============================================================================
# CELL 14: Test Set Evaluation
# ============================================================================
from ultralytics import YOLO

DATASET_ROOT = "/kaggle/input/c2a-dataset/C2A_Dataset/new_dataset3"
TEST_IMG_DIR = f"{DATASET_ROOT}/test/images"
TEST_LBL_DIR = f"{DATASET_ROOT}/test/labels"
VAL_IMG_DIR = f"{DATASET_ROOT}/val/images"
VAL_LBL_DIR = f"{DATASET_ROOT}/val/labels"

baseline_model = YOLO(f"{BASELINE_RUN}/weights/best.pt")
p2head_model = YOLO(f"{P2HEAD_RUN}/weights/best.pt")

print("\n" + "=" * 80 + f"\nPHASE 3: TEST SET ({TEST_IMAGES} images)\n" + "=" * 80)

baseline_test_df, baseline_test_summary, _ = evaluate_model_comprehensive(
    baseline_model, "yolo11m_baseline", TEST_IMG_DIR, TEST_LBL_DIR, "test", n_images=TEST_IMAGES)
p2head_test_df, p2head_test_summary, _ = evaluate_model_comprehensive(
    p2head_model, "yolo11m_p2head", TEST_IMG_DIR, TEST_LBL_DIR, "test", n_images=TEST_IMAGES)

visualize_predictions(baseline_model, TEST_IMG_DIR, TEST_LBL_DIR, "yolo11m_baseline", "test", min(15, TEST_IMAGES))
visualize_predictions(p2head_model, TEST_IMG_DIR, TEST_LBL_DIR, "yolo11m_p2head", "test", min(15, TEST_IMAGES))

test_comparison = generate_ablation_comparison(baseline_test_summary, p2head_test_summary, "P2-Head", "test")


# ============================================================================
# CELL 15: Validation Set Evaluation
# ============================================================================
print("\n" + "=" * 80 + f"\nPHASE 4: VALIDATION SET ({VAL_IMAGES} images)\n" + "=" * 80)

baseline_val_df, baseline_val_summary, _ = evaluate_model_comprehensive(
    baseline_model, "yolo11m_baseline", VAL_IMG_DIR, VAL_LBL_DIR, "val", n_images=VAL_IMAGES)
p2head_val_df, p2head_val_summary, _ = evaluate_model_comprehensive(
    p2head_model, "yolo11m_p2head", VAL_IMG_DIR, VAL_LBL_DIR, "val", n_images=VAL_IMAGES)

visualize_predictions(baseline_model, VAL_IMG_DIR, VAL_LBL_DIR, "yolo11m_baseline", "val", min(15, VAL_IMAGES))
visualize_predictions(p2head_model, VAL_IMG_DIR, VAL_LBL_DIR, "yolo11m_p2head", "val", min(15, VAL_IMAGES))

val_comparison = generate_ablation_comparison(baseline_val_summary, p2head_val_summary, "P2-Head", "val")


# ============================================================================
# CELL 16: Advanced Metrics
# ============================================================================
print("\n" + "=" * 80 + "\nPHASE 5: ADVANCED METRICS\n" + "=" * 80)

test_images = get_image_list(TEST_IMG_DIR)

print("\nConfidence Calibration:")
baseline_cal, baseline_ece = analyze_confidence_calibration(baseline_test_df, "yolo11m_baseline", "test")
p2head_cal, p2head_ece = analyze_confidence_calibration(p2head_test_df, "yolo11m_p2head", "test")

print("\nFailure Modes:")
baseline_failures = analyze_failure_modes(baseline_test_df, "yolo11m_baseline", "test")
p2head_failures = analyze_failure_modes(p2head_test_df, "yolo11m_p2head", "test")

print("\nSpeed Benchmarks:")
sample_img = f"{TEST_IMG_DIR}/{test_images[0]}"
print("Baseline:")
baseline_speed = benchmark_speed_vs_resolution(baseline_model, sample_img, "yolo11m_baseline")
print("\nP2-Head:")
p2head_speed = benchmark_speed_vs_resolution(p2head_model, sample_img, "yolo11m_p2head")


# ============================================================================
# CELL 17: Per-Size Recall Bar Chart (Key Ablation Visual)
# ============================================================================
size_categories = ['very_tiny', 'tiny', 'small', 'medium', 'large']
size_labels = ['Very Tiny\n(<8²px)', 'Tiny\n(8-16px)', 'Small\n(16-32px)', 'Medium\n(32-96px)', 'Large\n(>96px)']

baseline_recalls = [baseline_test_summary.get(f"{c}_recall", 0) for c in size_categories]
p2head_recalls = [p2head_test_summary.get(f"{c}_recall", 0) for c in size_categories]
baseline_counts = [baseline_test_summary.get(f"{c}_gt_count", 0) for c in size_categories]

x = np.arange(len(size_categories))
width = 0.35

fig, ax1 = plt.subplots(figsize=(14, 7))
bars1 = ax1.bar(x - width/2, baseline_recalls, width, label='Baseline', color='#2196F3', alpha=0.8)
bars2 = ax1.bar(x + width/2, p2head_recalls, width, label='P2-Head', color='#FF5722', alpha=0.8)

# Add count annotations
for i, count in enumerate(baseline_counts):
    ax1.text(i, max(baseline_recalls[i], p2head_recalls[i]) + 0.02,
             f'n={count}', ha='center', fontsize=9, color='gray')

ax1.set_xlabel('Object Size Category', fontsize=12)
ax1.set_ylabel('Recall', fontsize=12)
ax1.set_title('Per-Size Recall: Baseline vs P2-Head (Test Set)', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(size_labels)
ax1.legend(fontsize=11)
ax1.set_ylim([0, 1.15])
ax1.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/per_size_recall_comparison.png", dpi=300)
plt.close()
print("✓ Per-size recall comparison plot saved")


# ============================================================================
# CELL 18: Final Summary & Master Report
# ============================================================================
print("\n" + "=" * 80 + "\nFINAL SUMMARY\n" + "=" * 80)

all_summaries = pd.DataFrame([
    baseline_test_summary, p2head_test_summary,
    baseline_val_summary, p2head_val_summary
])
all_summaries.to_excel(f"{EXCEL_DIR}/complete_evaluation_summary.xlsx", index=False)

# Best epoch info
best_baseline_epoch = int(baseline_df.loc[baseline_df["metrics/mAP50(B)"].idxmax(), "epoch"])
best_p2head_epoch = int(p2head_df.loc[p2head_df["metrics/mAP50(B)"].idxmax(), "epoch"])

master_report = f"""
{'='*80}
DISASTER HUMAN DETECTION: P2 EXTRA DETECTION HEAD ABLATION STUDY
{'='*80}

CONFIGURATION:
  Test Mode:      {TEST_MODE}
  Epochs:         {NUM_EPOCHS}
  Data Fraction:  {TRAIN_FRACTION*100:.0f}%
  Image Size:     640px

MODEL COMPLEXITY:
  Baseline Params:   {baseline_info['Parameters']:,}
  P2-Head Params:    {p2head_info['Parameters']:,}
  Param Overhead:    {p2head_info['Parameters'] - baseline_info['Parameters']:,} ({(p2head_info['Parameters']/baseline_info['Parameters'] - 1)*100:.1f}%)

TRAINING (Best Epoch):
  Baseline: epoch {best_baseline_epoch} | val_total_loss={float(baseline_df.loc[baseline_df['metrics/mAP50(B)'].idxmax(), 'val/total_loss']):.4f}
  P2-Head:  epoch {best_p2head_epoch} | val_total_loss={float(p2head_df.loc[p2head_df['metrics/mAP50(B)'].idxmax(), 'val/total_loss']):.4f}

OFFICIAL METRICS:
  Baseline: mAP50={baseline_official['mAP50']:.4f} mAP50-95={baseline_official['mAP50-95']:.4f}
  P2-Head:  mAP50={p2head_official['mAP50']:.4f} mAP50-95={p2head_official['mAP50-95']:.4f}

TEST SET:
  Baseline: P={baseline_test_summary['Precision']:.4f} R={baseline_test_summary['Recall']:.4f} F1={baseline_test_summary['F1']:.4f}
  P2-Head:  P={p2head_test_summary['Precision']:.4f} R={p2head_test_summary['Recall']:.4f} F1={p2head_test_summary['F1']:.4f}

SMALL OBJECT RECALL (TEST):
  Very Tiny (<8²px):  Baseline={baseline_test_summary.get('very_tiny_recall',0):.4f} | P2-Head={p2head_test_summary.get('very_tiny_recall',0):.4f}
  Tiny (8-16px):      Baseline={baseline_test_summary.get('tiny_recall',0):.4f} | P2-Head={p2head_test_summary.get('tiny_recall',0):.4f}
  Small (16-32px):    Baseline={baseline_test_summary.get('small_recall',0):.4f} | P2-Head={p2head_test_summary.get('small_recall',0):.4f}

VALIDATION SET:
  Baseline: P={baseline_val_summary['Precision']:.4f} R={baseline_val_summary['Recall']:.4f} F1={baseline_val_summary['F1']:.4f}
  P2-Head:  P={p2head_val_summary['Precision']:.4f} R={p2head_val_summary['Recall']:.4f} F1={p2head_val_summary['F1']:.4f}

CALIBRATION:
  Baseline ECE: {baseline_ece if baseline_ece else 'N/A'}
  P2-Head ECE:  {p2head_ece if p2head_ece else 'N/A'}

RECOMMENDATION: {'P2-Head' if p2head_test_summary.get('very_tiny_recall', 0) > baseline_test_summary.get('very_tiny_recall', 0) else 'Baseline'} \
(very-tiny recall priority for disaster detection)

OUTPUTS:
  Excel:   {EXCEL_DIR}
  Plots:   {PLOT_DIR}
  Reports: {REPORT_DIR}
{'='*80}
"""

with open(f"{REPORT_DIR}/MASTER_SUMMARY_P2HEAD.txt", "w") as f:
    f.write(master_report)
print(master_report)


# ============================================================================
# CELL 19: Package Results
# ============================================================================
from IPython.display import FileLink
!zip -r /kaggle/working/p2head_results.zip {EXCEL_DIR} {PLOT_DIR} {REPORT_DIR}
FileLink("/kaggle/working/p2head_results.zip")

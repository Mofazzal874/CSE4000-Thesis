"""
================================================================================
DISASTER HUMAN DETECTION: CBAM ABLATION STUDY (BULLETPROOF FIX)
================================================================================

CRITICAL FIX APPLIED:
- CBAM now ALWAYS uses c1 (input channels), IGNORES c2
- This bypasses all YAML/parse_model issues
- Works regardless of what's in the YAML file

================================================================================
"""

# ============================================================================
# CELL 1: Install Dependencies (and fix any corrupted ultralytics)
# ============================================================================
# Reinstall ultralytics to fix any corrupted files from previous runs
!pip uninstall ultralytics -y -q 2>/dev/null
!pip install -q -U ultralytics timm thop pandas matplotlib openpyxl scikit-learn
print("✓ All dependencies installed successfully")


# ============================================================================
# CELL 2: Load Baseline Model and Extract YAML
# ============================================================================
from ultralytics import YOLO
import yaml

model = YOLO("yolo11m.pt")
model.info()

cfg = model.model.yaml
with open("yolov11m_original.yaml", "w") as f:
    yaml.dump(cfg, f)
with open("yolov11m_cbam.yaml", "w") as f:
    yaml.dump(cfg, f)

print("✓ Extracted YOLOv11m architecture")


# ============================================================================
# CELL 3: Replace C2PSA with CBAM
# ============================================================================
import yaml
import shutil

shutil.copy("yolov11m_cbam.yaml", "yolov11m_cbam_backup.yaml")

with open("yolov11m_cbam.yaml", "r") as f:
    cfg = yaml.safe_load(f)

replaced_count = 0
for i, layer in enumerate(cfg["backbone"]):
    if len(layer) >= 3 and layer[2] == "C2PSA":
        # CBAM with lazy init: args = [reduction, kernel_size]
        # Channels will be auto-detected from input tensor at runtime
        # This bypasses all Ultralytics parse_model channel computation issues
        cfg["backbone"][i] = [-1, 1, "CBAM", [16, 7]]
        replaced_count += 1
        print(f"Layer {i}: C2PSA → CBAM(reduction=16, kernel_size=7) [lazy init]")

assert replaced_count > 0, "❌ No C2PSA found!"

with open("yolov11m_cbam.yaml", "w") as f:
    yaml.dump(cfg, f)

print(f"\n✓ Replaced {replaced_count} layer(s) with lazy-init CBAM")


# ============================================================================
# CELL 4: Define and SAVE CBAM to File (for DDP)
# ============================================================================
import torch
import torch.nn as nn

# Define CBAM classes with LAZY INITIALIZATION
# This is the bulletproof solution that ignores YAML channel args
# and detects actual channels from input tensor at runtime
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
    
    CRITICAL: This implementation auto-detects input channels from the actual
    input tensor during the first forward pass. This bypasses all Ultralytics
    parse_model channel computation issues.
    
    Args (flexible - accepts various formats from Ultralytics):
        - CBAM(reduction, kernel_size): Standard format from YAML [16, 7]
        - CBAM(c1, c2, reduction, kernel_size): If channels are passed
        - CBAM(): Uses defaults (reduction=16, kernel_size=7)
    
    The actual channels are ALWAYS determined from input tensor shape.
    """
    def __init__(self, *args, **kwargs):
        super(CBAM, self).__init__()
        
        # Parse args flexibly - handle any format Ultralytics might pass
        # Default values
        self.reduction = 16
        self.kernel_size = 7
        
        if len(args) == 0:
            # No args - use defaults
            pass
        elif len(args) == 1:
            # Single arg - could be reduction
            if isinstance(args[0], int) and args[0] <= 32:  # Likely reduction
                self.reduction = args[0]
        elif len(args) == 2:
            # Two args - likely (reduction, kernel_size)
            if isinstance(args[0], int) and args[0] <= 32:
                self.reduction = args[0]
                self.kernel_size = args[1] if isinstance(args[1], int) else 7
            else:
                # Could be (c1, c2) - ignore, detect from input
                pass
        elif len(args) >= 4:
            # Four args - likely (c1, c2, reduction, kernel_size)
            # Ignore c1, c2 - we'll detect from input
            self.reduction = args[2] if isinstance(args[2], int) else 16
            self.kernel_size = args[3] if isinstance(args[3], int) else 7
        
        # Override with kwargs if provided
        self.reduction = kwargs.get("reduction", self.reduction)
        self.kernel_size = kwargs.get("kernel_size", self.kernel_size)
        
        # Ensure valid kernel_size
        if self.kernel_size not in (3, 7):
            self.kernel_size = 7
        
        # Lazy initialization flag - modules created on first forward
        self._initialized = False
        self.channel_attention = None
        self.spatial_attention = None
        self._channels = None
    
    def _lazy_init(self, channels, device, dtype):
        """Initialize attention modules with actual input channels"""
        self._channels = channels
        self.channel_attention = ChannelAttention(channels, self.reduction)
        self.spatial_attention = SpatialAttention(self.kernel_size)
        
        # Move to correct device and dtype
        self.channel_attention = self.channel_attention.to(device=device, dtype=dtype)
        self.spatial_attention = self.spatial_attention.to(device=device, dtype=dtype)
        
        self._initialized = True
    
    def forward(self, x):
        # Lazy initialization on first forward pass
        if not self._initialized:
            self._lazy_init(x.size(1), x.device, x.dtype)
        
        # Apply attention
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
'''

# Save to file that DDP can import
with open('/kaggle/working/cbam_module.py', 'w') as f:
    f.write(cbam_code)

# Also define in current namespace
exec(cbam_code)

# Test CBAM with various arg formats (simulating Ultralytics behavior)
print("Testing CBAM with lazy initialization...")
print("="*60)

# Test 1: Args format [reduction, kernel_size] - what we use in YAML
print("\nTest 1: CBAM(16, 7) - YAML format [reduction, kernel_size]")
cbam1 = CBAM(16, 7)
for ch in [512, 256, 1024]:  # Various input channels
    x = torch.randn(2, ch, 20, 20)
    cbam1._initialized = False  # Reset for testing different channels
    out = cbam1(x)
    assert out.shape == x.shape
    print(f"  ✓ Input {ch}ch → Output {out.shape[1]}ch (auto-detected)")

# Test 2: No args - pure defaults
print("\nTest 2: CBAM() - no args, pure defaults")
cbam2 = CBAM()
x = torch.randn(2, 512, 20, 20)
out = cbam2(x)
assert out.shape == x.shape
print(f"  ✓ Input 512ch → Output {out.shape[1]}ch")

# Test 3: Old format with channel args (should ignore channels, use from input)
print("\nTest 3: CBAM(1024, 1024, 16, 7) - channels in args (should be ignored)")
cbam3 = CBAM(1024, 1024, 16, 7)  # Even if args say 1024...
x = torch.randn(2, 512, 20, 20)   # ...input is 512
out = cbam3(x)
assert out.shape == x.shape
assert cbam3._channels == 512  # Should detect actual channels
print(f"  ✓ YAML said 1024ch, input was 512ch, CBAM used 512ch (correct!)")

print("\n" + "="*60)
print("✓ All CBAM tests passed!")
print("✓ CBAM saved to /kaggle/working/cbam_module.py")
print("\nCBAM is now BULLETPROOF:")
print("  - Ignores channel args from YAML")
print("  - Auto-detects actual input channels at runtime")
print("  - Works with any Ultralytics parse_model behavior")


# ============================================================================
# CELL 5: Register CBAM (Simple approach - no file patching)
# ============================================================================
import sys
import os

# Add working directory to path so imports work
if '/kaggle/working' not in sys.path:
    sys.path.insert(0, '/kaggle/working')

# Import from file
from cbam_module import CBAM, ChannelAttention, SpatialAttention

import ultralytics.nn.modules as modules
import ultralytics.nn.tasks as tasks

# Register in modules and tasks
modules.CBAM = CBAM
modules.ChannelAttention = ChannelAttention
modules.SpatialAttention = SpatialAttention

tasks.CBAM = CBAM
tasks.ChannelAttention = ChannelAttention
tasks.SpatialAttention = SpatialAttention

# Verify registration
assert "CBAM" in dir(modules), "CBAM not in modules!"
assert "CBAM" in dir(tasks), "CBAM not in tasks!"

print("✓ CBAM registered successfully")
print("⚠ Note: Multi-GPU (DDP) requires manual ultralytics patching")
print("  Using single GPU for reliable CBAM training")


# ============================================================================
# CELL 6: Dataset Configuration
# ============================================================================
dataset_yaml_content = """train: /kaggle/input/c2a-dataset/C2A_Dataset/new_dataset3/train/images
val: /kaggle/input/c2a-dataset/C2A_Dataset/new_dataset3/val/images
test: /kaggle/input/c2a-dataset/C2A_Dataset/new_dataset3/test/images

nc: 1
names: ['person']
"""

with open("c2a.yaml", "w") as f:
    f.write(dataset_yaml_content)

print("✓ Dataset configuration saved")


# ============================================================================
# CELL 7: Train Baseline YOLOv11m (Multi-GPU)
# ============================================================================
from ultralytics import YOLO
import torch

print("="*80)
print("TRAINING BASELINE YOLOv11m")
print("="*80)

# ===========================================
# TRAINING DATA FRACTION CONTROL
# ===========================================
# Set to 1.0 for full dataset, 0.2 for 20% of data
TRAIN_FRACTION = 1.0  # <-- CHANGE THIS: 0.2 = 20%, 1.0 = 100%
print(f"\n⚙ Training with {TRAIN_FRACTION*100:.0f}% of training data")
# ===========================================

num_gpus = torch.cuda.device_count()
print(f"\nAvailable GPUs: {num_gpus}")
for i in range(num_gpus):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

baseline = YOLO("yolo11m.pt")

if num_gpus >= 2:
    device_config = [0, 1]
    batch_size = 32
    print(f"\n✓ Multi-GPU: GPUs {device_config}, batch={batch_size}")
else:
    device_config = 0
    batch_size = 16
    print(f"\n✓ Single GPU: GPU 0, batch={batch_size}")

baseline.train(
    data="c2a.yaml",
    epochs=70,
    imgsz=640,
    batch=batch_size,
    device=device_config,
    name="yolo11m_baseline",
    patience=10,
    save=True,
    save_period=-1,
    verbose=True,
    plots=True,
    fraction=TRAIN_FRACTION  # Use specified fraction of training data
)

print("\n✓ Baseline training complete")


# ============================================================================
# CELL 8: Train CBAM-Modified YOLOv11m
# ============================================================================
from ultralytics import YOLO
import torch

print("="*80)
print("TRAINING CBAM-MODIFIED YOLOv11m")
print("="*80)

num_gpus = torch.cuda.device_count()
print(f"\nAvailable GPUs: {num_gpus}")
for i in range(num_gpus):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

cbam_model = YOLO("yolov11m_cbam.yaml")
cbam_model.load("yolo11m.pt")

print("✓ CBAM architecture loaded")
print("✓ Pretrained weights transferred")

# FORCE SINGLE GPU - DDP with custom modules requires complex patching
# Single GPU is reliable and still fast on T4
device_config = 0
batch_size = 16

print(f"\n✓ Single GPU training: GPU 0, batch={batch_size}")
print("  (Single GPU avoids DDP subprocess import issues with custom CBAM)")
print(f"⚙ Training with {TRAIN_FRACTION*100:.0f}% of training data")

cbam_model.train(
    data="c2a.yaml",
    epochs=70,
    imgsz=640,
    batch=batch_size,
    device=device_config,
    name="yolo11m_cbam",
    patience=10,
    save=True,
    save_period=-1,
    verbose=True,
    plots=True,
    fraction=TRAIN_FRACTION  # Use same fraction as baseline for fair comparison
)

print("\n✓ CBAM model training complete")




# ============================================================================
# CELL 9: Quick Validation (MODIFIED for CBAM)
# ============================================================================
from ultralytics import YOLO

baseline = YOLO("runs/detect/yolo11m_baseline/weights/best.pt")
cbam_model = YOLO("runs/detect/yolo11m_cbam/weights/best.pt")

print("Baseline Validation:")
base_metrics = baseline.val(data="c2a.yaml", split="val")

print("\nCBAM Validation:")
cbam_metrics = cbam_model.val(data="c2a.yaml", split="val")

print("\n✓ Quick validation complete")


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
CBAM_RUN = "runs/detect/yolo11m_cbam"

baseline_df = pd.read_csv(f"{BASELINE_RUN}/results.csv")
cbam_df = pd.read_csv(f"{CBAM_RUN}/results.csv")

baseline_df.to_excel(f"{EXCEL_DIR}/yolo11m_baseline_training.xlsx", index=False)
cbam_df.to_excel(f"{EXCEL_DIR}/yolo11m_cbam_training.xlsx", index=False)

def plot_losses(df, tag):
    plt.figure(figsize=(12, 6))
    for k in ["box", "cls", "dfl"]:
        if f"train/{k}_loss" in df.columns:
            plt.plot(df["epoch"], df[f"train/{k}_loss"], label=f"Train {k}", alpha=0.7)
            plt.plot(df["epoch"], df[f"val/{k}_loss"], label=f"Val {k}", linestyle='--')
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{tag} - Loss Curves")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{tag}_losses.png", dpi=300)
    plt.close()

plot_losses(baseline_df, "yolo11m_baseline")
plot_losses(cbam_df, "yolo11m_cbam")

print("✓ Training curves exported")


def plot_metrics(df, tag):
    plt.figure(figsize=(14, 6))
    for col in ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
        if col in df.columns:
            label = col.replace("metrics/", "").replace("(B)", "")
            plt.plot(df["epoch"], df[col], label=label, marker='o', markersize=3)
    plt.xlabel("Epoch"); plt.ylabel("Metric Value"); plt.title(f"{tag} - Validation Metrics")
    plt.legend(); plt.grid(True, alpha=0.3); plt.ylim([0, 1.05]); plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{tag}_metrics.png", dpi=300)
    plt.close()

plot_metrics(baseline_df, "yolo11m_baseline")
plot_metrics(cbam_df, "yolo11m_cbam")

print("✓ Metric evolution plots saved")



from datetime import datetime

def write_training_report(tag, df):
    best_idx = df["metrics/mAP50(B)"].idxmax() if "metrics/mAP50(B)" in df.columns else len(df)-1
    best = df.loc[best_idx]
    p, r = best.get("metrics/precision(B)", 0), best.get("metrics/recall(B)", 0)
    f1 = 2*p*r/(p+r+1e-9)
    
    report = f"""
{'='*80}
TRAINING SUMMARY: {tag}
{'='*80}
Best Epoch: {int(best['epoch'])}
Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}
mAP@0.5: {best.get('metrics/mAP50(B)', 0):.4f}
mAP@0.5:0.95: {best.get('metrics/mAP50-95(B)', 0):.4f}
{'='*80}
"""
    with open(f"{REPORT_DIR}/{tag}_training_summary.txt", "w") as f:
        f.write(report)
    print(f"✓ Report: {tag}_training_summary.txt")

write_training_report("yolo11m_baseline", baseline_df)
write_training_report("yolo11m_cbam", cbam_df)

print("\n✓ Training phase complete")




import os
import time
import cv2
import numpy as np
import torch
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
            boxes.append([xc-w/2, yc-h/2, xc+w/2, yc+h/2])
    return torch.tensor(boxes) if boxes else torch.empty((0, 4))

def match_predictions_to_gt(pred_boxes, gt_boxes, iou_thresh=0.5):
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
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
    if len(boxes) == 0:
        return {'tiny': torch.zeros(0, dtype=torch.bool), 'small': torch.zeros(0, dtype=torch.bool),
                'medium': torch.zeros(0, dtype=torch.bool), 'large': torch.zeros(0, dtype=torch.bool)}
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return {'tiny': areas < 16**2, 'small': (areas >= 16**2) & (areas < 32**2),
            'medium': (areas >= 32**2) & (areas < 96**2), 'large': areas >= 96**2}

print("✓ Helper functions loaded")







import pandas as pd
from tqdm import tqdm

def evaluate_model_comprehensive(model, model_name, img_dir, lbl_dir, split_name="test", n_images=None, conf=0.25):
    print(f"\n{'='*80}\nEVALUATING: {model_name} on {split_name.upper()}\n{'='*80}")
    image_list = get_image_list(img_dir)
    if n_images:
        image_list = image_list[:min(n_images, len(image_list))]
    
    results = []
    inference_times = []
    total_tp, total_fp, total_fn = 0, 0, 0
    size_stats = {'tiny': {'tp': 0, 'fn': 0}, 'small': {'tp': 0, 'fn': 0},
                  'medium': {'tp': 0, 'fn': 0}, 'large': {'tp': 0, 'fn': 0}}
    
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
            pred_boxes = preds[0].boxes.xyxy.cpu()
            avg_conf = float(preds[0].boxes.conf.cpu().mean())
        else:
            pred_boxes = torch.empty((0, 4))
            avg_conf = 0.0
        
        tp, fp, fn = match_predictions_to_gt(pred_boxes, gt_boxes, 0.5)
        total_tp += tp; total_fp += fp; total_fn += fn
        
        if len(gt_boxes) > 0:
            for size_cat, mask in categorize_by_size(gt_boxes).items():
                if mask.sum() > 0:
                    size_tp, _, size_fn = match_predictions_to_gt(pred_boxes, gt_boxes[mask], 0.5)
                    size_stats[size_cat]['tp'] += size_tp
                    size_stats[size_cat]['fn'] += size_fn
        
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        f2 = 5 * precision * recall / (4 * precision + recall + 1e-9)
        
        results.append({'Image': img_file, 'GT_Boxes': len(gt_boxes), 'Pred_Boxes': len(pred_boxes),
                       'TP': tp, 'FP': fp, 'FN': fn, 'Precision': precision, 'Recall': recall,
                       'F1': f1, 'F2': f2, 'Avg_Confidence': avg_conf, 'Inference_ms': infer_ms})
    
    df = pd.DataFrame(results)
    df.to_excel(f"{EXCEL_DIR}/{model_name}_{split_name}_detailed.xlsx", index=False)
    
    overall_precision = total_tp / (total_tp + total_fp + 1e-9)
    overall_recall = total_tp / (total_tp + total_fn + 1e-9)
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall + 1e-9)
    overall_f2 = 5 * overall_precision * overall_recall / (4 * overall_precision + overall_recall + 1e-9)
    
    size_recalls = {f"{cat}_recall": stats['tp'] / (stats['tp'] + stats['fn']) 
                   if stats['tp'] + stats['fn'] > 0 else 0.0 for cat, stats in size_stats.items()}
    
    summary = {'Model': model_name, 'Split': split_name, 'Images': len(image_list),
               'Total_GT': total_tp + total_fn, 'Total_Pred': total_tp + total_fp,
               'TP': total_tp, 'FP': total_fp, 'FN': total_fn,
               'Precision': overall_precision, 'Recall': overall_recall,
               'F1': overall_f1, 'F2': overall_f2, **size_recalls,
               'Avg_Inference_ms': np.mean(inference_times), 'Std_Inference_ms': np.std(inference_times)}
    
    print(f"\nSUMMARY: P={overall_precision:.4f} R={overall_recall:.4f} F1={overall_f1:.4f}")
    return df, summary, inference_times

print("✓ Evaluation function loaded")








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
        axes[i].set_title(f"GT: {gt_count} | Pred: {pred_count}", fontsize=10, color=color, fontweight='bold')
    
    for j in range(len(images), len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{model_name}_{split}_predictions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Visualization: {model_name}_{split}_predictions.png")

print("✓ Visualization function loaded")




def generate_ablation_comparison(baseline_summary, cbam_summary, split_name):
    metrics = [("Precision", "Precision"), ("Recall", "Recall"), ("F1", "F1"), ("F2", "F2"),
               ("Small Object Recall", "small_recall"), ("Avg Inference (ms)", "Avg_Inference_ms")]
    comparison_data = []
    for display_name, key in metrics:
        baseline_val = baseline_summary.get(key, 0.0)
        cbam_val = cbam_summary.get(key, 0.0)
        comparison_data.append({'Metric': display_name, 'Baseline': round(baseline_val, 4),
                               'CBAM': round(cbam_val, 4), 'Δ': round(cbam_val - baseline_val, 4),
                               'Δ (%)': round((cbam_val - baseline_val) / (baseline_val + 1e-9) * 100, 2)})
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_excel(f"{EXCEL_DIR}/ablation_comparison_{split_name}.xlsx", index=False)
    print(f"\n{'='*80}\nABLATION COMPARISON - {split_name.upper()}\n{'='*80}")
    print(comparison_df.to_string(index=False))
    print('='*80)
    return comparison_df

def generate_comprehensive_text_report(baseline_summary, cbam_summary, split_name):
    report = f"""
{'='*80}
ABLATION STUDY: {split_name.upper()} SET
{'='*80}
BASELINE: P={baseline_summary['Precision']:.4f} R={baseline_summary['Recall']:.4f} F1={baseline_summary['F1']:.4f}
CBAM:     P={cbam_summary['Precision']:.4f} R={cbam_summary['Recall']:.4f} F1={cbam_summary['F1']:.4f}

DELTA: ΔP={cbam_summary['Precision']-baseline_summary['Precision']:.4f} ΔR={cbam_summary['Recall']-baseline_summary['Recall']:.4f} ΔF1={cbam_summary['F1']-baseline_summary['F1']:.4f}

RECOMMENDATION: {'CBAM' if cbam_summary['Recall'] > baseline_summary['Recall'] else 'Baseline'} (Recall priority for disaster detection)
{'='*80}
"""
    with open(f"{REPORT_DIR}/ablation_study_{split_name}.txt", "w") as f:
        f.write(report)
    print(f"✓ Report: ablation_study_{split_name}.txt")
    return report

print("✓ Comparison functions loaded")




from sklearn.metrics import precision_recall_curve, average_precision_score

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
                'Count': bin_mask.sum(),
                'Gap': abs(df_with_preds.loc[bin_mask, 'Avg_Confidence'].mean() - df_with_preds.loc[bin_mask, 'Precision'].mean())
            })
    
    if not calibration_data:
        return None, None
    cal_df = pd.DataFrame(calibration_data)
    ece = np.average(cal_df['Gap'], weights=cal_df['Count'])
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
        times = [time.time() for _ in range(10)]
        for i in range(10):
            start = time.time()
            _ = model.predict(sample_img_path, imgsz=imgsz, verbose=False)
            times[i] = (time.time() - start) * 1000
        results.append({'Resolution': imgsz, 'Avg_ms': np.mean(times), 'FPS': 1000/np.mean(times)})
    
    speed_df = pd.DataFrame(results)
    speed_df.to_excel(f"{EXCEL_DIR}/{model_name}_speed_vs_resolution.xlsx", index=False)
    print(speed_df.to_string(index=False))
    return speed_df

print("✓ Advanced metrics loaded")








# ============================================================================
# CELL 18: Test Set Evaluation (MODIFIED for CBAM)
# ============================================================================
from ultralytics import YOLO

DATASET_ROOT = "/kaggle/input/c2a-dataset/C2A_Dataset/new_dataset3"
TEST_IMG_DIR = f"{DATASET_ROOT}/test/images"
TEST_LBL_DIR = f"{DATASET_ROOT}/test/labels"
VAL_IMG_DIR = f"{DATASET_ROOT}/val/images"
VAL_LBL_DIR = f"{DATASET_ROOT}/val/labels"

print("\n" + "="*80 + "\nPHASE 1: TEST SET (30 images)\n" + "="*80)

baseline_model = YOLO("runs/detect/yolo11m_baseline/weights/best.pt")
cbam_model = YOLO("runs/detect/yolo11m_cbam/weights/best.pt")  # Changed from eca_model

baseline_test_df, baseline_test_summary, _ = evaluate_model_comprehensive(
    baseline_model, "yolo11m_baseline", TEST_IMG_DIR, TEST_LBL_DIR, "test", n_images=30
)
cbam_test_df, cbam_test_summary, _ = evaluate_model_comprehensive(
    cbam_model, "yolo11m_cbam", TEST_IMG_DIR, TEST_LBL_DIR, "test", n_images=30
)

visualize_predictions(baseline_model, TEST_IMG_DIR, TEST_LBL_DIR, "yolo11m_baseline", "test", 15)
visualize_predictions(cbam_model, TEST_IMG_DIR, TEST_LBL_DIR, "yolo11m_cbam", "test", 15)

test_comparison = generate_ablation_comparison(baseline_test_summary, cbam_test_summary, "test")
test_report = generate_comprehensive_text_report(baseline_test_summary, cbam_test_summary, "test")


# ============================================================================
# CELL 19: Validation Set Evaluation (MODIFIED for CBAM)
# ============================================================================
print("\n" + "="*80 + "\nPHASE 2: VALIDATION SET (100 images)\n" + "="*80)

baseline_val_df, baseline_val_summary, _ = evaluate_model_comprehensive(
    baseline_model, "yolo11m_baseline", VAL_IMG_DIR, VAL_LBL_DIR, "val", n_images=100
)
cbam_val_df, cbam_val_summary, _ = evaluate_model_comprehensive(
    cbam_model, "yolo11m_cbam", VAL_IMG_DIR, VAL_LBL_DIR, "val", n_images=100
)

visualize_predictions(baseline_model, VAL_IMG_DIR, VAL_LBL_DIR, "yolo11m_baseline", "val", 15)
visualize_predictions(cbam_model, VAL_IMG_DIR, VAL_LBL_DIR, "yolo11m_cbam", "val", 15)

val_comparison = generate_ablation_comparison(baseline_val_summary, cbam_val_summary, "val")
val_report = generate_comprehensive_text_report(baseline_val_summary, cbam_val_summary, "val")


# ============================================================================
# CELL 20: Advanced Metrics Analysis (MODIFIED for CBAM)
# ============================================================================
print("\n" + "="*80 + "\nPHASE 3: ADVANCED METRICS\n" + "="*80)

test_images = get_image_list(TEST_IMG_DIR)

print("\nConfidence Calibration:")
baseline_cal, baseline_ece = analyze_confidence_calibration(baseline_test_df, "yolo11m_baseline", "test")
cbam_cal, cbam_ece = analyze_confidence_calibration(cbam_test_df, "yolo11m_cbam", "test")

print("\nFailure Modes:")
baseline_failures = analyze_failure_modes(baseline_test_df, "yolo11m_baseline", "test")
cbam_failures = analyze_failure_modes(cbam_test_df, "yolo11m_cbam", "test")

print("\nSpeed Benchmarks:")
sample_img = f"{TEST_IMG_DIR}/{test_images[0]}"
print("Baseline:")
baseline_speed = benchmark_speed_vs_resolution(baseline_model, sample_img, "yolo11m_baseline")
print("\nCBAM:")
cbam_speed = benchmark_speed_vs_resolution(cbam_model, sample_img, "yolo11m_cbam")


# ============================================================================
# CELL 21: Final Summary (MODIFIED for CBAM)
# ============================================================================
print("\n" + "="*80 + "\nFINAL SUMMARY\n" + "="*80)

all_summaries = pd.DataFrame([baseline_test_summary, cbam_test_summary, 
                               baseline_val_summary, cbam_val_summary])
all_summaries.to_excel(f"{EXCEL_DIR}/complete_evaluation_summary.xlsx", index=False)

master_report = f"""
{'='*80}
DISASTER HUMAN DETECTION: ABLATION STUDY (CBAM) COMPLETE
{'='*80}

TEST SET:
  Baseline: P={baseline_test_summary['Precision']:.4f} R={baseline_test_summary['Recall']:.4f} F1={baseline_test_summary['F1']:.4f}
  CBAM:     P={cbam_test_summary['Precision']:.4f} R={cbam_test_summary['Recall']:.4f} F1={cbam_test_summary['F1']:.4f}

VALIDATION SET:
  Baseline: P={baseline_val_summary['Precision']:.4f} R={baseline_val_summary['Recall']:.4f} F1={baseline_val_summary['F1']:.4f}
  CBAM:     P={cbam_val_summary['Precision']:.4f} R={cbam_val_summary['Recall']:.4f} F1={cbam_val_summary['F1']:.4f}

COMPARISON WITH PREVIOUS ECA RESULTS:
  ECA Test Recall: 0.8458
  CBAM Test Recall: {cbam_test_summary['Recall']:.4f}
  {'CBAM > ECA' if cbam_test_summary['Recall'] > 0.8458 else 'ECA >= CBAM'}

RECOMMENDATION: {'CBAM' if cbam_test_summary['Recall'] > baseline_test_summary['Recall'] else 'Baseline'} for disaster detection (recall priority)

CBAM ADVANTAGES:
  - Spatial attention captures object location
  - Better for partially occluded objects
  - More modeling capacity than ECA

OUTPUTS:
  Excel: {EXCEL_DIR}
  Plots: {PLOT_DIR}
  Reports: {REPORT_DIR}
{'='*80}
"""

with open(f"{REPORT_DIR}/MASTER_SUMMARY_CBAM.txt", "w") as f:
    f.write(master_report)

print(master_report)

from IPython.display import FileLink
!zip -r /kaggle/working/cbam_results.zip {EXCEL_DIR} {PLOT_DIR} {REPORT_DIR}

FileLink("/kaggle/working/cbam_results.zip")
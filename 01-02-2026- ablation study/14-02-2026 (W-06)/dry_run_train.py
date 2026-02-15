"""
DRY RUN: Smoke test for YOLO11m_CBAM_P2Head.py (modified version)
Tests: CBAM registration, YAML parsing, model build, dummy 1-epoch train
Run this on Kaggle BEFORE the full script to catch errors early.
"""

import os, sys, subprocess

# Auto-install dependencies if running on Kaggle/Colab
try:
    import ultralytics
except ImportError:
    print("Installing ultralytics & dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", 
                           "ultralytics", "timm", "thop", "pandas<3.0", "matplotlib<3.10", "openpyxl", "scikit-learn"])
    print("✓ Dependencies installed")

import shutil, yaml, gc
import cv2, numpy as np
import torch, torch.nn as nn
from ultralytics import YOLO
import ultralytics.nn.modules as modules
import ultralytics.nn.tasks as tasks

print("=" * 80)
print("DRY RUN SMOKE TEST: CBAM + P2 Head")
print("=" * 80)

# ========== 1. Check Pre-trained Models (from uploaded input) ==========
# Auto-detect the dataset folder (handles any nesting depth)
INPUT_DIR = None
_SEARCH_FILE = os.path.join("runs", "detect", "yolo11m_baseline", "weights", "best.pt")
print("\n[1/5] Searching for pre-trained models in /kaggle/input/...")

for root, dirs, files in os.walk("/kaggle/input"):
    if os.path.isfile(os.path.join(root, _SEARCH_FILE)):
        INPUT_DIR = root
        break

if INPUT_DIR:
    print(f"  ✓ Found dataset at: {INPUT_DIR}")
else:
    print("  ❌ Could not find pre-trained models!")
    print("  Listing /kaggle/input/ contents:")
    for root, dirs, files in os.walk("/kaggle/input"):
        level = root.replace("/kaggle/input", "").count(os.sep)
        if level < 4:
            indent = "     " + "  " * level
            print(f"{indent}📁 {os.path.basename(root)}/")
            for f in files[:5]:
                print(f"{indent}  📄 {f}")

BASELINE_BEST = f"{INPUT_DIR}/runs/detect/yolo11m_baseline/weights/best.pt" if INPUT_DIR else ""
CBAM_BEST = f"{INPUT_DIR}/runs/detect/yolo11m_cbam/weights/best.pt" if INPUT_DIR else ""
errors = 0
for path, label in [(BASELINE_BEST, "Baseline"), (CBAM_BEST, "CBAM")]:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1024**2
        print(f"  ✓ {label}: {path} ({size_mb:.1f} MB)")
    else:
        print(f"  ❌ {label}: NOT FOUND at {path}")
        errors += 1

if errors > 0:
    print(f"\n⚠ {errors} model(s) missing! Listing actual input contents:")
    if os.path.exists(INPUT_DIR):
        for root, dirs, files in os.walk(INPUT_DIR):
            level = root.replace(INPUT_DIR, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            if level < 3:  # Don't go too deep
                for f in files[:5]:
                    print(f'{indent}  {f}')
    else:
        print(f"  ❌ INPUT_DIR itself doesn't exist: {INPUT_DIR}")
        print("  → Check your Kaggle dataset name (slug)")

# ========== 2. Define & Register CBAM ==========
print("\n[2/5] Registering CBAM module...")

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        rc = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, rc, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(rc, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return x * self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=3 if kernel_size == 7 else 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class CBAM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.reduction = args[0] if len(args) >= 1 and isinstance(args[0], int) else 16
        self.kernel_size = args[1] if len(args) >= 2 and isinstance(args[1], int) else 7
        if self.kernel_size not in (3, 7): self.kernel_size = 7
        self._initialized = False
    def _lazy_init(self, channels, device, dtype):
        self.channel_attention = ChannelAttention(channels, self.reduction).to(device, dtype)
        self.spatial_attention = SpatialAttention(self.kernel_size).to(device, dtype)
        self._initialized = True
    def forward(self, x):
        if not self._initialized: self._lazy_init(x.size(1), x.device, x.dtype)
        return self.spatial_attention(self.channel_attention(x))

modules.CBAM = CBAM; tasks.CBAM = CBAM
modules.ChannelAttention = ChannelAttention; tasks.ChannelAttention = ChannelAttention
modules.SpatialAttention = SpatialAttention; tasks.SpatialAttention = SpatialAttention

# Write cbam_module.py so Ultralytics can import it when loading CBAM checkpoints
_cbam_module_code = '''
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        rc = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, rc, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(rc, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return x * self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=3 if kernel_size == 7 else 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class CBAM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.reduction = args[0] if len(args) >= 1 and isinstance(args[0], int) else 16
        self.kernel_size = args[1] if len(args) >= 2 and isinstance(args[1], int) else 7
        if self.kernel_size not in (3, 7): self.kernel_size = 7
        self._initialized = False
    def _lazy_init(self, channels, device, dtype):
        self.channel_attention = ChannelAttention(channels, self.reduction).to(device, dtype)
        self.spatial_attention = SpatialAttention(self.kernel_size).to(device, dtype)
        self._initialized = True
    def forward(self, x):
        if not self._initialized: self._lazy_init(x.size(1), x.device, x.dtype)
        return self.spatial_attention(self.channel_attention(x))
'''
with open("/kaggle/working/cbam_module.py", "w") as f:
    f.write(_cbam_module_code)
if "/kaggle/working" not in sys.path:
    sys.path.insert(0, "/kaggle/working")

x = torch.randn(1, 256, 20, 20)
assert CBAM(16, 7)(x).shape == x.shape
print("  ✓ CBAM registered, cbam_module.py written, forward pass OK")

# ========== 3. Build CBAM+P2 Model from YAML ==========
print("\n[3/5] Building CBAM+P2 architecture from YAML...")

cbam_p2_yaml = """
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
with open("_test_cbam_p2.yaml", "w") as f: f.write(cbam_p2_yaml)

try:
    model = YOLO("_test_cbam_p2.yaml")
    model.load("yolo11m.pt")
    detect = model.model.model[-1]
    n_scales = len(detect.stride)
    assert n_scales == 4, f"Expected 4 detection scales, got {n_scales}"
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"  ✓ Model built: {n_params:,} params, {n_scales} detection scales")
    print(f"  ✓ Strides: {detect.stride.tolist()}")
except Exception as e:
    print(f"  ❌ Model build FAILED: {e}")
    import traceback; traceback.print_exc()

# ========== 4. Dummy Training (1 epoch, 5 images) ==========
print("\n[4/5] Running 1-epoch dummy training...")

DUMMY = "_dry_run_data"
if os.path.exists(DUMMY): shutil.rmtree(DUMMY)
os.makedirs(f"{DUMMY}/images/train", exist_ok=True)
os.makedirs(f"{DUMMY}/labels/train", exist_ok=True)

for i in range(5):
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    cv2.imwrite(f"{DUMMY}/images/train/img_{i}.jpg", img)
    with open(f"{DUMMY}/labels/train/img_{i}.txt", "w") as f:
        f.write("0 0.5 0.5 0.1 0.1\n")

data_yaml = f"""
train: {os.path.abspath(DUMMY)}/images/train
val: {os.path.abspath(DUMMY)}/images/train
nc: 1
names: ['person']
"""
with open("_dry_run.yaml", "w") as f: f.write(data_yaml)

device = 0 if torch.cuda.is_available() else 'cpu'
try:
    model = YOLO("_test_cbam_p2.yaml")
    model.load("yolo11m.pt")
    model.train(data="_dry_run.yaml", epochs=1, imgsz=640, batch=2,
                device=device, verbose=False, plots=False,
                project="_dry_run_output", name="test", exist_ok=True)
    print("  ✓ Training completed successfully!")
except Exception as e:
    print(f"  ❌ Training FAILED: {e}")
    import traceback; traceback.print_exc()

# ========== 5. Load Pre-trained Models (if available) ==========
print("\n[5/5] Testing pre-trained model loading...")
if os.path.exists(BASELINE_BEST):
    try:
        m = YOLO(BASELINE_BEST)
        print(f"  ✓ Baseline loads OK: {sum(p.numel() for p in m.model.parameters()):,} params")
        del m
    except Exception as e:
        print(f"  ❌ Baseline load FAILED: {e}")

if os.path.exists(CBAM_BEST):
    try:
        m = YOLO(CBAM_BEST)
        print(f"  ✓ CBAM loads OK: {sum(p.numel() for p in m.model.parameters()):,} params")
        del m
    except Exception as e:
        print(f"  ❌ CBAM load FAILED: {e}")

# Cleanup
for p in [DUMMY, "_dry_run.yaml", "_test_cbam_p2.yaml", "_dry_run_output"]:
    if os.path.isdir(p): shutil.rmtree(p)
    elif os.path.isfile(p): os.remove(p)
torch.cuda.empty_cache(); gc.collect()

print("\n" + "=" * 80)
print("✅ DRY RUN COMPLETE" if errors == 0 else f"⚠ DRY RUN COMPLETE ({errors} warnings)")
print("=" * 80)

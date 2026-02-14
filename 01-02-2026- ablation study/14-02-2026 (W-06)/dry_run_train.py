
import os
import sys
import shutil
import cv2
import numpy as np
import yaml
import torch
import torch.nn as nn
from ultralytics import YOLO
import ultralytics.nn.modules as modules
import ultralytics.nn.tasks as tasks

print("="*80)
print("FULL DRY RUN TRAIN TEST: YOLO11m + CBAM + P2 HEAD")
print("="*80)

# 1. Setup Dummy Dataset
BASE = "dry_run_data"
if os.path.exists(BASE): shutil.rmtree(BASE)
os.makedirs(f"{BASE}/images/train", exist_ok=True)
os.makedirs(f"{BASE}/labels/train", exist_ok=True)

# Generate 5 dummy images + labels
for i in range(5):
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (50,50), (100,100), (255,255,255), -1)
    cv2.imwrite(f"{BASE}/images/train/img_{i}.jpg", img)
    with open(f"{BASE}/labels/train/img_{i}.txt", "w") as f:
        f.write("0 0.117 0.117 0.078 0.078\n") # centered box

# Create dataset.yaml
data_yaml = f"""
train: {os.path.abspath(BASE)}/images/train
val: {os.path.abspath(BASE)}/images/train
names: ['person']
nc: 1
"""
with open("dry_run_data.yaml", "w") as f: f.write(data_yaml)
print("✓ Dummy dataset created")

# 2. Define & Register CBAM
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
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(concat))

class CBAM(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CBAM, self).__init__()
        self.reduction = 16
        self.kernel_size = 7
        if len(args) == 1 and isinstance(args[0], int): self.reduction = args[0]
        elif len(args) >= 2: self.reduction = args[0]; self.kernel_size = args[1]
        self._initialized = False
    def _lazy_init(self, channels, device, dtype):
        self.channel_attention = ChannelAttention(channels, self.reduction).to(device, dtype)
        self.spatial_attention = SpatialAttention(self.kernel_size).to(device, dtype)
        self._initialized = True
    def forward(self, x):
        if not self._initialized: self._lazy_init(x.size(1), x.device, x.dtype)
        return self.spatial_attention(self.channel_attention(x))

modules.CBAM = CBAM
tasks.CBAM = CBAM
print("✓ CBAM registered")

# 3. Create CBAM+P2 YAML
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
  - [-1, 1, CBAM, [16, 7]] # CBAM Custom Module
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
with open("test_cbam_p2.yaml", "w") as f: f.write(cbam_p2_yaml)
print("✓ YAML created")

# 4. Train Loop
device = 0 if torch.cuda.is_available() else 'cpu'
print(f"✓ Training on device: {device}")

try:
    model = YOLO("test_cbam_p2.yaml")
    results = model.train(
        data="dry_run_data.yaml",
        epochs=1,
        imgsz=640,
        batch=2,
        device=device,
        verbose=True,
        plots=False,
        project="dry_run_results",
        name="test_run",
        exist_ok=True
    )
    print("\n✅ DRY RUN SUCCESS!")
    print(f"  Final Box Loss: {results.box.loss if hasattr(results.box, 'loss') else 'N/A'}")
    
except Exception as e:
    print(f"\n❌ DRY RUN FAILED: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
if os.path.exists(BASE): shutil.rmtree(BASE)
if os.path.exists("dry_run_data.yaml"): os.remove("dry_run_data.yaml")
if os.path.exists("test_cbam_p2.yaml"): os.remove("test_cbam_p2.yaml")

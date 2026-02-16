# YAML Configuration — How It Defines the Architecture

> **What you'll learn:** How a simple text file (YAML) controls every layer in YOLOv11m, and how editing it changes the architecture.

---

## 1. What is YAML?

**YAML = "YAML Ain't Markup Language"** — it's a human-readable data format. Think of it as a structured text file (like JSON but easier to read).

```yaml
# This is a YAML comment
name: "YOLOv11m"
nc: 1                    # number of classes
scales:
  m: [0.50, 1.00, 512]  # [depth, width, max_channels]
```

> YAML uses **indentation** (spaces, not tabs!) to define structure. It's the "blueprint" that tells Ultralytics how to build the neural network.

---

## 2. The Two YAMLs in Our Project

We use **2 types** of YAML files:

### YAML Type 1: Dataset Configuration (`c2a.yaml`)
Tells YOLO where your images and labels are:

```yaml
train: /kaggle/input/c2a-dataset/C2A_Dataset/new_dataset3/train/images
val: /kaggle/input/c2a-dataset/C2A_Dataset/new_dataset3/val/images
test: /kaggle/input/c2a-dataset/C2A_Dataset/new_dataset3/test/images

nc: 1              # Number of classes (we only have "person")
names: ['person']  # Class names
```

### YAML Type 2: Model Architecture (`yolov11m_p2head.yaml`)
Defines every single layer of the neural network. **This is the one that controls the architecture.**

---

## 3. Anatomy of a Model YAML

```yaml
nc: 1                      # ← Number of classes
scales:
  m: [0.50, 1.00, 512]    # ← Model size scaling

backbone:                  # ← Feature extraction layers
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - ...

head:                      # ← Detection layers
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - ...
  - [[16, 19, 22], 1, Detect, [nc]]  # ← Final output
```

---

## 4. How to Read Each Layer Line

Every layer is defined as a list with 4 elements:

```yaml
#  [from,  repeats,  module,       arguments]
   [-1,    1,        Conv,         [64, 3, 2]]
```

Let's break this down:

| Field | Value | Meaning |
|---|---|---|
| `from` | `-1` | Input comes from **previous layer** |
| `repeats` | `1` | Build this module **1 time** |
| `module` | `Conv` | The type of layer to build |
| `arguments` | `[64, 3, 2]` | Parameters for that module |

### `from` — Where Does the Input Come From?

```yaml
-1          → Previous layer (most common)
-2          → Two layers back
6           → Layer #6 specifically (skip connection!)
[-1, 6]     → BOTH previous layer AND layer #6 (for Concat)
[16, 19, 22] → Layers 16, 19, and 22 (for Detect head)
```

### `repeats` — How Many Times to Build
```yaml
1  → Build once (Conv, Upsample, Concat, SPPF)
2  → Build twice (C3k2 — creates 2 sequential bottleneck blocks)
```

> **Scaling factor matters!** If YAML says `repeats=2` and `depth=0.50`, actual repeats = `max(round(2 × 0.50), 1) = 1`

### Module Arguments by Type

```yaml
# Conv [out_channels, kernel_size, stride]
Conv [64, 3, 2]     # 64 output channels, 3×3 kernel, stride 2

# C3k2 [out_channels, shortcut, expansion_ratio]
C3k2 [256, False, 0.25]  # 256 channels, no shortcut, 25% expansion

# SPPF [out_channels, pool_size]
SPPF [1024, 5]      # 1024 channels, 5×5 max pooling

# C2PSA [out_channels]
C2PSA [1024]         # 1024 channels, spatial attention

# nn.Upsample [size, scale_factor, mode]
nn.Upsample [None, 2, "nearest"]  # Double spatial size

# Concat [dimension]
Concat [1]           # Concatenate along channel dimension

# Detect [num_classes]
Detect [nc]          # nc = 1 for our single-class task
```

---

## 5. The `scales` Parameter — One YAML, Multiple Models

```yaml
scales:
  n: [0.50, 0.25, 1024]  # nano:   Lightest, fastest, least accurate
  s: [0.50, 0.50, 1024]  # small:  Good for mobile
  m: [0.50, 1.00, 512]   # medium: Best balance (Thesis Choice)
  l: [1.00, 1.00, 512]   # large:  Accurate but slow
  x: [1.00, 1.50, 512]   # extra:  Heavy, max accuracy
```

The three numbers are `[depth_multiple, width_multiple, max_channels]`:

### 1. Depth Multiple (D)
Controls **how many layers** are in each block.
- `D=0.50`: If YAML says `repeats=2`, the model builds `2 * 0.5 = 1` layer.
- `D=1.00`: If YAML says `repeats=2`, the model builds `2 * 1.0 = 2` layers.
- **Effect**: Higher depth = more complex features, slower speed.

### 2. Width Multiple (W)
Controls **how many channels** (filters) are in each layer.
- `W=0.50`: If YAML says `channels=256`, the model uses `256 * 0.5 = 128` channels.
- `W=1.00`: If YAML says `channels=256`, the model uses `256 * 1.0 = 256` channels.
- **Effect**: Wider layers = can learn more patterns, uses more VRAM.

### 3. Max Channels (C_max)
A safety cap. Even if `width * channels` is huge, it won't exceed this number (usually 512 or 1024).

### 📊 Comparing the Variants
| Scale | Name | Depth | Width | Params (Approx) | Use Case |
|---|---|---|---|---|---|
| **n** | Nano | 0.50 | 0.25 | ~2.6M | Edge devices, real-time video |
| **s** | Small | 0.50 | 0.50 | ~9.4M | Mobile apps, reasonable accuracy |
| **m** | **Medium** | **0.50** | **1.00** | **~20.1M** | **General purpose (Our Thesis)** |
| **l** | Large | 1.00 | 1.00 | ~25.3M | High accuracy, GPU needed |
| **x** | X-Large | 1.00 | 1.50 | ~56.9M | Server-side, max accuracy |

> **Why we chose 'm' (Medium):** 
> - **'n'/'s'** are too weak for tiny 5px objects (too few channels).
> - **'l'/'x'** are too slow for real-time drone inference on T4 GPUs.
> - **'m'** offers the best trade-off: enough width (1.00) to capture fine details, but shallow enough (0.50) to run fast.

### Example: How scaling works for one layer

```yaml
# YAML definition:
- [-1, 2, C3k2, [256, False, 0.25]]

# For scale 'm' [0.50, 1.00, 512]:
#   repeats = max(round(2 × 0.50), 1) = 1
#   channels = min(256 × 1.00, 512)   = 256
# → Actual: C3k2 with 256 channels, 1 repeat

# For scale 'n' [0.50, 0.25, 1024]:
#   repeats = max(round(2 × 0.50), 1) = 1
#   channels = min(256 × 0.25, 1024)  = 64
# → Actual: C3k2 with 64 channels, 1 repeat (much smaller!)
```

---

## 6. Standard YOLOv11m YAML (Annotated)

```yaml
nc: 1
scales:
  m: [0.50, 1.00, 512]

backbone:
  #  from  repeats  module              args                  # idx  output
  - [-1,   1,       Conv,       [64, 3, 2]]                  # [0]  320×320×64
  - [-1,   1,       Conv,       [128, 3, 2]]                 # [1]  160×160×128  (P2)
  - [-1,   2,       C3k2,       [256, False, 0.25]]          # [2]  160×160×256
  - [-1,   1,       Conv,       [256, 3, 2]]                 # [3]  80×80×256   (P3)
  - [-1,   2,       C3k2,       [512, False, 0.25]]          # [4]  80×80×512
  - [-1,   1,       Conv,       [512, 3, 2]]                 # [5]  40×40×512   (P4)
  - [-1,   2,       C3k2,       [512, True]]                 # [6]  40×40×512
  - [-1,   1,       Conv,       [1024, 3, 2]]                # [7]  20×20×1024  (P5)
  - [-1,   2,       C3k2,       [1024, True]]                # [8]  20×20×1024
  - [-1,   1,       SPPF,       [1024, 5]]                   # [9]  20×20×1024
  - [-1,   2,       C2PSA,      [1024]]                      # [10] 20×20×1024

head:
  # FPN: Top-down pathway (upsample + merge)
  - [-1,    1, nn.Upsample, [None, 2, "nearest"]]            # [11] 40×40
  - [[-1, 6], 1, Concat, [1]]                                # [12] cat with layer 6
  - [-1,    2, C3k2, [512, False]]                            # [13] 40×40×512

  - [-1,    1, nn.Upsample, [None, 2, "nearest"]]            # [14] 80×80
  - [[-1, 4], 1, Concat, [1]]                                # [15] cat with layer 4
  - [-1,    2, C3k2, [256, False]]                            # [16] 80×80×256 → P3

  # PAN: Bottom-up pathway (downsample + merge)
  - [-1,    1, Conv, [256, 3, 2]]                             # [17] 40×40
  - [[-1, 13], 1, Concat, [1]]                               # [18] cat with layer 13
  - [-1,    2, C3k2, [512, False]]                            # [19] 40×40×512 → P4

  - [-1,    1, Conv, [512, 3, 2]]                             # [20] 20×20
  - [[-1, 10], 1, Concat, [1]]                               # [21] cat with layer 10
  - [-1,    2, C3k2, [1024, True]]                            # [22] 20×20×1024 → P5

  - [[16, 19, 22], 1, Detect, [nc]]                           # [23] 3-scale detect
```

---

### Generating the YAML in Python
In `YOLO11m_P2Head.py`, we don't just edit a file manually. We generate it with code to ensure it's correct every time:

```python
# From YOLO11m_P2Head.py
p2_yaml_content = """# YOLOv11m + P2 Extra Detection Head
nc: 1
scales:
  m: [0.50, 1.00, 512]

backbone:
  # ... (standard backbone) ...

head:
  # ... (standard head) ...
  
  # ★ THE NEW PART ADDED BY SCRIPT:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]          # 17
  - [[-1, 2], 1, Concat, [1]]                            # 18
  - [-1, 2, C3k2, [128, False]]                          # 19

  # ... (rest of head) ...

  - [[19, 22, 25, 28], 1, Detect, [nc]]                  # 29
"""

with open("yolov11m_p2head.yaml", "w") as f:
    f.write(p2_yaml_content)
```

> **Why do this?** This guarantees that the YAML file exists and is correct before training starts, even if you move the script to a new machine.

---

## 7. Where We Change the YAML (Our Ablation Study)

### Baseline (Standard — no changes)
Uses the YAML above exactly as-is: `C2PSA` in backbone, 3-scale detection.

### CBAM Variant (Attention swap)
**One line changed** in the backbone:
```diff
  # Layer 10 in backbone:
- - [-1, 2, C2PSA, [1024]]          # Standard: C2PSA attention
+ - [-1, 1, CBAM, [16, 7]]          # Modified: CBAM attention
```

### P2-Head Variant (Extra detection scale)
**Three lines added** to the head:
```diff
  # After P3 processing (layer 16), add P2:
+ - [-1, 1, nn.Upsample, [None, 2, "nearest"]]    # [17] Upsample to P2
+ - [[-1, 2], 1, Concat, [1]]                       # [18] Cat with backbone P2
+ - [-1, 2, C3k2, [128, False]]                     # [19] P2/4 detection features
```
And the Detect layer changes from 3-scale to 4-scale:
```diff
- - [[16, 19, 22], 1, Detect, [nc]]           # 3 scales: P3, P4, P5
+ - [[19, 22, 25, 28], 1, Detect, [nc]]       # 4 scales: P2, P3, P4, P5
```

---

## 8. How Ultralytics Parses the YAML (Code Flow)

```
1. You write:   YOLO("yolov11m_p2head.yaml")
2. Ultralytics reads the YAML file
3. It looks up the scale "m" → [0.50, 1.00, 512]
4. For each layer in backbone+head:
   a. Find the module class (Conv, C3k2, etc.)
   b. Apply scaling: channels × width, repeats × depth
   c. Build the PyTorch module
   d. Connect it using the "from" index
5. Stack all modules into model.model = nn.Sequential(...)
6. You get a working neural network!
```

> **This is why registering CBAM mattered**: When Ultralytics sees `CBAM` in the YAML, it looks for a class named `CBAM` in `ultralytics.nn.modules`. If we don't register it first, it crashes with "Unknown module: CBAM".

---

## Key Takeaways

| Concept | Summary |
|---|---|
| YAML is a text file | It defines every layer in the network |
| `from` field | Controls skip connections (which layers feed into which) |
| `scales` parameter | One YAML generates n/s/m/l/x model sizes |
| Changing one line | Can swap entire attention mechanisms (C2PSA → CBAM) |
| Adding lines | Can add new detection scales (P2 head) |
| Module registration | Custom modules must be registered before YAML parsing |

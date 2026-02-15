# YOLOv11m Architecture — From Scratch

> **Reading time:** ~20 minutes  
> **Prerequisite knowledge:** Basic understanding of neural networks (layers, weights)

---

## Table of Contents
1. [What is YOLO?](#1-what-is-yolo)
2. [The Big Picture: 3 Parts of YOLOv11m](#2-the-big-picture-3-parts-of-yolov11m)
3. [Part 1: The Backbone (Feature Extractor)](#3-part-1-the-backbone)
4. [Part 2: The Neck (Feature Combiner)](#4-part-2-the-neck)
5. [Part 3: The Head (Detector)](#5-part-3-the-head)
6. [Key Building Blocks Explained](#6-key-building-blocks)
7. [The Full Layer-by-Layer Walkthrough](#7-full-layer-walkthrough)
8. [Model Scales (n/s/m/l/x)](#8-model-scales)
9. [Parameters & What They Mean](#9-parameters)
10. [How Predictions Are Made](#10-predictions)

---

## 1. What is YOLO?

**YOLO = "You Only Look Once"**

Traditional object detectors look at an image multiple times (region proposals → classification). YOLO is different: it looks at the **entire image once** and predicts all bounding boxes and class labels in a single forward pass. This makes it extremely fast.

```
Traditional Detector:    Image → Find regions → Classify each region → Slow (2 passes)
YOLO:                    Image → Single neural network → All boxes + labels → Fast (1 pass)
```

**YOLOv11 (a.k.a YOLO11)** is the latest version released by Ultralytics in September 2024. The "m" in YOLOv11m stands for **"medium"** — it's one of 5 size variants.

---

## 2. The Big Picture: 3 Parts of YOLOv11m

Every YOLO model has three main sections:

```
┌──────────────────────────────────────────────────────────┐
│                    INPUT IMAGE (640×640)                  │
└──────────────────────┬───────────────────────────────────┘
                       │
           ┌───────────▼───────────┐
           │      BACKBONE         │  ← "Eyes" — Extracts features
           │  (Sees patterns)      │     from the raw image
           └───────────┬───────────┘
                       │
           ┌───────────▼───────────┐
           │        NECK           │  ← "Brain" — Combines features 
           │   (Mixes features)    │     from different scales
           └───────────┬───────────┘
                       │
           ┌───────────▼───────────┐
           │        HEAD           │  ← "Mouth" — Outputs predictions
           │  (Makes predictions)  │     (boxes, classes, confidence)
           └───────────────────────┘
```

---

## 3. Part 1: The Backbone

### What It Does
The backbone takes a raw image (640×640×3 pixels — height, width, RGB channels) and progressively extracts **features** — patterns like edges, textures, shapes, and objects.

### How It Works: Downsampling
The backbone reduces spatial size while increasing feature depth. For your **640×640 aerial images**:

```
Stage     Resolution    Channels    What it "sees" in C2A Dataset
──────────────────────────────────────────────────────────────────
Input     640 × 640     3 (RGB)     Raw drone image (houses, water)
P1/2      320 × 320     64          Edges of roofs, water lines
P2/4      160 × 160     128         Textures (tiles, waves)
P3/8       80 × 80      256         Small debris, limbs
P4/16      40 × 40      512         Whole objects (person on roof)
P5/32      20 × 20      512→1024    Scene context (flooded street)
```

> **Why this matters for you:** Your dataset has many small victims. Standard YOLO compresses everything to P5 (20×20). A small 5px person disappears by P5! This is why we are adding the **P2 head** later (to keep the 160×160 detail).

### The Backbone Layers (YOLOv11m)

```
Layer  Module    Output Size     What It Does
─────────────────────────────────────────────────────────────
[0]    Conv      320×320×64      First conv: 3×3 kernel, stride 2 → halves image
[1]    Conv      160×160×128     Second conv: stride 2 → halves again → P2/4
[2]    C3k2      160×160×256     Feature processing block (explained below)
[3]    Conv      80×80×256       Downsample → P3/8
[4]    C3k2      80×80×512       Feature processing
[5]    Conv      40×40×512       Downsample → P4/16
[6]    C3k2      40×40×512       Feature processing
[7]    Conv      20×20×1024      Downsample → P5/32
[8]    C3k2      20×20×1024      Feature processing
[9]    SPPF      20×20×1024      Multi-scale pooling
[10]   C2PSA     20×20×1024      Spatial attention (YOLO11's innovation)
```

### Analogy: The Backbone is Like a Funnel
```
640×640         ████████████████████████████████  (Lots of spatial detail)
320×320         ████████████████                  
160×160         ████████                          
 80×80          ████                              
 40×40          ██                                
 20×20          █  (Rich semantic meaning)
```
At the top: you can see every pixel. At the bottom: you understand what the image *means*.

---

## 4. Part 2: The Neck

### The Problem the Neck Solves
- **Deep layers (P5)** understand *what* objects are → but lost spatial detail
- **Shallow layers (P2)** have precise *where* information → but don't understand objects

The neck **combines both** by taking deep features and merging them with shallow features.

### How: Feature Pyramid Network (FPN) + PAN

```
                    BACKBONE OUTPUT (P5/32)
                           │
                    ┌──────▼──────┐
         ┌─────────│  Upsample   │  (Make bigger: 20→40)
         │         └─────────────┘
         │              │
         │    ┌─────────▼─────────┐
P4 ──────┼───→│   Concat + C3k2  │  Merge P4 features
         │    └─────────┬─────────┘
         │              │
         │    ┌─────────▼─────────┐
         │    │   Upsample        │  (Make bigger: 40→80)
         │    └─────────┬─────────┘
         │              │
         │    ┌─────────▼─────────┐
P3 ──────┼───→│   Concat + C3k2  │  Merge P3 features → DETECT P3
         │    └─────────┬─────────┘
         │              │
         │    ┌─────────▼─────────┐
         │    │   Downsample      │  (Make smaller: 80→40)
         │    └─────────┬─────────┘
         │              │
         │    ┌─────────▼─────────┐
         └───→│   Concat + C3k2  │  Merge back → DETECT P4
              └─────────┬─────────┘
                        │
              ┌─────────▼─────────┐
              │   Downsample      │  (Make smaller: 40→20)
              └─────────┬─────────┘
                        │
              ┌─────────▼─────────┐
              │   Concat + C3k2  │  Merge back → DETECT P5
              └─────────────────┘
```

This is called **PANet** (Path Aggregation Network): features flow both **top-down** (FPN) AND **bottom-up** (PAN).

---

## 5. Part 3: The Head

### What the Detect Layer Does
The `Detect` layer takes 3 feature maps (P3, P4, P5) and predicts:

| Output | Meaning |
|---|---|
| **Bounding box** (x, y, w, h) | Where the object is |
| **Confidence** | How sure the model is |
| **Class probabilities** | What the object is (person, car, etc.) |

### Multi-Scale Detection

```
P3 (80×80)  →  6,400 possible locations  →  Small objects  (people far away)
P4 (40×40)  →  1,600 possible locations  →  Medium objects (nearby people)
P5 (20×20)  →    400 possible locations  →  Large objects  (close-up)
                ─────
                8,400 total predictions → NMS filters to final boxes
```

---

## 6. Key Building Blocks

### Conv (Convolution Layer)
The most basic operation. A small "filter" (e.g., 3×3) slides over the image to detect patterns:

```python
Conv [channels_out, kernel_size, stride]
# Example: Conv [64, 3, 2]
#   - Output channels: 64
#   - Kernel size: 3×3
#   - Stride: 2 (skip every other pixel → halves the size)
```

**What happens inside:**
```
Input:  640×640×3     (3 = RGB color channels)
Filter: 3×3×3         (3×3 spatial, 3 input channels)
Stride: 2             (move filter 2 pixels at a time)
Output: 320×320×64    (halved spatial, 64 output channels)
```

A Conv layer = **Convolution** + **Batch Normalization** + **SiLU activation**:
- **Convolution**: Extracts patterns
- **Batch Normalization**: Stabilizes training (normalizes values)
- **SiLU**: Activation function (introduces non-linearity: `x * sigmoid(x)`)

### C3k2 (Cross Stage Partial with kernel-size 2)
YOLOv11's main feature processing block. It **splits** the feature map, processes one part, then **merges** both halves:

```
Input features
     │
     ├──────────┐
     │          │
  Process     Skip
  (Conv)     (Identity)
     │          │
     └────┬─────┘
          │
     Concatenate
          │
       Output
```

**Why split and merge?** It reduces computation by only processing HALF the features through expensive operations, while keeping the other half unchanged. This is the **CSP (Cross Stage Partial)** idea.

```python
C3k2 [channels_out, shortcut, ratio]
# Example: C3k2 [256, False, 0.25]
#   - 256 output channels
#   - False = no residual shortcut inside bottlenecks
#   - 0.25 = only 25% of channels go through the "heavy" path
```

### SPPF (Spatial Pyramid Pooling Fast)
Captures features at **multiple scales** without resizing the image:

```
Input → MaxPool(5×5) → MaxPool(5×5) → MaxPool(5×5)
  │          │              │              │
  └──────────┴──────────────┴──────────────┘
                    Concatenate
                        │
                      Conv
```

**Why?** A 5×5 pool sees local context. Two stacked 5×5 pools effectively see 9×9. Three see 13×13. This gives the model both local and global understanding.

### C2PSA (Cross Stage Partial with Spatial Attention)
**YOLOv11's key innovation** — replaces C2f from YOLOv8. It adds **attention**: the model learns to focus on the most important spatial regions.

```
Input
  │
  ├──────────┐
  │          │
  PSA       Skip
(Attention)   │
  │          │
  └────┬─────┘
  Concatenate
       │
    Output
```

**PSA = Parallel Spatial Attention**: Uses multi-head attention (similar to Transformers) to let the model "pay attention" to the most relevant parts of the feature map.

### nn.Upsample
Makes a feature map **bigger** (opposite of Conv with stride 2):

```python
nn.Upsample [None, 2, "nearest"]
# scale_factor=2, mode="nearest"
# 20×20 → 40×40 (doubles spatial dimensions)
```

**How "nearest" upsampling works:**
```
Before (2×2):    After (4×4):
┌───┬───┐        ┌───┬───┬───┬───┐
│ 1 │ 2 │        │ 1 │ 1 │ 2 │ 2 │
├───┼───┤   →    ├───┼───┼───┼───┤
│ 3 │ 4 │        │ 1 │ 1 │ 2 │ 2 │
└───┴───┘        ├───┼───┼───┼───┤
                 │ 3 │ 3 │ 4 │ 4 │
                 ├───┼───┼───┼───┤
                 │ 3 │ 3 │ 4 │ 4 │
                 └───┴───┴───┴───┘
```
Each pixel is simply duplicated to fill a 2×2 area. Simple and fast!

### Concat
Joins two feature maps along the **channel dimension**:

```
Feature A:  40×40×512
Feature B:  40×40×512
                         → Concat → 40×40×1024
```

> ⚠️ **Spatial sizes must match!** You can't concat 40×40 with 20×20. That's why we upsample first.

---

## 7. Full Layer-by-Layer Walkthrough

Here is the **complete YOLOv11m standard architecture** (3-scale detection):

```
BACKBONE:
──────────────────────────────────────────────────────────────
[0]  Conv [64, 3, 2]        640×640×3   → 320×320×64     P1/2
[1]  Conv [128, 3, 2]       320×320×64  → 160×160×128    P2/4
[2]  C3k2 [256, F, 0.25]    160×160×128 → 160×160×256
[3]  Conv [256, 3, 2]       160×160×256 → 80×80×256      P3/8
[4]  C3k2 [512, F, 0.25]    80×80×256   → 80×80×512
[5]  Conv [512, 3, 2]       80×80×512   → 40×40×512      P4/16
[6]  C3k2 [512, T]          40×40×512   → 40×40×512
[7]  Conv [1024, 3, 2]      40×40×512   → 20×20×1024     P5/32
[8]  C3k2 [1024, T]         20×20×1024  → 20×20×1024
[9]  SPPF [1024, 5]         20×20×1024  → 20×20×1024
[10] C2PSA [1024]           20×20×1024  → 20×20×1024

HEAD (Neck + Detect):
──────────────────────────────────────────────────────────────
[11] Upsample ×2            20×20  → 40×40                FPN ↑
[12] Concat [layer 6]       40×40×(1024+512) = 40×40×1536
[13] C3k2 [512]             40×40×1536 → 40×40×512

[14] Upsample ×2            40×40 → 80×80                 FPN ↑
[15] Concat [layer 4]       80×80×(512+512) = 80×80×1024
[16] C3k2 [256]             80×80×1024 → 80×80×256        ← P3 output

[17] Conv [256, 3, 2]       80×80 → 40×40                 PAN ↓
[18] Concat [layer 13]      40×40×(256+512) = 40×40×768
[19] C3k2 [512]             40×40×768 → 40×40×512         ← P4 output

[20] Conv [512, 3, 2]       40×40 → 20×20                 PAN ↓
[21] Concat [layer 10]      20×20×(512+1024) = 20×20×1536
[22] C3k2 [1024]            20×20×1536 → 20×20×1024       ← P5 output

[23] Detect [P3, P4, P5]    Predictions from 3 scales
```

---

## 8. Model Scales

YOLOv11 comes in 5 sizes, all from the **same YAML** with different scaling:

| Scale | Depth | Width | Max Channels | Params | Speed |
|---|---|---|---|---|---|
| **n** (nano) | 0.50 | 0.25 | 1024 | ~2.6M | Fastest |
| **s** (small) | 0.50 | 0.50 | 1024 | ~9.4M | Fast |
| **m** (medium) | 0.50 | 1.00 | 512 | ~20M | Balanced |
| **l** (large) | 1.00 | 1.00 | 512 | ~25M | Accurate |
| **x** (extra-large) | 1.00 | 1.50 | 512 | ~57M | Most accurate |

### What `depth` and `width` do:

```yaml
scales:
  m: [0.50, 1.00, 512]  # [depth_multiple, width_multiple, max_channels]
```

- **depth = 0.50**: If a layer says "repeat 2 times", it becomes `ceil(2 × 0.50) = 1` time
- **width = 1.00**: If a layer says "256 channels", it stays `256 × 1.00 = 256` channels  
- **max_channels = 512**: No layer can exceed 512 channels (saves memory)

---

## 9. Parameters & What They Mean

### Total Parameters: ~20M for YOLOv11m
A parameter is a single learnable number (weight) in the neural network. More parameters = more capacity to learn, but slower.

| Component | Approx. Params | % of Total |
|---|---|---|
| Backbone | ~10M | 50% |
| Neck | ~7M | 35% |
| Detect Head | ~3M | 15% |

### Key Training Parameters
| Parameter | What It Does | Our Value |
|---|---|---|
| `epochs` | Number of full passes over the dataset | 70 |
| `batch` | Images processed simultaneously | 8-16 |
| `imgsz` | Input image resolution | 640 |
| `patience` | Stop early if no improvement for N epochs | 10 |
| `lr0` | Initial learning rate | 0.01 (auto) |
| `amp` | Mixed precision (FP16) — saves GPU memory | True |
| `device` | GPU to use | 0 (single GPU) |

---

## 10. How Predictions Are Made

### Step 1: Forward Pass
Image (640×640) → Backbone → Neck → 3 feature maps (P3, P4, P5)

### Step 2: Dense Predictions
Each cell in each feature map predicts:
- 4 values: bounding box (x, y, w, h)
- 1 value: objectness confidence 
- N values: class probabilities (N = number of classes)

### Step 3: Non-Maximum Suppression (NMS)
Thousands of raw predictions → filter overlapping boxes → keep the best ones:

```
Raw predictions:  ~8,400 boxes
After confidence filter (>0.25):  ~200 boxes
After NMS (remove overlaps):      ~15 final detections
```

---

## Summary

```
YOLOv11m at a glance (ACTUAL from our runs):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:       640×640×3 (RGB image)
Backbone:    11 layers (Conv + C3k2 + SPPF + C2PSA)
Neck:        12 layers (FPN upsampling + PAN downsampling)
Head:        Detect on 3 scales (P3, P4, P5)
Parameters:  20,053,779 (20.05M)
GFLOPs:      34.1
Layers:      410
FPS @ 640:   32.6 (Tesla T4)
mAP@0.5:     0.8558 (on C2A test set)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

> **Full Results:** See [docs/08_results_analysis.md](08_results_analysis.md) for the 3-way comparison with CBAM and P2 variants.

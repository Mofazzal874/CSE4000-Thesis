# YOLO11m Baseline vs Mamba-Based YOLO: Architecture & Performance Comparison
## Date: 2026-03-30

---

# 1. YOLO11m Baseline — How It Works

## 1.1 Overall Architecture

YOLO11m follows the standard **Backbone → Neck → Head** paradigm:

```
Input (640x640x3)
    │
    ▼
┌─────────────────────────────────────┐
│  BACKBONE (Feature Extraction)       │
│  Conv → C3k2 → C3k2 → C3k2 → C3k2  │
│  Channels: 64 → 128 → 256 → 512     │
│       │         │                    │
│       ▼         ▼                    │
│     SPPF (Spatial Pyramid Pooling)   │
│       │                              │
│       ▼                              │
│     C2PSA (Attention Block) ← KEY    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  NECK (Feature Fusion — FPN + PAN)   │
│  Top-down: Upsample+Concat+C3k2     │
│  Bottom-up: Conv+Concat+C3k2        │
│  Produces P3, P4, P5 feature maps   │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  HEAD (Detection — 3 scales)         │
│  P3: 80x80  (stride 8)  — small     │
│  P4: 40x40  (stride 16) — medium    │
│  P5: 20x20  (stride 32) — large     │
└─────────────────────────────────────┘
```

## 1.2 C2PSA — The Attention Block in YOLO11m

**C2PSA = Cross Stage Partial with Spatial Attention.** This is the key attention mechanism that distinguishes YOLO11 from YOLOv8.

### How C2PSA works:
1. Input features are split into two branches (CSP-style)
2. One branch passes through a **multi-head spatial attention** mechanism:
   - Generates Query (Q), Key (K), Value (V) matrices from feature maps
   - Computes attention weights: `Attention(Q, K, V) = softmax(QKᵀ / √d) · V`
   - This is essentially a **self-attention mechanism** (transformer-style) applied to spatial feature positions
3. The attended branch is concatenated with the identity branch
4. A final convolution fuses the output

### Key characteristics:
- **Quadratic complexity:** Attention computation is O(N²) where N = H×W (spatial positions). For a 20×20 feature map, N=400 → 160K attention pairs. Manageable, but scales poorly to larger maps.
- **Global receptive field:** Every spatial position can attend to every other position in one step
- **Content-aware:** Attention weights depend on the actual feature values (what's in the image)
- **Position in YOLO11m:** Placed at the end of the backbone (after SPPF, before neck), operating only on the deepest 20×20 feature map

## 1.3 C3k2 — The Backbone/Neck Building Block

**C3k2 = CSP Bottleneck with Kernel size 2.**

- Uses **two smaller convolutions** instead of one large convolution (compared to C2f in YOLOv8)
- When `c3k=False`: acts like a standard C2f bottleneck
- When `c3k=True`: replaces bottleneck with a C3 module for deeper feature extraction
- More parameter-efficient than YOLOv8's C2f blocks

---

# 2. Mamba and State Space Models — How They Work

## 2.1 The State Space Model (SSM) Framework

State Space Models come from control theory. They model a system with a hidden state that evolves over time:

```
Continuous form:
    h'(t) = A·h(t) + B·x(t)     ← state equation
    y(t)  = C·h(t) + D·x(t)     ← output equation

Where:
    x(t) = input at time t
    h(t) = hidden state (compressed memory)
    y(t) = output
    A    = state transition matrix (how state evolves)
    B    = input projection matrix (how input affects state)
    C    = output projection matrix (how state maps to output)
    D    = skip connection (direct input-to-output)
```

### Discretization (making it work on digital data):

The continuous equations are converted to discrete steps using **Zero-Order Hold (ZOH)** with a learnable step size Δ (Delta):

```
Discrete form:
    Ā = exp(Δ·A)
    B̄ = (Δ·A)⁻¹ · (Ā - I) · Δ·B

    h[k] = Ā·h[k-1] + B̄·x[k]
    y[k] = C·h[k] + D·x[k]
```

This discrete form enables **dual computation modes**:
- **Recurrent mode** (inference): Process one token at a time, O(1) per step, constant memory
- **Convolutional mode** (training): Process all tokens in parallel using a convolution kernel

## 2.2 The Problem with Traditional SSMs

Traditional SSMs (S4, S5) use **fixed, time-invariant** matrices A, B, C. The same transformation is applied regardless of input content. This means:
- They cannot selectively remember or forget based on what they see
- They act as a fixed linear filter — great for periodic signals, bad for content-aware reasoning

## 2.3 Mamba (S6) — The Selective State Space Model

Mamba's key innovation: **make B, C, and Δ input-dependent**.

```
Traditional SSM:          Mamba (S6):
    B = fixed parameter       B = Linear(x)    ← depends on input
    C = fixed parameter       C = Linear(x)    ← depends on input
    Δ = fixed parameter       Δ = softplus(Linear(x))  ← depends on input
```

### What this enables:

The model can now **selectively filter** information:

```
Large Δ → "Remember this input" (state absorbs current token)
Small Δ → "Ignore this input" (state passes through unchanged)

B controls "how" to write to memory
C controls "what" to read from memory
```

This is analogous to the **gating mechanism** in LSTMs, but more efficient.

### Mamba Block Architecture:

```
Input (B, L, D)
    │
    ├──────────────────────┐
    ▼                      ▼
  Linear (expand)        Linear
    │                      │
    ▼                      │
  Conv1D (local context)   │
    │                      │
    ▼                      │
  SiLU activation          │
    │                      │
    ▼                      │
  SSM (selective scan) ←───┘ (gating via SiLU)
    │
    ▼
  Linear (project down)
    │
    ▼
Output (B, L, D)
```

## 2.4 Hardware-Aware Optimizations

Mamba achieves speed through three GPU optimization techniques:
1. **Kernel fusion:** Discretization + selective scan + multiplication combined into one GPU kernel (avoids slow DRAM round-trips)
2. **Recomputation:** Backward pass recomputes intermediate states instead of storing them (trades compute for memory)
3. **Parallel scan:** Training uses a parallel prefix-sum algorithm, enabling O(L) parallel computation

---

# 3. Head-to-Head: Attention (C2PSA) vs SSM (Mamba)

## 3.1 Fundamental Mechanism Comparison

| Aspect | C2PSA (Attention) | Mamba (SSM) |
|--------|-------------------|-------------|
| **Core operation** | Q·Kᵀ → softmax → weighted sum of V | Recurrent scan with selective gating |
| **Complexity** | **O(N²)** — quadratic in spatial tokens | **O(N)** — linear in spatial tokens |
| **How it "remembers"** | Full attention matrix (every token sees every token) | Compressed hidden state (fixed-size memory) |
| **Information flow** | Bidirectional (all-to-all in one step) | Sequential (forward scan, or bidirectional with 2 passes) |
| **Content-awareness** | Inherent (Q, K, V are input-dependent) | Only in Mamba/S6 (B, C, Δ are input-dependent) |
| **Memory at inference** | Stores full KV cache, grows with sequence | Fixed-size hidden state, constant |
| **Receptive field** | Global (but expensive) | Theoretically unbounded (through state accumulation) |
| **Parallelism** | Fully parallel (matrix multiply) | Training: parallel scan; Inference: recurrent |

## 3.2 Visual Intuition

```
ATTENTION (C2PSA):                    SSM (Mamba):
Every token talks to every token      Information flows through a compressed state

  T1 ←→ T2 ←→ T3 ←→ T4               T1 → [h1] → T2 → [h2] → T3 → [h3] → T4
  ↕    ↕    ↕    ↕                     ←─────────────────────────────────────
  T5 ←→ T6 ←→ T7 ←→ T8               T4 → [h4] → T3 → [h5] → T2 → [h6] → T1
  ↕    ↕    ↕    ↕                           (bidirectional = 2 scans)
  T9 ←→ T10←→ T11←→ T12

  N² connections                       2×N sequential steps
  (expensive but direct)               (cheap but indirect)
```

## 3.3 For Object Detection Specifically

| Factor | Attention Advantage | SSM/Mamba Advantage |
|--------|--------------------|--------------------|
| Small objects | Direct long-range attention helps relate small objects to global context | Linear cost enables higher-resolution feature maps (P2) without memory explosion |
| Dense scenes | All objects "see" each other simultaneously | Sequential scan can still aggregate context, but indirect |
| High-res feature maps | Prohibitively expensive: 160×160 = 25,600 tokens → 655M attention pairs | Linear: O(25,600) per channel — feasible |
| Parameter efficiency | Requires Q, K, V projection matrices | SSM matrices can be very compact (d_state=16 or even 4) |
| Inference speed | Cached but memory-heavy | Constant memory, recurrent step |

---

# 4. Architecture Differences: Our Models

## 4.1 What We Changed and Where

```
YOLO11m Baseline:
    Backbone: Conv → C3k2 → C3k2 → C3k2 → C3k2 → SPPF → [C2PSA] ← Attention here
    Neck:     C3k2 (FPN + PAN feature fusion)
    Head:     3-scale (P3, P4, P5)

Our Mamba+CBAM+P2Head:
    Backbone: Conv → C3k2 → C3k2 → C3k2 → C3k2 → SPPF → [CBAM] ← Replaced attention
    Neck:     [C3K2Mamba] (SSM replaces C3k2 at 6 neck layers) ← Mamba here
    Head:     4-scale (P2, P3, P4, P5) ← Added P2
```

### Key design decisions:
1. **CBAM replaces C2PSA in the backbone** — simpler channel+spatial attention, reduces params by 2.3%
2. **Mamba replaces C3k2 in the neck** — SSM-based feature fusion instead of convolutional bottlenecks
3. **P2 Head added** — 4th detection scale at 160×160 (stride 4) for tiny objects (as small as 4px)

## 4.2 How the Mamba Neck Works (C3K2Mamba)

In our implementation, the C3k2 bottleneck's inner convolutions are replaced with a bidirectional SSM block:

```
Standard C3k2 bottleneck:          Our C3K2Mamba bottleneck:
    Input                              Input
      │                                  │
    Conv 3×3                           Conv 1×1 (project)
      │                                  │
    Conv 3×3                           Flatten 2D → 1D sequence
      │                                  │
    Output                             SSM Forward Scan →
                                       SSM Backward Scan ←
                                         │
                                       Reshape 1D → 2D
                                         │
                                       Output
```

The 2D feature map is flattened into a 1D sequence (raster scan order), processed bidirectionally by the selective SSM, then reshaped back to 2D. This allows the neck to aggregate features using learned sequential dependencies rather than fixed-size convolution kernels.

## 4.3 AtrousSSM Variant (Dilated Mamba)

The AtrousMamba variant adds multi-scale scanning with dilated windows:

```
Input sequence: [t1, t2, t3, t4, t5, t6, t7, t8, ...]

Branch 1 (d=1): [t1, t2, t3, t4, t5, t6, t7, t8] — local detail
Branch 2 (d=2): [t1, t3, t5, t7, ...] — medium context
Branch 3 (d=4): [t1, t5, t9, t13, ...] — wide context

Each branch runs bidirectional SSM scan, then outputs are
combined via learned gated fusion:

    gate = σ(W · [branch1; branch2; branch3])
    output = gate ⊙ branch1 + (1-gate) ⊙ branch2 + ... (gated sum)
```

This is inspired by atrous/dilated convolutions and EfficientVMamba (AAAI 2025), which uses atrous selective scan for lightweight visual Mamba.

---

# 5. Performance Comparison

## 5.1 Detection Results (Test Set — VisDrone Aerial Dataset)

| Model | mAP50 | mAP50-95 | Precision | Recall | Params |
|-------|-------|----------|-----------|--------|--------|
| YOLO11m Baseline | 0.8558 | 0.6256 | 0.8257 | 0.8488 | 20.05M |
| + CBAM + P2Head | 0.8723 (+1.93%) | 0.6418 (+2.59%) | 0.8351 | **0.8619** | 19.59M |
| + Mamba Neck | **0.8770 (+2.47%)** | **0.6539 (+4.53%)** | **0.8547** | 0.8453 | 19.59M |
| + AtrousMamba Neck | 0.8228 (-3.86%) | 0.5540 (-11.4%) | 0.8127 | 0.7979 | 24.16M |

### Key observations:
- **Mamba neck achieves the best mAP50 (0.877) and mAP50-95 (0.654)** with zero parameter overhead vs CBAM+P2
- **+4.53% mAP50-95 over baseline** — significant improvement in localization quality
- **CBAM+P2 has the highest recall** (0.8619) — attention helps detect more objects
- **Mamba has the highest precision** (0.8547) — fewer false positives
- **AtrousMamba underperforms** — likely under-trained at 80 epochs (it was still improving) and +20% more parameters to learn

## 5.2 Small Object Performance (Test Set)

| Model | Very Tiny (<8px) | Tiny (8-16px) | Small (16-32px) | Medium (32+px) |
|-------|-----------------|---------------|-----------------|----------------|
| Baseline | 78.39% | 89.01% | 88.07% | 100% |
| + CBAM + P2 | **80.89%** | **89.72%** | 88.64% | 100% |
| + Mamba | 76.68% | 87.52% | **89.60%** | 89.59% |
| + AtrousMamba | 68.31% | 84.17% | 87.09% | 93.69% |

### Observations:
- **CBAM+P2 excels at very tiny objects** — direct spatial attention + P2 resolution helps the most for < 16px objects
- **Mamba leads on small objects (16-32px)** — SSM context aggregation works best at this scale
- Mamba's slight drop on very tiny objects suggests that direct attention (CBAM) is more effective than sequential scanning for the smallest objects

## 5.3 Efficiency Comparison

| Model | Params | GFLOPs | Delta Params | Delta GFLOPs |
|-------|--------|--------|--------------|--------------|
| Baseline | 20.05M | 34.1 | — | — |
| + CBAM + P2 | 19.59M | 43.7 | -2.30% | +28.2% |
| + Mamba | 19.59M | 43.7 | -2.30% | +28.2% |
| + AtrousMamba | 24.16M | 48.1 | +20.45% | +41.1% |

The Mamba neck adds **zero additional parameters** over the CBAM+P2 base. The SSM replaces the convolution bottleneck weights 1:1. The GFLOPs increase comes entirely from the P2 head (more detection grid cells), not from Mamba.

---

# 6. Literature Context

## 6.1 Mamba in Vision — Key Papers

| Paper | Venue | Key Finding |
|-------|-------|-------------|
| **VMamba** | NeurIPS 2024 | VMamba-T achieves 47.3 box mAP on COCO (+4.6 over Swin Transformer) with Mask R-CNN |
| **Vision Mamba (Vim)** | 2024 | 2.8× faster than DeiT, 86.8% less GPU memory at high resolution. Higher mAP than DeiT on COCO. |
| **EfficientVMamba** | AAAI 2025 | Atrous (dilated) selective scan for lightweight visual Mamba — directly related to our AtrousSSM approach |
| **MambaVision** | CVPR 2025 (NVIDIA) | Hybrid Mamba+Transformer outperforms pure Mamba. Best results: Mamba in early layers, attention in final layers. |

## 6.2 Consensus from Literature

> **Pure Mamba backbones outperform pure transformer backbones of equivalent size on detection tasks** (VMamba vs Swin, Vim vs DeiT). However, **hybrid Mamba+Attention models outperform both** (MambaVision).

Our architecture follows this hybrid philosophy:
- **CBAM (attention)** in the backbone for spatial/channel focus
- **Mamba (SSM)** in the neck for efficient feature fusion
- This separation lets each mechanism operate where it's most effective

---

# 7. Summary — Why Mamba Works for Our Task

| Question | Answer |
|----------|--------|
| Why replace C2PSA with CBAM? | CBAM is simpler, reduces params by 2.3%, and provides both channel + spatial attention. C2PSA's transformer-style attention on a 20×20 map isn't the bottleneck. |
| Why add Mamba in the neck? | The neck fuses multi-scale features — this is where sequential context aggregation (SSM) adds value. Mamba provides learned long-range dependencies without the quadratic cost of attention. |
| Why does Mamba+CBAM+P2 win? | Best of both worlds: attention (CBAM) decides "where to look" in the backbone, SSM (Mamba) efficiently aggregates features in the neck, and P2 provides high-resolution detection. Zero parameter overhead. |
| Why did AtrousMamba underperform? | +20% more parameters needed more training. At 80 epochs (vs 120 for Mamba), it was still improving. The dilated scanning concept is sound (validated by EfficientVMamba, AAAI 2025) but needs more training budget. |

---

## Sources

- Maarten Grootendorst, [A Visual Guide to Mamba and State Space Models](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state)
- Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
- Liu et al., "VMamba: Visual State Space Model" (NeurIPS 2024)
- Zhu et al., "Vision Mamba: Efficient Visual Representation Learning with Bidirectional SSM" (2024)
- Hatamizadeh & Kautz, "MambaVision: A Hybrid Mamba-Transformer Vision Backbone" (CVPR 2025)
- Pei et al., "EfficientVMamba: Atrous Selective Scan for Light Weight Visual Mamba" (AAAI 2025)
- Ultralytics, [YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11/)
- Jocher & Qiu, "YOLO11: An Overview of the Key Architectural Enhancements" (arXiv 2410.17725)

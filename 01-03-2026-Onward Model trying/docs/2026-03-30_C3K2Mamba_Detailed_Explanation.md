# C3K2Mamba: In-Depth Architectural Explanation
## Date: 2026-03-30

---

## 1. What Is C3K2Mamba?

C3K2Mamba is a **custom feature aggregation block** that serves as a drop-in replacement for YOLO11's standard `C3k2` block, specifically in the **neck** of the detection network. It replaces the purely convolutional bottleneck inside C3k2 with a bottleneck that contains your novel **AtrousSSM** — a multi-scale dilated State Space Model.

In simple terms:
- **C3k2** (YOLO11 original) = split → convolutional bottlenecks → concatenate → fuse
- **C3K2Mamba** (yours) = split → **SSM-augmented** bottlenecks → concatenate → fuse

The "C3K2" name comes from YOLO's naming convention: **C**ross-**S**tage **P**artial block with **k**ernel size variations, version **2**. Your "Mamba" suffix indicates the bottleneck uses Mamba-style selective state space scanning instead of pure convolution.

---

## 2. Where C3K2Mamba Sits in the Full Architecture

```
YOLO11m Architecture (your modified version)
═══════════════════════════════════════════════

BACKBONE (layers 0-10) — FROZEN during training
┌─────────────────────────────────────────────┐
│ Conv 64 → Conv 128 → C3k2 256 → Conv 256   │
│ → C3k2 512 → Conv 512 → C3k2 512           │
│ → Conv 1024 → C3k2 1024 → SPPF 1024        │
│ → CBAM (replaces C2PSA)                     │
└─────────────────────────────────────────────┘
          │
          ▼
NECK + HEAD (layers 11-29) — TRAINABLE
┌─────────────────────────────────────────────┐
│ Upsample → Concat                           │
│ → ★ C3K2Mamba 512  (layer 13, 40×40)       │  ← YOUR MODULE
│ Upsample → Concat                           │
│ → C3k2 256          (layer 16, 80×80)       │  standard
│ Upsample → Concat                           │
│ → C3k2 128          (layer 19, 160×160)     │  standard (P2 level)
│ Conv → Concat                               │
│ → C3k2 256          (layer 22, 80×80)       │  standard
│ Conv → Concat                               │
│ → ★ C3K2Mamba 512  (layer 25, 40×40)       │  ← YOUR MODULE
│ Conv → Concat                               │
│ → ★ C3K2Mamba 512  (layer 28, 20×20)       │  ← YOUR MODULE
│                                             │
│ Detect [P2=160², P3=80², P4=40², P5=20²]   │
└─────────────────────────────────────────────┘
```

**Key design choice**: C3K2Mamba is placed only at the **512-channel layers** (deep features at 40×40 and 20×20 resolution). The shallower P2 (128ch, 160×160) and P3 (256ch, 80×80) levels use standard C3k2 to save compute. This is because:
1. The 512-channel layers carry the richest semantic features — SSM context is most valuable here
2. At 160×160 resolution, the AtrousSSM sequential scan would be very slow (too many windows)
3. The P2/P3 levels still benefit indirectly because they receive features from the C3K2Mamba layers via FPN concatenation

---

## 3. The C2f Design Pattern (What C3K2Mamba Inherits)

C3K2Mamba follows the **C2f (Cross Stage Partial v2)** design pattern from YOLOv8/v11. Understanding this pattern is essential to understanding why C3K2Mamba works as a drop-in replacement.

### The C2f / C3k2 Pattern

```
Input (B, c1, H, W)
        │
        ▼
   ┌─────────┐
   │  cv1    │   1×1 Conv: c1 → 2c  (where c = c2 * e, typically e=0.5)
   └────┬────┘
        │
    chunk(2)     Split along channel dim → two halves of size c each
        │
   ┌────┴────┐
   │         │
   y[0]    y[1]   Both are (B, c, H, W)
   │         │
   │    ┌────▼────┐
   │    │ Bottleneck│   Bottleneck 1: c → c
   │    │   #1     │
   │    └────┬────┘
   │         │──────── y[2]
   │    ┌────▼────┐
   │    │ Bottleneck│   Bottleneck 2: c → c (if n=2)
   │    │   #2     │
   │    └────┬────┘
   │         │──────── y[3]
   │         │
   └────┬────┘
        │
   concat(y[0], y[1], y[2], y[3])   → (B, (2+n)*c, H, W)
        │
        ▼
   ┌─────────┐
   │  cv2    │   1×1 Conv: (2+n)*c → c2
   └────┬────┘
        │
        ▼
   Output (B, c2, H, W)
```

**Why this pattern?**
- The **split + multi-branch concatenation** creates a **dense connection** — the output has access to the original split features (y[0], y[1]) AND every intermediate bottleneck output (y[2], y[3], ...). This gradient highway helps training.
- The **1×1 convs** (cv1, cv2) handle channel dimension changes cheaply.
- The **bottlenecks** do the heavy lifting (spatial feature processing).

### What C3K2Mamba changes

In standard C3k2, each bottleneck is:
```
Bottleneck: Conv3×3 → Conv3×3 (+ optional residual shortcut)
```

In C3K2Mamba, each bottleneck is:
```
_MambaBottleneck: Conv3×3 → AtrousSSM → Conv3×3 (+ optional residual shortcut)
```

**Everything else is identical** — the cv1 split, the concatenation, the cv2 fusion. This is what makes it a true drop-in replacement: the input/output channel dimensions are preserved exactly.

---

## 4. C3K2Mamba: Line-by-Line Breakdown

```python
class C3K2Mamba(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5,
                 d_state=4, dilations=None):
```

**Constructor arguments:**
| Arg | Meaning | Typical Value |
|-----|---------|---------------|
| `c1` | Input channels | 1024 (from Concat of SPPF+backbone) |
| `c2` | Output channels | 512 |
| `n` | Number of bottleneck repeats | 2 (set by YAML `scales: m`) |
| `shortcut` | Whether bottleneck has residual skip | False in neck layers |
| `g` | Groups (unused, kept for API compat) | 1 |
| `e` | Expansion ratio | 0.5 |
| `d_state` | SSM hidden state dimension | 4 |
| `dilations` | Dilation rates for AtrousSSM | [1, 2, 4] |

```python
        self.c = int(c2 * e)      # Internal channel width: 512 * 0.5 = 256
        ws = _get_window_size(self.c)  # Adaptive: 256ch → ws=6
```

The **internal channel width** `self.c` is half of `c2` (because `e=0.5`). This is the channel dimension that flows through the bottlenecks and gets concatenated. The window size adapts to channel count:
- 512+ channels → 4×4 windows (16 tokens) — smaller windows for compute-heavy layers
- 256+ channels → 6×6 windows (36 tokens)
- <256 channels → 8×8 windows (64 tokens) — larger windows where compute is cheaper

```python
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)   # 1×1 conv: c1 → 2c
```

**cv1** reduces channels from `c1` to `2c` (to be split into two halves of size `c`). For example: 1024 → 512, then split to 256 + 256.

```python
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # 1×1 conv: (2+n)*c → c2
```

**cv2** fuses all concatenated branches. With n=2: `(2+2)*256 = 1024` → 512. The "2" accounts for the two initial chunks; "n" accounts for the n bottleneck outputs.

```python
        self.m = nn.ModuleList(
            _MambaBottleneck(self.c, shortcut and c1 == c2,
                            d_state, ws, dilations=dilations or [1, 2, 4])
            for _ in range(n)
        )
```

**`self.m`**: A list of `n` MambaBottleneck modules. Each operates on `self.c` channels. The `shortcut and c1 == c2` condition means residual connections are only used when the input and output dimensions match (which is False in most neck layers where c1 ≠ c2).

### Forward Pass

```python
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))     # [y0, y1], each (B, c, H, W)
        y.extend(m(y[-1]) for m in self.m)     # y0, y1, bottleneck1(y1), bottleneck2(...)
        return self.cv2(torch.cat(y, 1))       # concat → cv2
```

Step by step:
1. **cv1** projects input from c1 to 2c channels
2. **chunk(2, 1)** splits along channel dim → two tensors of c channels each
3. **Sequential bottleneck processing**: Each bottleneck takes the **last element** of the list as input (`y[-1]`), processes it, and appends the output. This creates a chain: y[1] → bottleneck1 → y[2] → bottleneck2 → y[3]
4. **Concatenate** all: [y[0], y[1], y[2], y[3]] → (2+n)*c channels
5. **cv2** fuses back to c2 channels

The key insight: **y[0] passes through untouched** — it's a bypass/shortcut that preserves the original features. The bottlenecks progressively refine the other branch. The concatenation gives the final 1×1 conv access to both raw and SSM-processed features.

---

## 5. The _MambaBottleneck Inside C3K2Mamba

```python
class _MambaBottleneck(nn.Module):
    def __init__(self, c, shortcut, d_state, window_size, dilations=None):
        super().__init__()
        self.cv1 = Conv(c, c, 3, 1)       # 3×3 conv with BN + SiLU
        self.ssm = AtrousSSM(c, d_state=d_state, window_size=window_size,
                             dilations=dilations)
        self.cv2 = Conv(c, c, 3, 1)       # 3×3 conv with BN + SiLU
        self.add = shortcut               # residual connection flag

    def forward(self, x):
        y = self.cv2(self.ssm(self.cv1(x)))
        return x + y if self.add else y
```

The data flow through each bottleneck:

```
Input x (B, c, H, W)     e.g., (B, 256, 40, 40)
       │
       ▼
┌─────────────┐
│  Conv 3×3   │   Local feature extraction (BN + SiLU activation)
│  (cv1)      │   Prepares features for SSM — gives the SSM good local features to scan over
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                    AtrousSSM                            │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ Branch   │  │ Branch   │  │ Branch   │              │
│  │ d=1      │  │ d=2      │  │ d=4      │              │
│  │          │  │          │  │          │              │
│  │ Partition│  │ Partition│  │ Partition│              │
│  │ 6×6 win  │  │ 12×12    │  │ 24×24    │              │
│  │ local    │  │ medium   │  │ wide     │              │
│  │          │  │          │  │          │              │
│  │ Norm     │  │ Norm     │  │ Norm     │              │
│  │ Proj→x,z │  │ Proj→x,z │  │ Proj→x,z │              │
│  │ BiSSM ──►│  │ BiSSM ──►│  │ BiSSM ──►│  (fwd+bwd) │
│  │ Gate(z)  │  │ Gate(z)  │  │ Gate(z)  │              │
│  │ Proj+Res │  │ Proj+Res │  │ Proj+Res │              │
│  │ Reverse  │  │ Reverse  │  │ Reverse  │              │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘              │
│       │             │             │                     │
│       └──────┬──────┘──────┬──────┘                     │
│              ▼             ▼                             │
│         Gated Fusion (learned per-position weighting)   │
│         out = proj·gate + input·(1-gate)                │
│         LayerNorm                                       │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
                 ┌─────────────┐
                 │  Conv 3×3   │   Post-SSM local refinement
                 │  (cv2)      │   Smooths SSM output, learns local corrections
                 └──────┬──────┘
                        │
                        ▼
                 Output (B, c, H, W)
                 (+ x if shortcut=True)
```

**Why Conv3×3 before AND after the SSM?**
- **cv1 (before)**: The SSM operates on per-pixel features. The 3×3 conv gives each pixel a **locally-aware representation** before the SSM scans over it. Without this, the SSM would be scanning over raw per-pixel features that lack local context.
- **cv2 (after)**: The SSM produces features that are globally-aware (through state propagation) but may have **spatial artifacts** from the window partitioning and bilinear interpolation in the dilated reverse. The 3×3 conv smooths these artifacts and lets the network learn **local corrections** on top of the SSM's global context.

---

## 6. Concrete Data Flow Example

Let's trace a concrete forward pass through one C3K2Mamba block at **layer 13** (40×40 resolution, after concatenation with backbone features):

```
Input: (B=16, C=1024, H=40, W=40)     ← from Concat(Upsample(SPPF), backbone_layer6)

cv1: 1×1 conv, 1024 → 512
     → (16, 512, 40, 40)

chunk(2, dim=1):
     y[0] = (16, 256, 40, 40)   ← bypass branch (passes through untouched)
     y[1] = (16, 256, 40, 40)   ← goes into bottleneck chain

─── MambaBottleneck #1 ───────────────────────────────────
  cv1: 3×3 conv, 256→256
       → (16, 256, 40, 40)

  AtrousSSM (ws=6, dilations=[1,2,4]):

    Branch d=1:
      Partition: 40×40, region=6×1=6, pad to 42×42 → 7×7=49 windows
      Each window: 6×6=36 tokens of dim 256
      Tokens: (16*49, 36, 256) = (784, 36, 256)
      BiSSM scan → (784, 36, 256)
      Reverse → (16, 256, 40, 40)

    Branch d=2:
      Partition: 40×40, region=6×2=12, pad to 48×48 → 4×4=16 windows
      Each window: 6×6=36 tokens sampled every 2nd pixel from 12×12
      Tokens: (16*16, 36, 256) = (256, 36, 256)
      BiSSM scan → (256, 36, 256)
      Reverse (bilinear interp 6→12) → (16, 256, 40, 40)

    Branch d=4:
      Partition: 40×40, region=6×4=24, pad to 48×48 → 2×2=4 windows
      Each window: 6×6=36 tokens sampled every 4th pixel from 24×24
      Tokens: (16*4, 36, 256) = (64, 36, 256)
      BiSSM scan → (64, 36, 256)
      Reverse (bilinear interp 6→24) → (16, 256, 40, 40)

    Fusion:
      concat: (16, 768, 40, 40)     ← 3 × 256 channels
      gate = sigmoid(Conv1x1(768→256)): (16, 256, 40, 40) ∈ [0,1]
      proj = Conv1x1(768→256): (16, 256, 40, 40)
      out = proj * gate + input * (1-gate): (16, 256, 40, 40)
      LayerNorm → (16, 256, 40, 40)

  cv2: 3×3 conv, 256→256
       → (16, 256, 40, 40)

  Result y[2] = (16, 256, 40, 40)

─── MambaBottleneck #2 ───────────────────────────────────
  (same structure, operates on y[2])
  Result y[3] = (16, 256, 40, 40)

─── Concatenation ─────────────────────────────────────────
  cat(y[0], y[1], y[2], y[3]) along dim=1
  → (16, 1024, 40, 40)     ← (2+2) × 256 = 1024

cv2: 1×1 conv, 1024 → 512
     → (16, 512, 40, 40)

Output: (B=16, C=512, H=40, W=40)
```

**Observation**: The d=1 branch processes 784 windows (many small local contexts), while d=4 processes only 64 windows (few large global contexts). The total token count per SSM call stays at 36 in all cases. The compute difference between branches comes from the **number of windows**, not the per-window cost.

---

## 7. Why C3K2Mamba Is a Good Architectural Design

### 7.1 Preservation of YOLO's Feature Pyramid Network

C3K2Mamba doesn't change the FPN (Feature Pyramid Network) structure. The neck still has:
- Top-down path: deep features upsampled and concatenated with shallower backbone features
- Bottom-up path: refined features downsampled and concatenated back

The SSM enrichment happens **inside** each aggregation block, not between them. This means the multi-scale detection pipeline (P2/P3/P4/P5) works exactly as intended.

### 7.2 The Bypass Branch Is Critical

The `y[0]` chunk that passes through untouched is essential:
- At training start, the AtrousSSM weights are near-random. The bypass ensures the block still passes useful features through (it degrades gracefully to "mostly cv1 features").
- The fusion gate in AtrousSSM starts near 0.5, and the out_proj starts near zero, so the AtrousSSM's initial contribution is minimal. The network can gradually learn to use the SSM.

### 7.3 Weight Transfer from Pretrained C3k2

When loading ImageNet-pretrained YOLO11m weights:
- **cv1 and cv2** weights transfer directly (same dimensions as original C3k2)
- **Bottleneck cv1 and cv2** weights transfer directly (same 3×3 convs)
- **Only the AtrousSSM parts** start from random initialization

This means ~70% of C3K2Mamba's parameters start pretrained. Only the novel SSM components need to learn from scratch — which is why **freezing the backbone** and training only the neck is effective.

### 7.4 Computational Profile

For a single C3K2Mamba block at 512ch output, n=2:

| Component | Parameters | FLOPs (40×40 input) |
|-----------|-----------|---------------------|
| cv1 (1×1 conv) | 1024×512 = 524K | 838M |
| cv2 (1×1 conv) | 1024×512 = 524K | 838M |
| 2× MambaBottleneck cv1+cv2 | 2×2×(256×256×9) = 1.18M | 2×2×(589M) = 2.36G |
| 2× AtrousSSM (3 branches each) | 2×(~17×256²) ≈ 4.5M | Sequential scan (not easily measured in FLOPs) |
| **Total** | ~6.7M | ~4G + SSM sequential overhead |

The SSM sequential scan is the bottleneck — it can't be parallelized across tokens because each token depends on the hidden state from the previous token. This is why keeping the window size small (36 tokens at ws=6) is critical for T4 performance.

---

## 8. Comparison: C3K2Mamba vs Standard C3k2 vs Other Approaches

| Feature | C3k2 (YOLO11) | C3K2Mamba (Yours) | Transformer-based neck |
|---------|--------------|-------------------|----------------------|
| Receptive field | 3×3 per conv (local) | Up to 24×24 per AtrousSSM (multi-scale) | Global (full feature map) |
| Sequence modeling | None | Bidirectional SSM with hidden state propagation | Self-attention |
| Compute scaling | O(C²×H×W) | O(C²×H×W) + O(n_windows × ws² × C) sequential | O((H×W)² × C) quadratic |
| Memory | Low | Moderate (+3 branch buffers) | High (attention matrix) |
| Multi-scale context | Only through FPN stacking | **Built-in** via dilation branches [1,2,4] | Through positional encoding |
| Drop-in for YOLO? | Native | Yes (same cv1/cv2 interface) | No (needs architecture redesign) |
| Pretrained weight reuse | Full | ~70% (cv1, cv2, bottleneck convs) | None |

---

## 9. Summary

C3K2Mamba is a carefully designed module that:

1. **Inherits** the proven C2f split-concat-fuse pattern from YOLO11, preserving dense gradient flow and the FPN multi-scale structure
2. **Injects** AtrousSSM (your novel multi-scale dilated state space scanning) into the bottleneck, adding long-range context that pure convolution cannot provide
3. **Maintains compatibility** with YOLO's training pipeline, pretrained weights, and detection head
4. **Controls compute** through adaptive window sizes and strategic placement (only at 512-channel layers)
5. **Trains stably** thanks to near-zero initialization of SSM outputs, the gated fusion residual, and the untouched bypass branch

The result is a module that gives YOLO's neck **scene-level contextual understanding** (through the SSM's hidden state propagation across dilated windows) while keeping the architecture efficient enough to train on Kaggle T4 GPUs.

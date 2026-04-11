# Bidirectional Local-Window SSM Scanning: In-Depth Explanation
## Date: 2026-03-30

---

## 1. The Three Concepts Combined

"Bidirectional local-window SSM scanning" is actually three ideas stacked together:

1. **SSM scanning** — processing a sequence of tokens through a State Space Model (selective scan)
2. **Local-window** — the sequence comes from a small spatial window of the feature map, not the entire image
3. **Bidirectional** — scanning the same sequence twice (forward and backward) and combining the results

Let's unpack each one from the ground up.

---

## 2. SSM Scanning (The Foundation)

### What is a State Space Model?

A State Space Model (SSM) processes a sequence of tokens **one at a time**, maintaining a **hidden state** that accumulates information from all previously seen tokens. It's conceptually similar to an RNN, but with a specific mathematical formulation derived from continuous-time dynamical systems.

The core equations (discretized form, as in Mamba):

```
For each position i in the sequence:
    x[i] = A_bar[i] * x[i-1] + B_bar[i] * u[i]     (state update)
    y[i] = C[i] * x[i] + D * u[i]                    (output)
```

Where:
- `u[i]` = input token at position i (a D-dimensional vector, e.g., D=256)
- `x[i]` = hidden state at position i (a D×N matrix, where N=d_state=4)
- `y[i]` = output at position i (D-dimensional, same as input)
- `A_bar[i]` = state transition matrix (how much of the old state to retain)
- `B_bar[i]` = input mixing matrix (how much of the new input to absorb)
- `C[i]` = output projection (which parts of the state to read out)
- `D` = skip connection (direct input-to-output passthrough)

### Why "Selective" (Mamba's Innovation)?

In classic SSMs (like S4), A, B, C are **fixed** — the same for every input. Mamba made them **input-dependent** ("selective"):

```python
# In your _SelectiveScan1D:
xBC_dt = self.x_proj(u_act)                              # Project input
dt_raw, B_param, C_param = xBC_dt.split(...)             # Split into dt, B, C
dt = F.softplus(self.dt_proj(dt_raw))                     # Discretization step (input-dependent)
```

This means the model **decides per-token** how much to remember, how much to forget, and what to output. A token representing a human edge might trigger "strong remember" while a background token triggers "mostly forget." This selectivity is what makes Mamba competitive with Transformers.

### The Sequential Scan

The scan is inherently **sequential** — you can't compute `x[5]` without `x[4]`, because each state depends on the previous one:

```
u[0] → x[0] → y[0]
         ↓
u[1] → x[1] → y[1]
         ↓
u[2] → x[2] → y[2]
         ↓
u[3] → x[3] → y[3]
         ...
```

This is the SSM's **strength** (long-range information propagation through hidden state) and **weakness** (can't be parallelized across the sequence dimension like attention).

### What Your Code Does

In your `_SelectiveScan1D.forward()`:

```python
# The sequential scan loop
x = torch.zeros(B_win, D, self.N)     # Initialize hidden state to zero
ys = []
for i in range(L):                     # L = number of tokens in window
    x = deltaA[:, i] * x + deltaB_u[:, i]    # State update
    x = x.clamp(-1e4, 1e4)                    # Stability clamp
    y_i = (x * C_param[:, i]).sum(-1)          # Read out
    ys.append(y_i)
y = torch.stack(ys, dim=1)            # Stack all outputs
```

This loop processes tokens **one by one**, building up the hidden state. By the time it reaches token 36 (last in a 6×6 window), the hidden state `x` carries accumulated information from all 35 previous tokens.

---

## 3. Local-Window (Where the Tokens Come From)

### The Problem with Scanning Entire Feature Maps

A Mamba scan needs a **1D sequence** of tokens. A feature map is **2D** (H×W). The naive approach is to flatten the entire feature map:

```
40×40 feature map → 1600-token sequence
```

Problems:
- 1600 sequential steps is **very slow** (remember, each step depends on the previous)
- The hidden state might **saturate** or **forget early tokens** over such long sequences
- Memory usage grows with sequence length

### Local Windowing: Divide and Conquer

Instead, you **partition** the feature map into small, non-overlapping windows and scan each window independently:

```
Feature map 40×40 (with ws=6, after padding to 42×42)
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│ win  │ win  │ win  │ win  │ win  │ win  │ win  │
│ 0,0  │ 0,1  │ 0,2  │ 0,3  │ 0,4  │ 0,5  │ 0,6  │
│ 6×6  │ 6×6  │ 6×6  │ 6×6  │ 6×6  │ 6×6  │ 6×6  │
├──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ win  │ win  │ win  │ ...                        │
│ 1,0  │ 1,1  │ 1,2  │                            │
│ 6×6  │ 6×6  │ 6×6  │                            │
├──────┼──────┼──────┤         7 × 7 = 49         │
│ ...  │      │      │         windows total       │
│      │      │      │                            │
│      │      │      │                            │
└──────┴──────┴──────┴────────────────────────────┘
```

Each 6×6 window contains **36 pixels**. These 36 pixels are **flattened into a 1D sequence** and fed to the SSM:

```
Window (6×6 spatial grid):                 Flattened to 1D sequence:

 p0  p1  p2  p3  p4  p5                   p0 → p1 → p2 → p3 → p4 → p5 →
 p6  p7  p8  p9  p10 p11       ───►       p6 → p7 → p8 → p9 → p10 → p11 →
 p12 p13 p14 p15 p16 p17                  p12 → ... → p35
 p18 p19 p20 p21 p22 p23
 p24 p25 p26 p27 p28 p29                  36 tokens, each of dimension D=256
 p30 p31 p32 p33 p34 p35
```

**Benefits:**
- Each scan is only 36 steps (fast!)
- All 49 windows can be **batched together** and processed in parallel (batch dimension)
- The SSM hidden state stays fresh and focused within each window

**Drawback:**
- Each window is isolated — **no information flows between windows**
- A token in window (0,0) knows nothing about what's in window (3,5)
- This is what your AtrousSSM addresses with dilated sampling (expanding the effective window coverage)

### How Windowing Works in Code

In `_dilated_partition` (with dilation=1, the standard case):

```python
# Starting: x has shape (B, C, H, W) = (16, 256, 40, 40)

# 1. Pad to multiple of window size
#    40 → pad to 42 (nearest multiple of 6)
x = F.pad(x, (0, 2, 0, 2))  # → (16, 256, 42, 42)

# 2. Reshape into grid of windows
x = x.reshape(B, C, 7, 6, 7, 6)      # 7 windows × 6 pixels in each dim
x = x.permute(0, 2, 4, 1, 3, 5)      # → (B, 7, 7, C, 6, 6) = (B, nH, nW, C, ws, ws)

# 3. Flatten windows into batch dimension, flatten spatial into sequence
tokens = x.reshape(B * 49, C, 36)     # 49 windows per image, 36 tokens each
tokens = tokens.transpose(1, 2)        # → (B*49, 36, C) = (784, 36, 256)
```

Now you have **784 independent sequences** of 36 tokens each, ready for the SSM.

---

## 4. Bidirectional Scanning (Why Two Directions?)

### The Problem with One-Direction Scanning

When you scan left-to-right (forward), each token's output depends on **all tokens before it** (through the hidden state). But it knows **nothing about tokens after it**:

```
FORWARD scan: information flows left → right

Token:    p0    p1    p2    p3    p4    p5
           │     │     │     │     │     │
State:    x0 → x1 → x2 → x3 → x4 → x5
           │     │     │     │     │     │
Output:   y0    y1    y2    y3    y4    y5

y0 knows about: {p0}
y1 knows about: {p0, p1}
y2 knows about: {p0, p1, p2}
y5 knows about: {p0, p1, p2, p3, p4, p5}     ← most informed
```

**Problem**: `y0` (top-left pixel) has **zero context** — it was processed first, with an empty hidden state. `y5` is the most informed because the hidden state has accumulated everything. This creates an **asymmetric representation** where late tokens are much richer than early tokens.

For object detection, this is bad — a human could be at any position in the window. You don't want the top-left corner of a window to have worse representations than the bottom-right.

### The Solution: Add a Backward Scan

Run a **second, independent SSM** that scans the sequence in **reverse** (right-to-left):

```
BACKWARD scan: information flows right → left

Token:    p0    p1    p2    p3    p4    p5
           │     │     │     │     │     │
State:    x0 ← x1 ← x2 ← x3 ← x4 ← x5
           │     │     │     │     │     │
Output:   y0    y1    y2    y3    y4    y5

y0 knows about: {p0, p1, p2, p3, p4, p5}     ← most informed
y5 knows about: {p5}
```

Now **combine** forward + backward:

```
BIDIRECTIONAL: every token sees the full sequence

Forward output:   yf0   yf1   yf2   yf3   yf4   yf5
                   +     +     +     +     +     +
Backward output:  yb0   yb1   yb2   yb3   yb4   yb5
                   =     =     =     =     =     =
Combined:         y0    y1    y2    y3    y4    y5

y0 knows: {p0} from forward + {p0,...,p5} from backward = FULL CONTEXT
y3 knows: {p0,...,p3} from forward + {p3,...,p5} from backward = FULL CONTEXT
y5 knows: {p0,...,p5} from forward + {p5} from backward = FULL CONTEXT
```

**Every token now has context from the entire window**, regardless of its position.

### How Your Code Does It

In the `AtrousSSM.forward()` method, for each branch:

```python
# Two independent SSM modules with SEPARATE learned weights
y_fwd = branch['scan_fwd'](x_in)              # Forward scan: left → right
y_bwd = branch['scan_bwd'](x_in.flip(1)).flip(1)  # Backward scan: right → left

y = (y_fwd + y_bwd) * F.silu(z)               # Combine + gate
```

Let's break down the backward scan step by step:

```python
x_in.flip(1)       # Step 1: Reverse the token sequence
                    #   (B, 36, 256) → tokens in order [p35, p34, ..., p1, p0]

branch['scan_bwd'](...)  # Step 2: Run SSM forward on the reversed sequence
                          #   The SSM "thinks" it's scanning left→right,
                          #   but the input is reversed, so it's effectively
                          #   scanning the original sequence right→left

.flip(1)            # Step 3: Reverse the output back to original order
                    #   So output position 0 corresponds to input position 0
```

**Important**: `scan_fwd` and `scan_bwd` are **two completely separate `_SelectiveScan1D` modules** with **independent learned weights**. They don't share parameters. This means:
- The forward SSM can specialize in patterns that emerge left-to-right (e.g., scanning across an object from left edge to right edge)
- The backward SSM can specialize in patterns that emerge right-to-left
- The combination captures **both directional patterns**

### Visual Example: Scanning a 6×6 Window

Consider a 6×6 window that contains part of a person lying on the ground:

```
Window content (6×6 pixels):

  sky   sky   sky   sky   sky   sky
  sky   sky   sky   sky   sky   sky
  sky   sky   HEAD  TORSO sky   sky
  ground HEAD  TORSO LEGS  ground ground
  ground ground LEGS  ground ground ground
  ground ground ground ground ground ground
```

Flattened to 1D (row-major order):
```
[sky, sky, sky, sky, sky, sky, sky, sky, sky, sky, sky, sky,
 sky, sky, HEAD, TORSO, sky, sky, ground, HEAD, TORSO, LEGS, ground, ground,
 ground, ground, LEGS, ground, ground, ground, ground, ground, ground, ground, ground, ground]
```

**Forward scan** (→): When the SSM reaches TORSO at position 15, its hidden state already contains information about the HEAD at position 14 (and the sky context before it). When it reaches LEGS at position 21, it knows about TORSO and HEAD. But the HEAD at position 14 didn't know TORSO was coming.

**Backward scan** (←): When the SSM reaches HEAD at position 14 (which is processed after position 15 in the reversed sequence), it already knows about TORSO. The HEAD token now has context that body parts follow it.

**Combined**: Every body-part token knows about all other body-part tokens in the window, regardless of scan order. The HEAD knows there's a TORSO and LEGS to its right. The LEGS know there's a TORSO and HEAD to their left. This **symmetric context** helps the detector understand "this is a complete human body" rather than just "this pixel looks like a head."

---

## 5. How Dilated Windowing Changes the Picture

Standard local-window scanning uses dilation=1: the window covers a contiguous 6×6 region. Your AtrousSSM adds dilation=2 and dilation=4 branches:

### Dilation = 1 (standard)
```
Feature map 40×40:
┌──────────────────────────────────────────┐
│ ■ ■ ■ ■ ■ ■                             │   One window covers
│ ■ ■ ■ ■ ■ ■                             │   6×6 = 36 contiguous pixels
│ ■ ■ ■ ■ ■ ■     (rest of the map)       │
│ ■ ■ ■ ■ ■ ■                             │   At P4 (40×40, stride 16):
│ ■ ■ ■ ■ ■ ■                             │   6 pixels × 16 stride = 96×96
│ ■ ■ ■ ■ ■ ■                             │   input pixels covered
│                                          │
│              (49 such windows)           │
└──────────────────────────────────────────┘
```
The bidirectional SSM within this window captures context across a **96×96 pixel** area of the original image.

### Dilation = 2
```
Feature map 40×40:
┌──────────────────────────────────────────┐
│ ■ · ■ · ■ · ■ · ■ · ■ ·                │   One window covers
│ · · · · · · · · · · · ·                │   12×12 region, sampling every 2nd pixel
│ ■ · ■ · ■ · ■ · ■ · ■ ·                │   Still 6×6 = 36 tokens
│ · · · · · · · · · · · ·                │
│ ■ · ■ · ■ · ■ · ■ · ■ ·                │   At P4: 12 × 16 = 192×192
│ · · · · · · · · · · · ·                │   input pixels covered
│ ■ · ■ · ■ · ■ · ■ · ■ ·                │
│ · · · · · · · · · · · ·                │
│ ■ · ■ · ■ · ■ · ■ · ■ ·                │
│ · · · · · · · · · · · ·                │
│ ■ · ■ · ■ · ■ · ■ · ■ ·                │
│ · · · · · · · · · · · ·                │
│              (16 such windows)           │
└──────────────────────────────────────────┘

■ = sampled    · = skipped
```
The bidirectional SSM within this window captures context across a **192×192 pixel** area — **2× the spatial coverage** with the **same 36 tokens**.

### Dilation = 4
```
Coverage: 24×24 feature pixels = 384×384 input pixels
Only 4 windows on a 40×40 map (after padding to 48×48)
Same 36 tokens per window
```

### The Bidirectional Scan Operates Within Each Dilated Window

The bidirectional scanning doesn't change between dilations — it always scans the 36 tokens forward and backward. What changes is **what those 36 tokens represent spatially**:

| Dilation | Spatial coverage per window | Token spacing | What the SSM "sees" |
|----------|---------------------------|---------------|---------------------|
| d=1 | 96×96 px (local) | Adjacent pixels | Fine detail: edges, texture, body parts |
| d=2 | 192×192 px (medium) | Every 2nd pixel | Medium context: person + immediate surroundings |
| d=4 | 384×384 px (wide) | Every 4th pixel | Scene context: person + environment (rubble, water, etc.) |

In all cases, the bidirectional scan ensures every sampled token has context from every other sampled token in that window.

---

## 6. The Full Pipeline: Putting It All Together

Here's the complete flow for one AtrousSSM branch (e.g., d=2):

```
Step 1: DILATED PARTITION
═════════════════════════
Input feature map (B, 256, 40, 40)
    │
    ▼
Pad to 48×48 (multiple of region=12)
    │
    ▼
Reshape into 4×4 = 16 windows of 12×12 each
    │
    ▼
Sample every 2nd pixel → 16 windows of 6×6 each
    │
    ▼
Flatten each 6×6 to 36-token 1D sequence
    │
    ▼
Batch all windows: (B*16, 36, 256)


Step 2: BIDIRECTIONAL SSM SCAN
══════════════════════════════
(B*16, 36, 256)
    │
    ├──► Forward SSM ──► y_fwd (B*16, 36, 256)
    │     token 0 → 1 → 2 → ... → 35
    │     Each output accumulates LEFT context
    │
    └──► Reverse input → Forward SSM → Reverse output ──► y_bwd (B*16, 36, 256)
          token 35 → 34 → 33 → ... → 0
          Each output accumulates RIGHT context
    │
    ▼
y = (y_fwd + y_bwd) * SiLU(z)     ← combine + gate
    │
    Every token now has FULL bidirectional context
    within its dilated window


Step 3: DILATED REVERSE
═══════════════════════
(B*16, 36, 256)
    │
    ▼
Reshape to 16 windows of 6×6
    │
    ▼
Bilinear interpolate 6×6 → 12×12 (fill in the skipped pixels)
    │
    ▼
Reassemble 4×4 grid of 12×12 windows → 48×48
    │
    ▼
Crop padding → (B, 256, 40, 40)
```

---

## 7. Why Bidirectional Matters Specifically for Object Detection

In **language** (where Mamba originated), unidirectional scanning is natural — text flows left to right. But **images have no inherent direction**. The raster-scan order (row by row, left to right) is arbitrary. An object could be anywhere in the window.

Consider two scenarios in a 1D scan of 6 tokens:

```
Scenario A: Object at the START of the scan
─────────────────────────────────────────────
Tokens: [HUMAN, HUMAN, HUMAN, rubble, rubble, rubble]

Forward only:
  HUMAN[0]: knows only itself          → weak representation
  HUMAN[1]: knows HUMAN[0]             → slightly better
  HUMAN[2]: knows HUMAN[0,1]           → decent
  rubble tokens: know all HUMANs       → rich context (but they're background!)

Bidirectional:
  HUMAN[0]: forward{itself} + backward{everything} → FULL CONTEXT ✓
  HUMAN[2]: forward{HUMAN 0,1,2} + backward{rubble,rubble,rubble} → FULL CONTEXT ✓
  Every token is equally informed ✓


Scenario B: Object at the END of the scan
──────────────────────────────────────────
Tokens: [rubble, rubble, rubble, HUMAN, HUMAN, HUMAN]

Forward only:
  HUMAN[3]: knows rubble context       → good
  HUMAN[5]: knows everything           → best

Bidirectional:
  Every token equally informed ✓
```

With **forward-only scanning**, the quality of a token's representation depends on its **position in the scan order**, which is arbitrary. The first token in a window always gets the worst representation. For detection, this means objects near the top-left of windows would be systematically harder to detect.

**Bidirectional scanning eliminates this positional bias.** Every token in every window gets equal access to full context, regardless of where the object happens to fall in the raster-scan order.

---

## 8. Cost of Bidirectionality

The cost is straightforward: **2× the SSM computation** per window. Instead of one scan of 36 tokens, you do two (forward + backward). But:

- The two scans are **independent** and use separate weight matrices — they can theoretically be parallelized (in practice, your code runs them sequentially since Kaggle T4 VRAM is the bottleneck, not compute)
- 36 tokens × 2 scans = 72 sequential steps per window. Still far less than scanning the full 1600-token feature map (40×40)
- The learned weights are separate, so the forward and backward SSMs can **specialize** — this isn't just "the same scan twice"; each direction learns different patterns

---

## 9. Summary

| Concept | What it does | Why it matters |
|---------|-------------|----------------|
| **SSM scanning** | Processes tokens sequentially with a hidden state that accumulates information | Gives each token context from all previous tokens — long-range dependency modeling |
| **Local window** | Partitions the 2D feature map into small windows, scans each independently | Makes SSM tractable (36 tokens vs 1600+), enables batched parallel processing across windows |
| **Bidirectional** | Runs two independent SSMs (forward + backward) and combines their outputs | Eliminates positional bias — every token gets full context regardless of scan position |
| **Dilated window** (your addition) | Samples tokens from a larger area with stride, keeping token count constant | Expands receptive field without increasing per-scan cost — multi-scale context |

Together: **Bidirectional local-window SSM scanning** means "partition the feature map into small windows, scan each window's tokens forward and backward through independent SSMs with hidden state propagation, so every token gets symmetric context from all other tokens in its window." Your AtrousSSM extends this by making the windows **dilated** — sampling from larger areas at multiple scales.

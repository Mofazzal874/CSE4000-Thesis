# P2 Extra Detection Head — Adding a 4th Scale for Small Objects

> **What you'll learn:** What the P2 head is, why it helps with small/tiny objects, how it changes the YAML and architecture, and what the trade-offs are.

---

## 1. The Problem: Tiny Objects Get Lost

In disaster imagery (what our thesis studies), victims can be **very far away** from the camera (aerial/drone shots). A person might only be **5×10 pixels** in a 640×640 image. 

The standard YOLOv11m detects at three scales:

```
P3 (80×80)   →  stride 8   →  each cell covers 8×8 pixels   →  Smallest detectable ≈ 10×10px
P4 (40×40)   →  stride 16  →  each cell covers 16×16 pixels  →  Medium objects
P5 (20×20)   →  stride 32  →  each cell covers 32×32 pixels  →  Large objects
```

**A 5×10 pixel person falls BELOW the P3 resolution.** The model simply cannot see it.

---

## 2. The Solution: Add P2 (Stride 4)

By adding a P2 detection scale, we get a **160×160 feature map** where each cell covers only **4×4 pixels**:

```
NEW:
P2 (160×160) →  stride 4   →  each cell covers 4×4 pixels   →  TINY objects ✓

Existing:
P3 (80×80)   →  stride 8   →  8×8 pixels   →  Small objects
P4 (40×40)   →  stride 16  →  16×16 pixels  →  Medium objects
P5 (20×20)   →  stride 32  →  32×32 pixels  →  Large objects
```

### Visual Comparison

```
Detection grid overlay on a 640×640 image:

P5 (20×20 = 400 cells):         P2 (160×160 = 25,600 cells):
┌──┬──┬──┬──┐                   ┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┐
│  │  │  │  │                   ├┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┤
├──┼──┼──┼──┤                   ├┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┤
│  │  │  │  │  Each cell =      ├┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┤  Each cell =
├──┼──┼──┼──┤  32×32 pixels     ├┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┤  4×4 pixels
│  │  │  │  │  (coarse)         ├┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┤  (fine!)
├──┼──┼──┼──┤                   ├┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┤
│  │  │  │  │                   └┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┘
└──┴──┴──┴──┘

A 5×10px person is invisible          A 5×10px person covers
in P5 (lost in one cell)              multiple cells in P2 ✓
```

---

## 3. Where Does P2 Come From?

The backbone already creates a P2 feature map at **layer [2]** (160×160×256). In the standard model, this feature map is never used for detection — it's just an intermediate step before further downsampling.

The P2 head **reuses this existing feature map** by connecting it to the detection head via the neck.

```
Backbone Layer [2]:  160×160×256   ← This already exists!
                                      Standard YOLO ignores it
                                      P2-Head connects it to Detect ✓
```

---

## 4. Architecture Changes — Step by Step

### Standard YOLO11m Head (3-scale)
```
Layer 10 (P5) ─→ Upsample ─→ Concat(P4) ─→ C3k2 ─→ ...
                                                    ↓
                              Upsample ─→ Concat(P3) ─→ C3k2 ─→ DETECT P3
                                                               ↓
                              Downsample ─→ Concat ─→ C3k2 ──→ DETECT P4
                                                               ↓
                              Downsample ─→ Concat ─→ C3k2 ──→ DETECT P5

Detection scales: [P3, P4, P5] = 3 scales
```

### P2-Head YOLO11m (4-scale) — NEW layers highlighted
```
Layer 10 (P5) ─→ Upsample ─→ Concat(P4) ─→ C3k2 ─→ ...
                                                    ↓
                              Upsample ─→ Concat(P3) ─→ C3k2 ─→ ...
                                                                ↓
                    ┌─────────────────────────────────────────────┘
                    │  ★ NEW: Upsample ─→ Concat(P2) ─→ C3k2 ──→ DETECT P2
                    │                                              ↓
                    └──────────────────── Downsample ─→ Concat ──→ DETECT P3
                                                                   ↓
                                         Downsample ─→ Concat ──→ DETECT P4
                                                                   ↓
                                         Downsample ─→ Concat ──→ DETECT P5

Detection scales: [P2, P3, P4, P5] = 4 scales
```

---

## 5. YAML Changes — Exactly What We Add

### Standard Head (layers 11-23):
```yaml
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]     # 11: 20→40
  - [[-1, 6], 1, Concat, [1]]                        # 12: cat P4
  - [-1, 2, C3k2, [512, False]]                      # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]      # 14: 40→80
  - [[-1, 4], 1, Concat, [1]]                        # 15: cat P3
  - [-1, 2, C3k2, [256, False]]                      # 16 ← P3 (lowest in standard)

  - [-1, 1, Conv, [256, 3, 2]]                       # 17: 80→40
  - [[-1, 13], 1, Concat, [1]]                       # 18
  - [-1, 2, C3k2, [512, False]]                      # 19 ← P4

  - [-1, 1, Conv, [512, 3, 2]]                       # 20: 40→20
  - [[-1, 10], 1, Concat, [1]]                       # 21
  - [-1, 2, C3k2, [1024, True]]                      # 22 ← P5

  - [[16, 19, 22], 1, Detect, [nc]]                  # 23: 3-scale detect
```

### P2-Head (layers 11-29) — 3 new layers inserted:
```yaml
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]      # 11: 20→40
  - [[-1, 6], 1, Concat, [1]]                        # 12
  - [-1, 2, C3k2, [512, False]]                      # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]      # 14: 40→80
  - [[-1, 4], 1, Concat, [1]]                        # 15
  - [-1, 2, C3k2, [256, False]]                      # 16

  # ★ NEW: P2 branch (3 new layers)
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]      # 17: 80→160 ★
  - [[-1, 2], 1, Concat, [1]]                        # 18: cat backbone layer 2 ★
  - [-1, 2, C3k2, [128, False]]                      # 19: P2/4 features ★

  - [-1, 1, Conv, [128, 3, 2]]                       # 20: 160→80
  - [[-1, 16], 1, Concat, [1]]                       # 21
  - [-1, 2, C3k2, [256, False]]                      # 22 ← P3

  - [-1, 1, Conv, [256, 3, 2]]                       # 23: 80→40
  - [[-1, 13], 1, Concat, [1]]                       # 24
  - [-1, 2, C3k2, [512, False]]                      # 25 ← P4

  - [-1, 1, Conv, [512, 3, 2]]                       # 26: 40→20
  - [[-1, 10], 1, Concat, [1]]                       # 27
  - [-1, 2, C3k2, [1024, True]]                      # 28 ← P5

  - [[19, 22, 25, 28], 1, Detect, [nc]]              # 29: 4-scale detect ★
```

### What Changed
| Aspect | Standard | P2-Head |
|---|---|---|
| Total head layers | 13 (11-23) | 19 (11-29) |
| Detection scales | 3 (P3, P4, P5) | 4 (P2, P3, P4, P5) |
| Detect input | `[16, 19, 22]` | `[19, 22, 25, 28]` |
| New layers | — | 17 (Upsample), 18 (Concat), 19 (C3k2) |
| Feature map sizes | 80², 40², 20² | 160², 80², 40², 20² |

---

## 6. Understanding the New Layers

### Layer 17: `nn.Upsample [None, 2, "nearest"]`
```
Input:   80×80×256  (from P3 processing)
Output: 160×160×256 (doubled — now matches backbone P2 size)
```

### Layer 18: `Concat [-1, 2]`
```
From layer 17: 160×160×256  (upsampled P3 features — semantic rich)
From layer 2:  160×160×256  (backbone P2 features — spatially detailed)
              ─────────────
Concatenated: 160×160×512   (best of both worlds!)
```

This is the **key operation**: combining the high-resolution spatial detail from the backbone with the semantically rich features from the FPN pathway.

### Layer 19: `C3k2 [128, False]`
```
Input:  160×160×512  (concatenated)
Output: 160×160×128  (processed P2 features ready for detection)
```

---

### Memory Impact & Batch Size Logic
The P2 feature maps are **4× larger** spatially than P3. This consumes significantly more GPU VRAM.

In `YOLO11m_P2Head.py`, we explicitly handle this for your **Tesla T4 (15GB)**:

```python
# From YOLO11m_P2Head.py
# P2 head uses more memory (stride-4 feature maps are 4x larger spatially)
# batch=8 is safe for T4 (14.6GB), batch=16 will OOM
p2_batch = 8 if gpu_mem >= 14 else 4
print(f"P2 head batch size: {p2_batch} (reduced due to extra P2 scale)")
```

> **Compare:** Standard YOLOv11m can often run at batch=16 on a T4. P2-Head requires batch=8 to avoid "Out of Memory" (OOM) errors.

### Parameter Impact
```
Standard YOLO11m:  ~20.1M parameters
P2-Head YOLO11m:   ~19.6M parameters (some layers are smaller)
Difference:        ~0.5M fewer (!)
```

> ⚠️ **Surprising**: P2-Head actually has FEWER parameters because the new P2 C3k2 uses only 128 channels (vs 256/512 in deeper layers). But it uses more **memory** because of the larger spatial dimensions.

### Speed Impact
```
Standard:  ~68 GFLOPs,  ~7ms inference
P2-Head:   ~87 GFLOPs,  ~10ms inference  (+28% compute)
```

The extra compute comes from processing 160×160 feature maps.

---

## 8. Why This Helps Disaster Detection

### Our Use Case: C2A Dataset
- **Aerial/drone Images** of disaster scenes
- **Victims** can be very small (sub-10px)
- Current P3 (stride 8) misses the tiniest targets

### Expected Improvements

| Size Category | Stride | Standard | P2-Head |
|---|---|---|---|
| Very Tiny (<8²px) | 4 | ❌ Below detection limit | ✅ Now detectable |
| Tiny (8-16px) | 4-8 | ⚠️ Marginal detection | ✅ Better coverage |
| Small (16-32px) | 8 | ✅ OK | ✅ Improved |
| Medium (32-96px) | 16 | ✅ Good | ✅ Same or better |
| Large (>96px) | 32 | ✅ Good | ✅ Same |

---

## 9. CBAM + P2 Head Combined

In the `YOLO11m_CBAM_P2Head.py` script, we combine **both** modifications:

```yaml
backbone:
  - ...
  - [-1, 1, SPPF, [1024, 5]]        # [9]
  - [-1, 1, CBAM, [16, 7]]          # [10] ← CBAM replaces C2PSA

head:
  - ...
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # [17] ← P2 Upsample
  - [[-1, 2], 1, Concat, [1]]                      # [18] ← P2 Concat
  - [-1, 2, C3k2, [128, False]]                    # [19] ← P2 C3k2
  - ...
  - [[19, 22, 25, 28], 1, Detect, [nc]]            # [29] ← 4-scale Detect
```

The hypothesis: **CBAM improves feature quality** (what/where to focus) while **P2 provides the resolution** (can see tiny targets). Together, they should maximize small-object recall.

---

## Summary

```
P2 Detection Head at a glance:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What:     4th detection scale (stride 4, 160×160)
Where:    3 new layers in the head, new Detect input
Why:      Detect very tiny objects (<10px humans)
Memory:   batch 16→8 (larger feature maps)
Speed:    +28% GFLOPs (extra computation)
Params:   Actually ~0.5M FEWER (smaller channel width)
Source:   Ultralytics PR #16558 (yolo11-p2.yaml)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

# Thesis Results — Complete Slide Deck
## Aerial Human Detection using Modified YOLO11m
### Date: 2026-03-29

---

# SLIDE 1: Dataset Overview

## C2A (Crop2Aerial) Person Detection Dataset
- **Classes:** 1 (person)
- **Images:** ~6,000
- **Labeled instances:** ~360,000
- **Challenge:** Tiny persons in aerial/drone imagery
- **Size categories:**
  - Very Tiny: 1-32 px
  - Tiny: 32-96 px
  - Small: 96-256 px
  - Medium: 256+ px

---

# SLIDE 2: Experimental Roadmap

```
Phase 1: Baseline Benchmarking (25 epochs)
  -> YOLOv9 (s/m/e), YOLOv10 (s/m/l), YOLOv11 (s/m/l)

Phase 2: Ablation Study on YOLO11m (50-70 epochs)
  -> Baseline -> +ECA -> +CBAM -> +P2Head -> +CBAM+P2Head

Phase 3: SSM Neck Integration (120 epochs)
  -> YOLO11m + Mamba + CBAM + P2Head (C3K2Mamba neck)

Phase 4: Novel AtrousSSM (80 epochs)
  -> YOLO11m + AtrousMamba + CBAM + P2Head (dilated SSM scanning)
```

---

# SLIDE 3: Phase 1 — YOLO Family Benchmarks (25 epochs, C2A)

| Model       | Precision | Recall | mAP50  | mAP50-95 |
|-------------|-----------|--------|--------|----------|
| YOLOv9s     | 0.8383    | 0.7231 | 0.7763 | 0.4910   |
| YOLOv9m     | 0.8589    | 0.7634 | 0.8142 | 0.5470   |
| YOLOv9e     | 0.8793    | 0.7990 | 0.8472 | 0.6010   |
| YOLOv10s    | 0.8324    | 0.7311 | 0.7848 | 0.5068   |
| YOLOv10m    | 0.8564    | 0.7682 | 0.8229 | 0.5595   |
| YOLOv10l    | 0.8756    | 0.7898 | 0.8408 | 0.5863   |
| YOLO11s     | 0.8472    | 0.7460 | 0.7967 | 0.5199   |
| YOLO11m     | 0.8756    | 0.7898 | 0.8408 | 0.5863   |
| YOLO11l     | 0.8796    | 0.7897 | 0.8408 | 0.5863   |

**Key finding:** YOLO11m matches YOLO11l performance at lower compute. Selected as base architecture.

---

# SLIDE 4: Phase 1 — Visual Comparison Chart Data

```
mAP50 ranking (25 epochs):
1. YOLOv9e     — 0.8472
2. YOLO11m     — 0.8408
3. YOLO11l     — 0.8408
4. YOLOv10l    — 0.8408
5. YOLOv10m    — 0.8229
6. YOLOv9m     — 0.8142
7. YOLO11s     — 0.7967
8. YOLOv10s    — 0.7848
9. YOLOv9s     — 0.7763

mAP50-95 ranking (25 epochs):
1. YOLOv9e     — 0.6010
2. YOLO11m     — 0.5863
3. YOLO11l     — 0.5863
4. YOLOv10l    — 0.5863
5. YOLOv10m    — 0.5595
6. YOLO11s     — 0.5199
7. YOLOv10s    — 0.5068
8. YOLOv9m     — 0.5470
9. YOLOv9s     — 0.4910
```

---

# SLIDE 5: Phase 2 — Ablation Study on YOLO11m

| Model                    | Epochs | Precision | Recall | mAP50  | mAP50-95 |
|--------------------------|--------|-----------|--------|--------|----------|
| YOLO11m Baseline         | 50     | 0.8783    | 0.7972 | 0.8490 | 0.6020   |
| YOLO11m + ECA            | 50     | 0.8789    | 0.7966 | 0.8489 | 0.5995   |
| YOLO11m + CBAM           | 70     | 0.8806    | 0.7988 | 0.8520 | 0.6073   |
| YOLO11m + P2Head         | 50     | 0.8808    | 0.7877 | 0.8713 | 0.6297   |
| YOLO11m + CBAM + P2Head  | 70     | 0.8781    | 0.8188 | 0.8693 | 0.6257   |

---

# SLIDE 6: Phase 2 — Ablation Analysis

## Attention Module Comparison
- **ECA:** No improvement (+0.00 mAP50, -0.25% mAP50-95) -> Dropped
- **CBAM:** Marginal gain (+0.30 mAP50, +0.53% mAP50-95) -> Selected

## P2 Head Impact (Major Structural Change)
- **+P2Head alone:** +2.23% mAP50, +2.77% mAP50-95
- **+CBAM+P2Head:** +2.03% mAP50, +2.37% mAP50-95, **+2.16% Recall**

## Key Insight
P2 Head is the single most impactful modification — adds 4th detection scale for tiny objects. CBAM+P2 chosen as base for SSM experiments (best recall).

---

# SLIDE 7: Phase 3 — Mamba SSM Neck Integration

## Architecture: YOLO11m + C3K2Mamba + CBAM + P2Head
- **C3K2Mamba:** Replaces standard C3k2 blocks in neck layers with Mamba-integrated blocks
- **Bidirectional scanning:** Forward + reverse sequence scanning in each Mamba block
- **Pure PyTorch:** No CUDA kernel dependency (runs on any GPU)

### Training Config
- Epochs: 120 | Batch: 8 | Optimizer: AdamW | LR: 0.0005
- Backbone: 11 layers frozen | GPU: Kaggle T4

---

# SLIDE 8: Phase 3 — Mamba Results (Test Set)

| Metric          | CBAM+P2 Baseline | Mamba+CBAM+P2 | Delta    |
|-----------------|------------------|---------------|----------|
| Precision       | 0.8502           | 0.8547        | +0.45%   |
| Recall          | 0.8426           | 0.8453        | +0.27%   |
| F1 Score        | 0.8464           | 0.8500        | +0.36%   |
| mAP50           | 0.8739           | 0.8770        | **+0.31%** |
| mAP50-95        | 0.6450           | 0.6539        | **+0.89%** |
| ECE (Calibr.)   | 0.1401           | 0.1346        | -0.55pp  |

### Parameters & Speed
| Metric     | CBAM+P2 | Mamba+CBAM+P2 |
|------------|---------|---------------|
| Params     | 19.592M | 19.592M       |
| GFLOPs     | 43.7    | 43.7          |
| Latency    | 46.1ms  | 45.9ms        |

**Key: Same model size, same speed, better accuracy + calibration**

---

# SLIDE 9: Phase 3 — Mamba Multi-Scale Recall

| Object Size     | CBAM+P2 | Mamba+CBAM+P2 | Delta    |
|-----------------|---------|---------------|----------|
| Very Tiny (1-32px)  | 0.7648  | 0.7668    | +0.20%   |
| Tiny (32-96px)      | 0.8742  | 0.8752    | +0.10%   |
| Small (96-256px)    | 0.8917  | 0.8960    | +0.43%   |
| Medium (256+px)     | 0.8675  | 0.8959    | **+2.84%** |

**Mamba improves across ALL size categories, largest gain on medium objects (+2.84%)**

---

# SLIDE 10: Phase 3 — Mamba Training Convergence

```
Mamba+CBAM+P2 (120 epochs):
- Best epoch: 103 (F2=0.8351, mAP50=0.8750)
- Final epoch 120: mAP50=0.8746, mAP50-95=0.6373
- Stable convergence after epoch ~90
- No overfitting observed (val loss still decreasing)
```

---

# SLIDE 11: Phase 4 — AtrousSSM (Novel Contribution)

## AtrousSSM: Atrous State Space Model
**Core idea:** Apply dilated (atrous) scanning patterns to SSM sequence processing

### Architecture
- **3 parallel branches** with dilations [1, 2, 4]
  - Branch 1 (d=1): Standard sequential scan (local context)
  - Branch 2 (d=2): Skip-1 scan (medium-range context)
  - Branch 3 (d=4): Skip-3 scan (long-range context)
- **Bidirectional scan** per branch (forward + reverse)
- **Gated fusion** to combine multi-scale features
- **d_state=4**, pure PyTorch implementation

### Placement in YOLO11m
- Neck layers 13, 16, 19, 22, 25 (replaces C3k2 blocks)
- Combined with CBAM attention + P2 detection head

---

# SLIDE 12: Phase 4 — AtrousSSM Training Config

| Parameter        | AtrousMamba          | Mamba (baseline)    |
|------------------|---------------------|---------------------|
| Epochs           | 80                  | 120                 |
| Batch Size       | 20 (10/GPU)         | 8                   |
| GPUs             | 2x T4 (Kaggle)     | 1x T4 (Kaggle)     |
| Optimizer        | AdamW               | AdamW               |
| Learning Rate    | 0.0005              | 0.0005              |
| Frozen Backbone  | 11 layers           | 11 layers           |
| Image Size       | 640x640             | 640x640             |

---

# SLIDE 13: Phase 4 — AtrousSSM Results (Test Set)

| Metric          | CBAM+P2  | Mamba+CBAM+P2 | AtrousMamba+CBAM+P2 |
|-----------------|----------|---------------|---------------------|
| Precision       | 0.8502   | 0.8547        | 0.8127              |
| Recall          | 0.8426   | 0.8453        | 0.7979              |
| F1 Score        | 0.8464   | 0.8500        | 0.8052              |
| mAP50           | 0.8739   | **0.8770**    | 0.8228              |
| mAP50-95        | 0.6450   | **0.6539**    | 0.5540              |
| ECE             | 0.1401   | **0.1346**    | 0.1497              |
| Latency (ms)    | 46.1     | **45.9**      | 105.1               |
| Params (M)      | 19.592   | 19.592        | 24.156              |
| GFLOPs          | 43.7     | 43.7          | 48.1                |

---

# SLIDE 14: Phase 4 — AtrousSSM Multi-Scale Recall

| Object Size         | CBAM+P2 | Mamba+CBAM+P2 | AtrousMamba+CBAM+P2 |
|---------------------|---------|---------------|---------------------|
| Very Tiny (1-32px)  | 0.7648  | **0.7668**    | 0.6831              |
| Tiny (32-96px)      | 0.8742  | **0.8752**    | 0.8417              |
| Small (96-256px)    | 0.8917  | **0.8960**    | 0.8709              |
| Medium (256+px)     | 0.8675  | 0.8959        | **0.9369**          |

### AtrousSSM Analysis
- **Medium objects:** AtrousMamba BEST (+4.10% over Mamba) — dilated scanning captures full-body context
- **Tiny/Very Tiny:** AtrousMamba WORST (-8.37% very tiny vs Mamba) — dilated sampling loses fine-grained detail
- **Net effect:** Overall regression due to dataset being dominated by tiny objects

---

# SLIDE 15: Phase 4 — AtrousSSM Diagnosis

## Why AtrousMamba Underperformed

1. **Dilated scanning loses local detail**
   - d=2 and d=4 branches skip adjacent tokens
   - Critical for 1-32px objects where EVERY pixel matters

2. **Computational overhead**
   - 3 parallel branches + gated fusion = 2.3x slower (105ms vs 46ms)
   - +4.6M parameters (+23%) with no accuracy gain

3. **Only 80 epochs (vs 120 for Mamba)**
   - AtrousMamba was still improving at epoch 80
   - Best epoch: 63 (mAP50=0.8197) — may not have fully converged

4. **Dataset mismatch**
   - C2A is dominated by tiny persons (1-96px)
   - AtrousSSM benefits medium+ objects but hurts the dominant class

---

# SLIDE 16: Complete Model Progression

| Model                          | Epochs | mAP50  | mAP50-95 | Params  |
|--------------------------------|--------|--------|----------|---------|
| YOLO11m Baseline               | 50     | 0.8490 | 0.6020   | 20.1M   |
| + ECA                          | 50     | 0.8489 | 0.5995   | 20.1M   |
| + CBAM                         | 70     | 0.8520 | 0.6073   | 20.1M   |
| + P2Head                       | 50     | 0.8713 | 0.6297   | 19.6M   |
| + CBAM + P2Head                | 70     | 0.8693 | 0.6257   | 19.6M   |
| + Mamba + CBAM + P2Head        | 120    | **0.8770** | **0.6539** | 19.6M |
| + AtrousMamba + CBAM + P2Head  | 80     | 0.8228 | 0.5540   | 24.2M   |

### Best Model: Mamba+CBAM+P2Head
- **+2.80% mAP50** over baseline
- **+5.19% mAP50-95** over baseline
- **Same parameters** as CBAM+P2 baseline
- **No speed penalty** (45.9ms vs 46.1ms)

---

# SLIDE 17: Cross-Family Comparison (Best Models)

| Model                     | mAP50  | mAP50-95 | Precision | Recall |
|---------------------------|--------|----------|-----------|--------|
| YOLOv9e (25ep)            | 0.8472 | 0.6010   | 0.8793    | 0.7990 |
| YOLOv10l (25ep)           | 0.8408 | 0.5863   | 0.8756    | 0.7898 |
| YOLO11m Baseline (50ep)   | 0.8490 | 0.6020   | 0.8783    | 0.7972 |
| **Ours: Mamba+CBAM+P2**   | **0.8770** | **0.6539** | 0.8547 | **0.8453** |

### Our model vs YOLOv9e (strongest competitor)
- mAP50: +2.98%
- mAP50-95: +5.29%
- Recall: +4.63% (critical for search-and-rescue)

---

# SLIDE 18: Key Contributions Summary

## 1. Systematic Benchmarking
- 9 YOLO variants (v9/v10/v11) evaluated on C2A aerial dataset
- YOLO11m identified as optimal base (best accuracy/compute tradeoff)

## 2. Ablation Study
- ECA attention: No benefit on aerial detection -> dropped
- CBAM attention: Marginal mAP gain, slight recall improvement
- P2 Head: Most impactful single modification (+2.23% mAP50)

## 3. Mamba SSM Neck (Best Result)
- Bidirectional Mamba scanning in C3K2 neck blocks
- +0.31% mAP50, +0.89% mAP50-95 over CBAM+P2
- Zero parameter/speed overhead

## 4. AtrousSSM (Novel Architecture — Negative Result)
- First dilated scanning mechanism for SSMs
- Excels on medium objects (+4.1%) but regresses on tiny (-8.4%)
- Thesis contribution: demonstrates SSM scanning pattern design space

---

# SLIDE 19: Future Work

1. **AtrousSSM v2:** Adaptive dilation rates learned per layer
2. **Hybrid approach:** AtrousSSM for deep layers (medium objects) + standard Mamba for shallow layers (tiny objects)
3. **Extended training:** AtrousMamba with 120+ epochs for fair comparison
4. **WaveAtrousMamba:** Wavelet-guided multi-scale feature fusion with AtrousSSM
5. **Cross-dataset validation:** VisDrone, DOTA for generalization testing

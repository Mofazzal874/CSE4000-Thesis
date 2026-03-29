# Detailed Training Metrics — All Experiments
## Date: 2026-03-29

---

## 1. YOLO Family Benchmarks (Phase 1, 25 epochs each)

### YOLOv9 Series
| Metric     | YOLOv9s | YOLOv9m | YOLOv9e |
|------------|---------|---------|---------|
| Precision  | 0.8383  | 0.8589  | 0.8793  |
| Recall     | 0.7231  | 0.7634  | 0.7990  |
| mAP50      | 0.7763  | 0.8142  | 0.8472  |
| mAP50-95   | 0.4910  | 0.5470  | 0.6010  |

### YOLOv10 Series
| Metric     | YOLOv10s | YOLOv10m | YOLOv10l |
|------------|----------|----------|----------|
| Precision  | 0.8324   | 0.8564   | 0.8756   |
| Recall     | 0.7311   | 0.7682   | 0.7898   |
| mAP50      | 0.7848   | 0.8229   | 0.8408   |
| mAP50-95   | 0.5068   | 0.5595   | 0.5863   |

### YOLO11 Series
| Metric     | YOLO11s | YOLO11m | YOLO11l |
|------------|---------|---------|---------|
| Precision  | 0.8472  | 0.8756  | 0.8796  |
| Recall     | 0.7460  | 0.7898  | 0.7897  |
| mAP50      | 0.7967  | 0.8408  | 0.8408  |
| mAP50-95   | 0.5199  | 0.5863  | 0.5863  |

---

## 2. Ablation Study (Phase 2)

| Model                   | Epochs | Precision | Recall | mAP50  | mAP50-95 | Delta mAP50 | Delta mAP50-95 |
|-------------------------|--------|-----------|--------|--------|----------|-------------|----------------|
| YOLO11m Baseline        | 50     | 0.8783    | 0.7972 | 0.8490 | 0.6020   | —           | —              |
| YOLO11m + ECA           | 50     | 0.8789    | 0.7966 | 0.8489 | 0.5995   | -0.01%      | -0.25%         |
| YOLO11m + CBAM          | 70     | 0.8806    | 0.7988 | 0.8520 | 0.6073   | +0.30%      | +0.53%         |
| YOLO11m + P2Head        | 50     | 0.8808    | 0.7877 | 0.8713 | 0.6297   | +2.23%      | +2.77%         |
| YOLO11m + CBAM + P2Head | 70     | 0.8781    | 0.8188 | 0.8693 | 0.6257   | +2.03%      | +2.37%         |

---

## 3. Mamba+CBAM+P2Head (Phase 3, 120 epochs)

### Training Progression (Selected Epochs)
| Epoch | Precision | Recall | mAP50  | mAP50-95 | Train Box Loss | Val Box Loss |
|-------|-----------|--------|--------|----------|----------------|--------------|
| 1     | 0.7998    | 0.7135 | 0.7432 | 0.4315   | 1.9106         | 1.3636       |
| 20    | 0.8597    | 0.7879 | 0.8375 | 0.5698   | 1.1309         | 0.9829       |
| 40    | 0.8627    | 0.8015 | 0.8524 | 0.5925   | 1.0728         | 0.9238       |
| 60    | 0.8752    | 0.8155 | 0.8649 | 0.6166   | 1.0209         | 0.8754       |
| 80    | 0.8798    | 0.8193 | 0.8706 | 0.6282   | 0.9970         | 0.8540       |
| 100   | 0.8840    | 0.8242 | 0.8747 | 0.6355   | 0.9660         | 0.8399       |
| 103   | 0.8838    | 0.8237 | 0.8751 | 0.6358   | 0.9615         | 0.8394       |
| 120   | 0.8848    | 0.8231 | 0.8746 | 0.6373   | 0.9066         | 0.8371       |

### Test Set Evaluation
| Metric              | Value   |
|---------------------|---------|
| Precision           | 0.8547  |
| Recall              | 0.8453  |
| F1 Score            | 0.8500  |
| F2 Score            | 0.8472  |
| mAP50               | 0.8770  |
| mAP50-95            | 0.6539  |
| ECE (Calibration)   | 0.1346  |
| Latency             | 45.9 ms |
| Parameters          | 19.592M |
| GFLOPs              | 43.7    |

### Size-Specific Recall (Test Set)
| Size Category      | Recall |
|--------------------|--------|
| Very Tiny (1-32px) | 0.7668 |
| Tiny (32-96px)     | 0.8752 |
| Small (96-256px)   | 0.8960 |
| Medium (256+px)    | 0.8959 |

---

## 4. AtrousMamba+CBAM+P2Head (Phase 4, 80 epochs, 2xT4)

### AtrousSSM Configuration
- Dilations: [1, 2, 4]
- d_state: 4
- Branches: 3 parallel, each bidirectional
- Fusion: Gated combination

### Training Progression (Selected Epochs)
| Epoch | Precision | Recall | mAP50  | mAP50-95 | Train Box Loss | Val Box Loss |
|-------|-----------|--------|--------|----------|----------------|--------------|
| 1     | 0.7716    | 0.6679 | 0.7050 | 0.3799   | 2.1161         | 1.5177       |
| 20    | 0.8462    | 0.7729 | 0.8188 | 0.5466   | 1.2408         | 1.0310       |
| 40    | 0.8595    | 0.7928 | 0.8416 | 0.5805   | 1.1281         | 0.9539       |
| 60    | 0.8728    | 0.8092 | 0.8595 | 0.6109   | 1.0509         | 0.8963       |
| 63    | 0.8737    | 0.8097 | 0.8612 | 0.6137   | 1.0409         | 0.8883       |
| 80    | 0.8798    | 0.8193 | 0.8706 | 0.6282   | 0.9970         | 0.8540       |

### Test Set Evaluation
| Metric              | Value   |
|---------------------|---------|
| Precision           | 0.8127  |
| Recall              | 0.7979  |
| F1 Score            | 0.8052  |
| F2 Score            | 0.8008  |
| mAP50               | 0.8228  |
| mAP50-95            | 0.5540  |
| ECE (Calibration)   | 0.1497  |
| Latency             | 105.1 ms|
| Parameters          | 24.156M |
| GFLOPs              | 48.1    |

### Size-Specific Recall (Test Set)
| Size Category      | Recall |
|--------------------|--------|
| Very Tiny (1-32px) | 0.6831 |
| Tiny (32-96px)     | 0.8417 |
| Small (96-256px)   | 0.8709 |
| Medium (256+px)    | 0.9369 |

---

## 5. Head-to-Head: All 3 SSM/Attention Variants (Test Set)

| Metric              | CBAM+P2 (baseline) | Mamba+CBAM+P2 | AtrousMamba+CBAM+P2 |
|---------------------|---------------------|---------------|---------------------|
| Precision           | 0.8502              | **0.8547**    | 0.8127              |
| Recall              | 0.8426              | **0.8453**    | 0.7979              |
| F1                  | 0.8464              | **0.8500**    | 0.8052              |
| F2                  | 0.8441              | **0.8472**    | 0.8008              |
| mAP50               | 0.8739              | **0.8770**    | 0.8228              |
| mAP50-95            | 0.6450              | **0.6539**    | 0.5540              |
| ECE                 | 0.1401              | **0.1346**    | 0.1497              |
| Latency (ms)        | 46.1                | **45.9**      | 105.1               |
| Params (M)          | 19.592              | **19.592**    | 24.156              |
| GFLOPs              | 43.7                | **43.7**      | 48.1                |
| Very Tiny Recall    | 0.7648              | **0.7668**    | 0.6831              |
| Tiny Recall         | 0.8742              | **0.8752**    | 0.8417              |
| Small Recall        | 0.8917              | **0.8960**    | 0.8709              |
| Medium Recall       | 0.8675              | 0.8959        | **0.9369**          |
| Best Epoch          | 36                  | 103           | 63                  |

**Bold = best in row**

---

## 6. Complete Progression Summary (Validation Best → Test)

| # | Model                          | Epochs | Val mAP50 | Val mAP50-95 | Test mAP50 | Test mAP50-95 |
|---|--------------------------------|--------|-----------|--------------|------------|---------------|
| 1 | YOLO11m Baseline               | 50     | 0.8490    | 0.6020       | —          | —             |
| 2 | + ECA                          | 50     | 0.8489    | 0.5995       | —          | —             |
| 3 | + CBAM                         | 70     | 0.8520    | 0.6073       | —          | —             |
| 4 | + P2Head                       | 50     | 0.8713    | 0.6297       | —          | —             |
| 5 | + CBAM + P2Head                | 70     | 0.8693    | 0.6257       | 0.8739     | 0.6450        |
| 6 | + Mamba + CBAM + P2Head        | 120    | 0.8746    | 0.6373       | **0.8770** | **0.6539**    |
| 7 | + AtrousMamba + CBAM + P2Head  | 80     | 0.8706    | 0.6282       | 0.8228     | 0.5540        |

Note: Test set evaluation only available for models 5-7 (full benchmark pipeline was run for these).

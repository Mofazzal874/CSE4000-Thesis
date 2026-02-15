# Key References — Where to Learn More

> **What you'll find here:** Links to the original papers, official documentation, and community resources used to build this documentation.

---

## 1. Scientific Papers (The "Source Code" of Ideas)

**YOLOv11** (released Sep 2024, no paper yet!)
- **Source:** [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- **Key Features:** C3k2, C2PSA, SPPF

**CBAM: Convolutional Block Attention Module** (ECCV 2018)
- **Paper:** [https://arxiv.org/abs/1807.06521](https://arxiv.org/abs/1807.06521)
- **Authors:** Sanghyun Woo et al.
- **Why read:** Explains "channel attention" vs "spatial attention" in detail.

**FPN: Feature Pyramid Networks** (CVPR 2017)
- **Paper:** [https://arxiv.org/abs/1612.03144](https://arxiv.org/abs/1612.03144)
- **Concept:** How to build a pyramid of features (P3, P4, P5).

**PANet: Path Aggregation Network** (CVPR 2018)
- **Paper:** [https://arxiv.org/abs/1803.01534](https://arxiv.org/abs/1803.01534)
- **Concept:** Adding the "bottom-up" path to FPN (used in YOLO neck).

**CSPNet: Cross Stage Partial Network** (CVPR 2020)
- **Paper:** [https://arxiv.org/abs/1911.11929](https://arxiv.org/abs/1911.11929)
- **Concept:** The "split and merge" idea used in `C3k2`.

---

## 2. Official Documentation

**Ultralytics YOLO Docs**
- [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
- **Best for:** How to train, validate, and predict.

**PyTorch Documentation**
- [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
- **Look up:** `nn.Conv2d`, `nn.Upsample`, `nn.AdaptiveAvgPool2d`.

---

## 3. Helpful Explainer Videos

**What is YOLO? (Computer Vision)**
- *Video by CodeEmporium*: Great for high-level intuition.

**Attention Mechanisms in Computer Vision**
- *Video by Andrew Ng (DeepLearning.AI)*: The best explanation of *why* attention works.

**Understanding FPN (Feature Pyramid Networks)**
- *Video by Aladdin Persson*: Excellent whiteboard explanation of P3/P4/P5 scales.

---

## 4. Glossary of Terms Used in This Project

- **Backbone**: Feature extractor part of the network.
- **Neck**: Feature aggregator part (FPN+PAN).
- **Head**: Detection part (predicts boxes).
- **Stride**: Reduction factor (stride 2 = half size).
- **Channel**: Depth of a feature map (e.g., 256 filters).
- **Batch**: Number of images processed at once.
- **Epoch**: One complete pass through all training images.
- **Ablation Study**: Systematically removing/adding parts (like CBAM or P2) to see their effect.

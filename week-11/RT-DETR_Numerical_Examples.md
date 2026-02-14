# RT-DETR: Numerical Walkthrough with Examples

This document provides concrete numerical examples to help you understand how data flows through RT-DETR.

## 1. Feature Map Dimensions: Step-by-Step

### Input Image
```
Input: (640, 640, 3)
- 640 pixels wide
- 640 pixels tall  
- 3 color channels (RGB)
```

### After Backbone (ResNet-50)

The backbone progressively reduces spatial dimensions while increasing channels:

```
Stage 3 (S3): (80, 80, 256)
- 80 × 80 = 6,400 spatial positions
- Each position has 256-dimensional feature vector
- Total: 6,400 feature vectors

Stage 4 (S4): (40, 40, 512)
- 40 × 40 = 1,600 spatial positions
- Each position has 512-dimensional feature vector
- Total: 1,600 feature vectors

Stage 5 (S5): (20, 20, 1024)
- 20 × 20 = 400 spatial positions
- Each position has 1024-dimensional feature vector
- Total: 400 feature vectors
```

**Why multiple scales?**
- S3: High resolution, good for small objects
- S4: Medium resolution, good for medium objects
- S5: Low resolution, good for large objects and semantic understanding

---

## 2. Understanding Feature Vectors

### What is a Feature Vector?

Think of each position in a feature map as having a "description":

```
Position (10, 10) in S5:
Feature Vector = [0.23, -0.45, 0.89, 0.12, ..., -0.33]
                  ↑                              ↑
                  dim 0                     dim 1023
                  
This 1024-dimensional vector encodes:
- What objects might be there (cat? dog? car?)
- Texture information
- Color patterns
- Spatial context
```

### Simple 2D Example (for intuition)

Let's simplify to 2D to visualize:

```
Imagine S5 as 4×4 grid with 3-dimensional features:

Position (0,0): [0.8, 0.2, 0.1]  ← "probably contains sky"
Position (0,1): [0.7, 0.3, 0.2]  ← "probably contains sky"
Position (2,2): [0.1, 0.9, 0.7]  ← "probably contains cat"
Position (3,3): [0.2, 0.1, 0.8]  ← "probably contains grass"
```

---

## 3. Traditional vs RT-DETR Encoder: Number Comparison

### Traditional Multi-Scale Encoder (SLOW)

```
Step 1: Flatten all scales
S3: 6,400 vectors (256-dim) → Reshape to (6400, 256)
S4: 1,600 vectors (512-dim) → Resize to (1600, 256) 
S5: 400 vectors (1024-dim) → Resize to (400, 256)

Step 2: Concatenate
Combined = (6400 + 1600 + 400, 256) = (8400, 256)

Step 3: Multi-scale Transformer
- Each of 8,400 features attends to ALL 8,400 features
- Attention complexity: O(8400²) = O(70,560,000)
- VERY EXPENSIVE!

Memory required for attention:
8400 × 8400 = 70,560,000 float values
× 4 bytes = 282 MB per attention head!
```

### RT-DETR Efficient Hybrid Encoder (FAST)

```
Step 1: AIFI (Attention only on S5)
S5: (400, 1024) → Resize to (400, 256)
- Only 400 features attend to each other
- Complexity: O(400²) = O(160,000)
- 441× LESS computation than traditional!

Memory for attention:
400 × 400 = 160,000 float values
× 4 bytes = 640 KB per attention head

Step 2: CCFF (CNN Fusion)
S3 (6400, 256) ──┐
S4 (1600, 256) ──┼→ CNN operations (cheap!)
F5 (400, 256) ───┘

CNN operations: O(N × k²) where k=3 (kernel size)
Much cheaper than attention's O(N²)

Final Output: (8400, 256) - same as traditional but MUCH faster!
```

**Speed Comparison**:
- Traditional: 13.3 ms
- RT-DETR: 9.3 ms
- **30% faster!**

---

## 4. Attention Mechanism: Concrete Example

Let's see how one feature vector "attends to" others.

### Setup

Imagine we have 5 feature vectors (simplified to 3D):

```
f1 = [1.0, 0.0, 0.0]  (represents "cat")
f2 = [0.9, 0.1, 0.0]  (represents "cat, similar")
f3 = [0.0, 1.0, 0.0]  (represents "dog")
f4 = [0.0, 0.0, 1.0]  (represents "grass")
f5 = [0.1, 0.0, 0.9]  (represents "grass, similar")
```

### Attention Calculation for f1

**Step 1: Calculate similarities (dot products)**

```
Similarity(f1, f1) = f1 · f1 = 1.0×1.0 + 0.0×0.0 + 0.0×0.0 = 1.0
Similarity(f1, f2) = f1 · f2 = 1.0×0.9 + 0.0×0.1 + 0.0×0.0 = 0.9
Similarity(f1, f3) = f1 · f3 = 1.0×0.0 + 0.0×1.0 + 0.0×0.0 = 0.0
Similarity(f1, f4) = f1 · f4 = 1.0×0.0 + 0.0×0.0 + 0.0×1.0 = 0.0
Similarity(f1, f5) = f1 · f5 = 1.0×0.1 + 0.0×0.0 + 0.0×0.9 = 0.1
```

**Step 2: Normalize with softmax (convert to weights)**

```
Raw scores: [1.0, 0.9, 0.0, 0.0, 0.1]
After softmax: [0.45, 0.40, 0.05, 0.05, 0.05]
                ↑     ↑                      ↑
         High attention to f2            Low attention to others
```

**Step 3: Weighted combination**

```
Updated f1 = 0.45×f1 + 0.40×f2 + 0.05×f3 + 0.05×f4 + 0.05×f5
           = 0.45×[1.0,0.0,0.0] + 0.40×[0.9,0.1,0.0] + ...
           = [0.86, 0.04, 0.05]

Interpretation: f1 now "knows about" similar features (f2) 
and slightly aware of other features
```

**Result**: The cat feature is now enhanced with information from other cat-like features!

---

## 5. Query Selection: Concrete Example

### Scenario

Encoder outputs 8,400 feature vectors. We need to select 300 as initial queries.

### Feature Example Data (10 features shown)

| Feature ID | Class Score | IoU Score | Uncertainty | Selected? |
|------------|-------------|-----------|-------------|-----------|
| 1 | 0.95 | 0.90 | 0.05 | ✅ Yes (low uncertainty) |
| 2 | 0.88 | 0.85 | 0.03 | ✅ Yes (low uncertainty) |
| 3 | 0.92 | 0.45 | 0.47 | ❌ No (HIGH uncertainty) |
| 4 | 0.60 | 0.90 | 0.30 | ❌ No (HIGH uncertainty) |
| 5 | 0.78 | 0.82 | 0.04 | ✅ Yes (low uncertainty) |
| 6 | 0.95 | 0.25 | 0.70 | ❌ No (VERY HIGH uncertainty) |
| 7 | 0.70 | 0.72 | 0.02 | ✅ Yes (low uncertainty) |
| 8 | 0.45 | 0.88 | 0.43 | ❌ No (HIGH uncertainty) |
| 9 | 0.85 | 0.80 | 0.05 | ✅ Yes (low uncertainty) |
| 10 | 0.40 | 0.35 | 0.05 | ❌ No (both scores low) |

### Analysis

**Feature 1** (Selected ✅):
- Says: "I'm 95% confident there's an object" (Class: 0.95)
- Also says: "I know exactly where it is" (IoU: 0.90)
- Uncertainty: |0.95 - 0.90| = 0.05 ← LOW, GOOD!

**Feature 3** (Rejected ❌):
- Says: "I'm 92% confident there's an object" (Class: 0.92)
- But says: "I'm not sure where it is" (IoU: 0.45)
- Uncertainty: |0.92 - 0.45| = 0.47 ← HIGH, BAD!
- Problem: False confidence!

**Feature 6** (Rejected ❌):
- Says: "I'm 95% confident there's an object" (Class: 0.95)
- But says: "Location is probably wrong" (IoU: 0.25)
- Uncertainty: |0.95 - 0.25| = 0.70 ← VERY HIGH, VERY BAD!
- Problem: Confident but completely wrong location!

### Selection Process

```python
# Pseudo-code
for each of 8400 features:
    uncertainty = |classification_score - iou_score|
    
# Sort by uncertainty (ascending)
sorted_features = sort(features, key=uncertainty)

# Select top 300
initial_queries = sorted_features[0:300]

Result: 300 features where class and location AGREE
```

### Why This Matters

Compare two initialization strategies on same image:

**Vanilla (select by class score only)**:
```
Selected features might include:
- Feature 3: High class (0.92) but bad IoU (0.45)
- Feature 6: High class (0.95) but terrible IoU (0.25)

Decoder must work hard to fix bad initial guesses
Final AP: 47.9%
```

**Uncertainty-minimal (select by low uncertainty)**:
```
Selected features only include:
- Feature 1: High class (0.95) AND good IoU (0.90)
- Feature 2: High class (0.88) AND good IoU (0.85)

Decoder starts with good initial guesses
Final AP: 48.7% (+0.8% improvement!)
```

---

## 6. Decoder: Layer-by-Layer Example

### Initial State

```
Input to Decoder:
- Object Queries: 300 vectors, shape (300, 256)
- Image Features: 8,400 vectors, shape (8400, 256)

Initial Query 1: [0.23, -0.45, 0.89, ..., -0.33]  (256-dim)
Initial Query 2: [0.67, 0.12, -0.34, ..., 0.89]   (256-dim)
...
Initial Query 300: [-0.11, 0.78, 0.45, ..., 0.22] (256-dim)
```

### Layer 1 Processing

**Step 1: Self-Attention** (Queries interact with each other)
```
Query 1 attends to all 300 queries
→ Learns: "Query 2 is detecting something similar to me"
→ Updated Query 1

Query 2 attends to all 300 queries  
→ Learns: "Query 1 and Query 5 are nearby"
→ Updated Query 2

... (for all 300 queries)
```

**Step 2: Cross-Attention** (Queries gather info from image)
```
Query 1 looks at image features:
→ Attends strongly to features at position (45, 32) in S4
→ Learns: "There's likely a cat-like object there"
→ Further updated Query 1

Query 2 looks at image features:
→ Attends strongly to features at position (10, 15) in S3
→ Learns: "There's likely a person there"
→ Further updated Query 2
```

**Step 3: Feed-Forward** (Non-linear transformation)
```
Each query passes through 2-layer MLP:
Hidden dim: 1024
Output dim: 256

Query 1 → [Linear(256→1024)] → [ReLU] → [Linear(1024→256)] → Final Query 1
```

**Step 4: Make Predictions**
```
Query 1 → Prediction Head:
  - Class logits: [0.1, 0.2, ..., 8.5, ..., 0.3]  (80 classes)
                                ↑
                         High score for "cat" (class 17)
  - Bounding box: [0.45, 0.32, 0.15, 0.20]  (cx, cy, w, h normalized)
  
Query 2 → Prediction Head:
  - Class logits: [..., 9.2, ...]  
                      ↑
                High score for "person" (class 1)
  - Bounding box: [0.10, 0.15, 0.08, 0.25]
```

### Layer 2-6 (Iterative Refinement)

Each subsequent layer refines the predictions:

```
Layer 1 Output:
Query 1 predicts: Cat at (0.45, 0.32), confidence 0.78

Layer 2 Output:  
Query 1 predicts: Cat at (0.46, 0.31), confidence 0.85  (better!)

Layer 3 Output:
Query 1 predicts: Cat at (0.465, 0.315), confidence 0.91  (even better!)

...

Layer 6 Output:
Query 1 predicts: Cat at (0.467, 0.318), confidence 0.94  (final!)
```

### Final Output

```
300 predictions:
[
  {class: "cat", score: 0.94, box: [0.467, 0.318, 0.15, 0.20]},
  {class: "person", score: 0.91, box: [0.10, 0.15, 0.08, 0.25]},
  {class: "car", score: 0.88, box: [0.70, 0.65, 0.20, 0.18]},
  {class: "background", score: 0.12, box: [...]},  ← Will be filtered
  {class: "background", score: 0.08, box: [...]},  ← Will be filtered
  ...
]

Filter by confidence > 0.3:
Final detections: ~15-30 objects (depends on image)
```

---

## 7. Training: Bipartite Matching Example

### Ground Truth
```
Image has 3 objects:
GT1: Cat at [0.47, 0.32, 0.15, 0.20]
GT2: Person at [0.10, 0.15, 0.08, 0.25]
GT3: Car at [0.70, 0.65, 0.20, 0.18]
```

### Model Predictions (300 predictions)
```
Pred1: Cat, 0.94, [0.467, 0.318, 0.15, 0.20]
Pred2: Person, 0.91, [0.10, 0.15, 0.08, 0.25]
Pred3: Dog, 0.45, [0.50, 0.40, 0.12, 0.18]  ← Wrong class!
Pred4: Car, 0.88, [0.70, 0.65, 0.20, 0.18]
Pred5-300: Background, low scores
```

### Bipartite Matching (Hungarian Algorithm)

Compute cost matrix between predictions and ground truths:

```
Cost Matrix (lower is better):
           GT1(Cat)  GT2(Person)  GT3(Car)
Pred1       0.05      999         999
Pred2       999       0.08        999
Pred3       0.70      999         999
Pred4       999       999         0.10
Pred5-300   ~5        ~5          ~5

Optimal matching:
Pred1 ↔ GT1 (cost 0.05)
Pred2 ↔ GT2 (cost 0.08)
Pred4 ↔ GT3 (cost 0.10)

Unmatched predictions (Pred3, Pred5-300) → background loss
```

### Loss Calculation

```
Matched pairs:
1. Pred1 ↔ GT1:
   - Class loss: CrossEntropy(pred_cat, target_cat) = 0.06
   - Box loss: L1(pred_box, gt_box) + GIoU_loss = 0.12
   
2. Pred2 ↔ GT2:
   - Class loss: 0.09
   - Box loss: 0.08
   
3. Pred4 ↔ GT3:
   - Class loss: 0.13
   - Box loss: 0.15

Unmatched predictions:
- Pred3, Pred5-300: Class loss for "background"

Total Loss = Σ(matched losses) + Σ(background losses)
```

---

## 8. Complete Example: One Image Through RT-DETR

### Input
```
Image: cat_and_dog.jpg (640×640)
Ground truth: 
- Cat at [0.45, 0.30, 0.18, 0.22]
- Dog at [0.70, 0.60, 0.20, 0.25]
```

### Forward Pass

**1. Backbone**
```
(640,640,3) → ResNet50 → {S3:(80,80,256), S4:(40,40,512), S5:(20,20,1024)}
```

**2. Efficient Hybrid Encoder**
```
S5 (400 features) → AIFI → F5 (enhanced, 400 features)
{S3:6400, S4:1600, F5:400} → CCFF → Fused(8400, 256)

Time: 9.3 ms
```

**3. Query Selection**
```
Compute uncertainty for all 8400 features:
Feature 523: uncertainty = 0.03 ✅ (high class, high IoU)
Feature 1042: uncertainty = 0.04 ✅
...
Feature 7821: uncertainty = 0.52 ❌ (high class, low IoU)

Select top 300 with lowest uncertainty
→ Initial queries ready
```

**4. Decoder (6 layers)**
```
Layer 1: Rough predictions
- Query 15: "Maybe cat around (0.40, 0.25)?"
- Query 82: "Maybe dog around (0.68, 0.58)?"

Layer 3: Better predictions
- Query 15: "Cat at (0.44, 0.29), conf 0.85"
- Query 82: "Dog at (0.69, 0.61), conf 0.82"

Layer 6: Final predictions
- Query 15: "Cat at (0.45, 0.30), conf 0.93" ✅
- Query 82: "Dog at (0.70, 0.60), conf 0.91" ✅

Time: 9.3 ms (includes all 6 layers)
```

**5. Post-processing**
```
Filter predictions by confidence > 0.3:
- 2 valid detections (cat and dog)
- 298 background predictions (discarded)

NO NMS NEEDED!
Time: 0 ms
```

### Total Inference Time
```
Backbone: ~10 ms
Encoder: 9.3 ms
Decoder: 9.3 ms (already counted)
Post-processing: 0 ms
TOTAL: ~19.3 ms = ~52 FPS

(Actual paper reports 108 FPS with optimized TensorRT implementation)
```

---

## 9. Memory Footprint Comparison

### Traditional DETR (Deformable)

```
Encoder:
- Feature sequence: 8400 × 256 = 2,150,400 values
- Attention maps: 8400 × 8400 × 8 heads = 564,480,000 values
- Total encoder: ~2.3 GB per batch

Decoder (6 layers):
- Per layer: 300 × 8400 × 8 heads = 20,160,000 values
- Total decoder: ~500 MB per batch

TOTAL: ~2.8 GB per image
```

### RT-DETR

```
Encoder:
- AIFI attention: 400 × 400 × 8 heads = 1,280,000 values
- CCFF (CNN): Negligible (kernel weights)
- Total encoder: ~100 MB per batch

Decoder (same as traditional):
- Total decoder: ~500 MB per batch

TOTAL: ~600 MB per image

Savings: 78% less memory!
```

---

## 10. Key Formulas with Numbers

### Attention Complexity

```
Traditional: O(N²) = O(8400²) = O(70,560,000)
RT-DETR: O(N_S5²) = O(400²) = O(160,000)

Reduction factor: 70,560,000 / 160,000 = 441×
```

### FLOPs Calculation

```
Traditional Encoder:
Attention: 8400² × 256 × 2 = 36.2 GFLOPs
FFN: 8400 × (256 × 1024 + 1024 × 256) × 2 = 8.6 GFLOPs
Total: ~45 GFLOPs

RT-DETR Encoder:
AIFI: 400² × 256 × 2 = 0.08 GFLOPs
CCFF: 8400 × 256 × 9 (3×3 conv) × 3 scales = 0.6 GFLOPs
Total: ~0.7 GFLOPs

Speedup: 45 / 0.7 = 64× faster!
```

---

## Summary: The Magic Numbers

| Metric | Value | Meaning |
|--------|-------|---------|
| **Input** | 640×640×3 | Image size |
| **S5 features** | 400 | Only these get attention |
| **Total features** | 8,400 | All scales combined |
| **Attention reduction** | 441× | Fewer attention operations |
| **Queries** | 300 | Fixed predictions |
| **Decoder layers** | 6 | Iterative refinement |
| **Final AP** | 53.1% | Better than YOLOv8-L |
| **FPS** | 108 | Faster than YOLOv8-L |
| **Memory savings** | 78% | vs traditional DETR |

The key insight: By carefully choosing WHICH 400 features get attention (only S5) and using fast CNNs for fusion, RT-DETR achieves 441× reduction in attention operations while maintaining accuracy!

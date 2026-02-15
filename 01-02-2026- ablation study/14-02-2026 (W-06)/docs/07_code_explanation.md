# Code Explanation: YOLO11m_CBAM_P2Head.py

> **Purpose:** This file breaks down the script `YOLO11m_CBAM_P2Head.py` line by line. It explains **what** the code does, **why** we wrote it that way, and **how** the complex parts (like CBAM and P2-Head) work under the hood.

---

## Cell 1: Configuration & Dependencies

```python
# ===================== CONTROL FLAGS =====================
TEST_MODE = True           # True = 5% data, 2 epochs | False = full run
RESUME_TRAINING = False    # Set True to resume CBAM+P2 from checkpoint
RESUME_CBAM_P2_PT = ""     # Path to cbam+p2 last.pt (if resuming)
```

### Explanation
- **`TEST_MODE`**: A master switch.
  - `True`: Runs a "Dry Run" (fast). We use only 5% of the data and train for 2 epochs. This is critical for debugging before committing to a 12-hour training run.
  - `False`: Runs the "Full Training" (slow). Uses 100% data and 70 epochs.
- **`RESUME_TRAINING`**: If your Kaggle notebook times out (after 12 hours), you can set this to `True` and provide the path to `last.pt` to continue exactly where you left off.

### Dependencies
```python
!pip install -q -U ultralytics timm thop ...
```
- **`ultralytics`**: The core YOLO library.
- **`timm`**: "PyTorch Image Models" — needed for some backbone features.
- **`thop`**: "Torch-Op" — calculates GFLOPs (how heavy the model is).

### Data Types & Operations
- **`TEST_MODE` (bool):** A simple True/False flag.
  - Operation: `if TEST_MODE:` checks control the flow.
- **`!pip install` (System Command):** This runs in the shell, not Python.
  - `> /dev/null`: Redirects output to "nowhere" (silences the logs).
  - `2>&1`: Redirects errors to standard output.

> **See Also:** For a list of all papers and libraries used, check **[docs/06_references.md](06_references.md)**.

---

## Cell 2: Training Config & Auto-Detection

### GPU Memory Logic
```python
gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
# ...
p2_batch = 8 if gpu_mem >= 14 else 4
```
- **Why?** The P2 Head creates very large feature maps (160×160). This consumes a lot of Video RAM (VRAM).
- **Tesla P100 (16GB)** or **T4 (15GB)** can handle `batch=8`.
- Smaller GPUs (like local RTX 3060) might need `batch=4` to avoid crashing.

### Auto-Detecting Data
```python
for root, dirs, files in os.walk("/kaggle/input"):
    if os.path.isfile(os.path.join(root, "runs/detect/.../best.pt")):
        INPUT_DIR = root
        break
```
- **Problem:** Kaggle changes dataset paths sometimes (e.g., `/kaggle/input/dataset-v2` vs `/v3`).
- **Solution:** We use `os.walk` (recursive search) to find *exactly* where your uploaded models are, no matter the folder name.

---

## Cell 3: Dataset Configuration (`c2a.yaml`)

```python
dataset_yaml_content = """train: /path/to/train/images
val: /path/to/val/images
test: /path/to/test/images
nc: 1
names: ['person']
"""
```
- **YAML (Yet Another Markup Language)**: This file tells YOLO where to find the images.
- **`nc: 1`**: We are detecting only **one class**: "person" (for disaster victims).

---

## Cell 4: The CBAM Module (The "Brain")

This is the most complex part. We define a custom PyTorch module that implements **Attention**.

### Class `CBAM`
```python
class CBAM(nn.Module):
    def __init__(self, *args, **kwargs):
        # ... sets up kernel_size=7, reduction=16 ...
        self._initialized = False

    def _lazy_init(self, channels, ...):
        # ... builds the layers using 'channels' ...
        self.channel_attention = ChannelAttention(channels, ...)
        self.spatial_attention = SpatialAttention(...)
```

### Why "Lazy Initialization"?
- **Standard PyTorch:** You must manually tell a layer "Expect 512 input channels".
- **YOLO Yaml:** We just write `CBAM` in the list. The YAML parser *doesn't* tell us the input channels easily.
- **Solution:** We make the module **lazy**. It sits empty until the first image passes through.
  1. Image arrives with shape `(Batch, 1024, 20, 20)`.
  2. `CBAM` sees `1024` channels.
  3. `_lazy_init(1024)` is called.
  4. It builds the weights *just in time*.

### Why "Lazy Initialization"?
- **Problem:** Standard PyTorch needs `in_channels` defined upfront (e.g., `nn.Conv2d(512, ...)`).
- **YOLO Context:** The YAML parser builds layers sequentially. We don't easily know in Python that "Layer 10 has 512 channels".
- **Solution:** We make `_lazy_init` method.
  - **Input Tensor `x`**: Has shape `(Batch_Size, Channels, Height, Width)`.
  - Example: `x.shape = (8, 512, 20, 20)`.
  - We read `x.shape[1]` (which is 512) and initialize weights *then*.

> **Deep Dive:** See **[docs/03_cbam_integration.md](03_cbam_integration.md)** for diagram and math.

---

## Cell 5: Registering CBAM (Monkey Patching)

```python
import ultralytics.nn.modules as modules
modules.CBAM = CBAM  # <--- The Magic
```
- **Operation:** We inject our class into the `ultralytics` library at runtime.
- **Why?** So when the YAML parser sees `CBAM`, it looks in `modules` and finds *our* class, not an error.

---

## Cell 6: Generating the Architecture (YAML)

This cell writes the `yolov11m_cbam_p2head.yaml` file. It's a recipe for the model.

### The Backbone Change (Adding CBAM)
```yaml
# Standard YOLOv11m
- [-1, 2, C2PSA, [1024]]

# Our Modification
- [-1, 1, CBAM, [16, 7]]  # Replaces C2PSA
```
- **C2PSA:** Uses specialized "Partial Spatial Attention" (complex, heavy).
- **CBAM:** Uses our custom "Channel + Spatial Attention" (lightweight, simple).

### The P2 Head Layers (Detailed)
```yaml
# 1. Upsample P3 (80x80) -> 160x160
- [-1, 1, nn.Upsample, [None, 2, "nearest"]]
```
- **Operation:** "Nearest Neighbor" resizing.
- **Why?** We need to match the size of the P2 backbone feature map (160×160) before we can mix them.

```yaml
# 2. Concat with Backbone Layer 2 (P2)
- [[-1, 2], 1, Concat, [1]]
```
- **Operation:** Stacks tensors along dimension 1 (Channels).
- **Shape Change:** `(B, 256, 160, 160)` + `(B, 128, 160, 160)` → `(B, 384, 160, 160)`.
- **Why?** Merges "semantic meaning" (from Head) with "fine detail" (from Backbone).

> **Deep Dive:** See **[docs/02_yaml_configuration.md](02_yaml_configuration.md)** for full YAML syntax and **[docs/04_p2head_integration.md](04_p2head_integration.md)** for P2 logic.

### The Detect Layer
```yaml
- [[19, 22, 25, 28], 1, Detect, [nc]]
```
- **Indices:**
  - `19`: P2 scale (High resolution, for tiny objects) **[NEW]**
  - `22`: P3 scale (Medium-High)
  - `25`: P4 scale (Medium-Low)
  - `28`: P5 scale (Low resolution, for large objects)

---

## Cell 9: Training the CBAM + P2 Model

```python
# Create model from YAML (architecture only, random weights)
cbam_p2_model = YOLO("yolov11m_cbam_p2head.yaml")

# Load pretrained weights (transfer learning)
cbam_p2_model.load("yolo11m.pt")
```

### Why Load Pretrained Weights?
- **From Scratch:** The model starts "dumb" (random weights). It takes days to learn basic features (edges, shapes).
- **Transfer Learning:** loading `yolo11m.pt` gives it a head start. It already knows how to see objects. We just finetune it to detect people and use our new P2 head.

### Training Command Parameters
```python
cbam_p2_model.train(
    data="c2a.yaml",
    epochs=NUM_EPOCHS,
    imgsz=640,
    batch=p2_batch,      # [8] for T4 (15GB), [4] for smaller GPUs
    amp=True,            # Automated Mixed Precision (float16)
    cache=False,         # Data loading strategy
    workers=4,           # CPU threads
)
```
- **`amp=True`**: Uses `float16` (half precision) instead of `float32`.
  - **Benefit:** 2x faster, 50% less VRAM.
  - **Risk:** Slight loss of precision (negligible for YOLO).
- **`cache=False`**: Reads images from disk every time.
  - **True:** Loads all 10,000 images into RAM (fast, but crashes if RAM < 32GB).
  - **False:** Conservative memory usage.
- **`workers=4`**: Uses 4 CPU cores to unzip/resize images while the GPU trains.

> **Hardware Note:** See **[docs/04_p2head_integration.md](04_p2head_integration.md)** for more on the batch size calculation.

---

## Cell 10: Calculating Total Loss

The standard YOLO log file `results.csv` has separate columns for box, class, and dfl loss. We want the **Total Loss** to visualize convergence.

```python
# Summing columns
# train/box_loss + train/cls_loss + train/dfl_loss = train/total_loss
df["train/total_loss"] = df[train_loss_cols].sum(axis=1)
df["val/total_loss"] = df[val_loss_cols].sum(axis=1)
```

This lets us plot a single curve to see "is the model getting better overall?".

---

## Cell 11: Complexity Comparison

We compare the "cost" of our models.

```python
from thop import profile
# ...
flops, params = profile(model, inputs=(dummy_input,))
```

- **Parameters:** The number of learnable weights (neurons). More params = smarter but slower.
- **GFLOPs (Giga Floating Point Operations):** The number of math operations per image. This measures **speed**.
  - **Standard YOLO11m:** ~20M params, ~68 GFLOPs
  - **P2 Head Addition:** ~19.6M params, ~87 GFLOPs.
  - **Wait, fewer params?** Yes! The P2 layers we added use fewer channels (128) than the deep layers (512/1024) they replaced/augmented.
  - **But GFLOPs go UP:** Because the P2 feature map is huge (160×160), we do *more* math operations, even with fewer weights.

---

## Cell 14: Comprehensive Evaluation Pipeline

This is our custom evaluation function `evaluate_model_comprehensive`. It's much more detailed than standard YOLO `val()`.

### Why Custom Eval?
Standard `model.val()` gives just one number: mAP. We want to know:
1. **Per-Size Recall:** "Does it find tiny objects?"
2. **Failure Analysis:** "When does it be confident but wrong?"
3. **Calibration:** "Does 90% confidence actually mean 90% accuracy?"

### The Pipeline Steps (Data Flow)

1. **Prediction**
   ```python
   preds = model.predict(img_path)
   # preds[0].boxes.xyxy: GPU Tensor (float32) -> .cpu() -> .numpy()
   ```
   - **Data Type:** We move tensors from GPU (`cuda:0`) to CPU because `numpy` (used for math) doesn't work on GPU.

2. **Match Predictions to Ground Truth**
   ```python
   tp, fp, fn = match_predictions_to_gt(pred_boxes, gt_boxes)
   ```
   - **IoU (Intersection over Union):** Overlap area / Union area.
   - **Threshold:** We use 0.5. If IoU > 0.5, it's a match/TP.

3. **Categorize by Size**
   ```python
   # Area < 8*8 = 64 pixels -> "Very Tiny"
   'very_tiny': areas < 64
   ```
   - **Masking:** `areas < 64` creates a boolean array (`[True, False, ...]`) to filter boxes.
   - **Recall Calculation:** `TP_tiny / (TP_tiny + FN_tiny)`. This gives us the specific performance for that size bucket.

---

## Cell 18: Calibration & Failure Modes

### Calibration (ECE)
We compare predicted **Confidence** vs actual **Precision**.
- **Perfect:** If confidence is 0.8, precision should be 0.8.
- **Overconfident:** Confidence 0.9, but precision only 0.5. (Bad!)
- **ECE:** The average gap between confidence and precision. Lower is better.

### Failure Modes
We look for "Dangerous" errors:
- **High Confidence False Positives:** The model says "Definitely a person!" (0.9 conf) but it's a rock. This wastes rescue resources.
- **High Confidence False Negatives:** The model is "sure" there's nothing, but there is a person. (Tragic).

---

## Cell 20: Master Report

This cell gathers all the numbers from Baseline, CBAM, and CBAM+P2 and prints a final summary.

### Actual Output (from our run):
```text
DISASTER HUMAN DETECTION: CBAM + P2 HEAD ABLATION STUDY (3-WAY)
================================================================
CONFIG: Test=False | Epochs=70 | Fraction=100% | ImgSz=640

MODEL COMPLEXITY:
  Baseline:    20,053,779 params |   34.1 GFLOPs
  CBAM:        19,095,669 params |   33.7 GFLOPs (-4.8%)
  CBAM+P2:     19,592,246 params |   43.7 GFLOPs (-2.3%)

OFFICIAL mAP (test split):
  Baseline: mAP50=0.8558 mAP50-95=0.6256
  CBAM:     mAP50=0.8557 mAP50-95=0.6230
  CBAM+P2:  mAP50=0.8723 mAP50-95=0.6418

SMALL OBJECT RECALL (TEST):
  Very Tiny (<8²px):  Base=0.7839 | CBAM=0.7812 | CBAM+P2=0.8089  ← +2.5%!
  Tiny (8-16px):      Base=0.8901 | CBAM=0.8972 | CBAM+P2=0.8972
  Small (16-32px):    Base=0.8807 | CBAM=0.8920 | CBAM+P2=0.8864

BEST MODEL: yolo11m_cbam_p2 (very-tiny recall priority)
```

> **Full analysis:** See [docs/08_results_analysis.md](08_results_analysis.md) for interpretation.

---

## Conclusion
This script is a complete **Ablation Study system**. It systematically:
1. Builds 3 variations of the model.
2. Trains the CBAM+P2 variant (loading others from disk).
3. Evaluates all 3 on the exact same test set.
4. Generates a "Master Report" proving which one is best for **tiny object detection**.

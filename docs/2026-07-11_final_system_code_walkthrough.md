# 2026-07-11 — Code walkthrough: how CBAM + P2 + SAHI + TTA are actually implemented

Q&A prep: which code block defines which component, and how each is inserted.
Sources (all on D:, verbatim from the runs that produced the report numbers):
- **Training / model definition:** `Last Month/24_01_26- Benchmarking YOLOs/CBAM_P2Head/yolo11m_cbam_p2head_thesis.py` (the 20260602 run)
- **SAHI + TTA study:** same folder, `sahi_tta_cbam_p2_thesis.py`
- **CBAM classes (demo copy, verbatim snapshot):** `Defense/demo/app/cbam_modules.py`
- **Weights:** `Defense/demo/models/cbam_p2head.pt` (training checkpoint) and `cbam_p2head.onnx` (export for the CPU demo engine)

---

## 1. The CBAM component — three classes (`cbam_modules.py` 17–90)

### 1a. `ChannelAttention` (lines 17–32) = the M_c equation
```python
self.avg_pool = nn.AdaptiveAvgPool2d(1)      # AvgPool(F): each channel → 1 number
self.max_pool = nn.AdaptiveMaxPool2d(1)      # MaxPool(F): each channel → 1 number
self.fc1 = nn.Conv2d(channels, reduced, 1)   # MLP layer 1: 512 → 32  (reduction 16)
self.fc2 = nn.Conv2d(reduced, channels, 1)   # MLP layer 2: 32 → 512
...
avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
return x * self.sigmoid(avg_out + max_out)   # M_c(F) ⊗ F  — one weight per channel
```
Line-by-line ↔ formula: `avg_pool/max_pool` are the two pooled descriptors; `fc1→relu→fc2`
is the SHARED MLP (shared because both descriptors pass through the *same* fc1/fc2 —
that's the "+" inside σ in the equation); `sigmoid(avg+max)` is M_c; `x * …` is the ⊗.
**Q&A detail:** the "MLP" is implemented as two 1×1 convolutions — on a 1×1 pooled map a
1×1 conv IS a fully-connected layer, but it keeps everything in NCHW tensors (no
flatten/reshape). Parameter count: 512×32 + 32×512 ≈ 33 k.

### 1b. `SpatialAttention` (lines 35–48) = the M_s equation
```python
self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)   # f^{7×7}
...
avg_out = torch.mean(x, dim=1, keepdim=True)   # pool ACROSS channels → 1×H×W
max_out, _ = torch.max(x, dim=1, keepdim=True) # ditto
concat = torch.cat([avg_out, max_out], dim=1)  # [AvgPool;MaxPool] → 2×H×W
return x * self.sigmoid(self.conv(concat))     # M_s(F′) ⊗ F′ — one weight per pixel
```
`dim=1` is the channel dimension — that's what makes this SPATIAL attention: pooling
collapses channels, keeping the H×W layout, and the 7×7 conv scores each location from
its neighbourhood.

### 1c. `CBAM` wrapper (lines 51–90) — sequential application + **lazy init**
```python
def forward(self, x):
    if not self._initialized:
        self._lazy_init(x.size(1), x.device, x.dtype)   # discover channels at runtime
    x = self.channel_attention(x)     # F′ = M_c ⊗ F   (channel FIRST)
    x = self.spatial_attention(x)     # F″ = M_s ⊗ F′  (spatial SECOND)
    return x
```
**Why lazy initialization? (a likely committee question).** Ultralytics builds models
from YAML via `parse_model()`, which does channel bookkeeping for *known* modules. A
custom module would need patches inside the library to receive its input-channel count.
Instead, our CBAM defers creating its sub-modules until the FIRST forward pass, reads
`x.size(1)` (=512 at layer 10), and builds itself to match. No library surgery, and the
module adapts to any placement. The odd-looking `__init__` argument parsing exists so
the same class works whether YAML passes `[16, 7]` (reduction, kernel) or parse_model
passes channel args.

---

## 2. Registration — making Ultralytics accept a class it doesn't know
(`register_cbam()`, training script 434–448; demo mirror `cbam_modules.register()` 106–135)

Two separate problems, one function:
1. **YAML parsing:** when `parse_model` reads the row `[-1, 1, CBAM, [16, 7]]` it
   resolves the *name* "CBAM" inside `ultralytics.nn.modules` / `nn.tasks`. So we
   `setattr` our class into both namespaces BEFORE `YOLO(yaml)` is called.
2. **Checkpoint unpickling:** `torch.save` pickles *references* — "class CBAM from
   module X". Loading `cbam_p2head.pt` later fails unless a module named X with a CBAM
   attribute exists. The demo's `register()` therefore also injects the classes into
   `__main__` and creates **stub modules** (`yolo11m_cbam_p2head_thesis`,
   `joint_c2a_sard_train`, …) matching every module name a checkpoint may have been
   pickled under, plus `torch.serialization.add_safe_globals` for torch ≥ 2.6.
**One-liner for Q&A:** *"The class is registered into Ultralytics' namespaces before
build, and into stub modules before load — YAML resolves the name at build time, pickle
resolves it at load time."*

---

## 3. The architecture itself — one YAML string (training script 456–501)

The ENTIRE proposed model is declared as `CBAM_P2HEAD_YAML`, based on Ultralytics'
official `yolo11-p2.yaml` plus the CBAM swap. This maps 1:1 to report Table 3.1:

```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]          # 0   P1/2 stem
  - [-1, 1, Conv, [128, 3, 2]]         # 1   P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]  # 2   ← P2 SKIP SOURCE (160²×256)
  - [-1, 1, Conv, [256, 3, 2]]         # 3   P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]  # 4   ← P3 skip source
  - [-1, 1, Conv, [512, 3, 2]]         # 5   P4/16
  - [-1, 2, C3k2, [512, True]]         # 6   ← P4 skip source
  - [-1, 1, Conv, [1024, 3, 2]]        # 7   P5/32
  - [-1, 2, C3k2, [1024, True]]        # 8
  - [-1, 1, SPPF, [1024, 5]]           # 9
  - [-1, 1, CBAM, [16, 7]]             # 10  ★ CBAM REPLACES C2PSA (mod #1)
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 11  FPN top-down begins
  - [[-1, 6], 1, Concat, [1]]                    # 12  cat P4 skip
  - [-1, 2, C3k2, [512, False]]                  # 13
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 14
  - [[-1, 4], 1, Concat, [1]]                    # 15  cat P3 skip
  - [-1, 2, C3k2, [256, False]]                  # 16  (P3/8 fused)
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 17  ★ NEW — P2 branch (mod #2)
  - [[-1, 2], 1, Concat, [1]]                    # 18  ★ cat layer-2 P2 skip
  - [-1, 2, C3k2, [128, False]]                  # 19  ★ P2/4 feature 160²×128
  - [-1, 1, Conv, [128, 3, 2]]                   # 20  PAN bottom-up begins
  - [[-1, 16], 1, Concat, [1]]                   # 21
  - [-1, 2, C3k2, [256, False]]                  # 22  (P3 out)
  - [-1, 1, Conv, [256, 3, 2]]                   # 23
  - [[-1, 13], 1, Concat, [1]]                   # 24
  - [-1, 2, C3k2, [512, False]]                  # 25  (P4 out)
  - [-1, 1, Conv, [512, 3, 2]]                   # 26
  - [[-1, 10], 1, Concat, [1]]                   # 27  cat P5 (post-CBAM!)
  - [-1, 2, C3k2, [1024, True]]                  # 28  (P5 out)
  - [[19, 22, 25, 28], 1, Detect, [nc]]          # 29  ★ FOUR-scale head
```
Reading a row: `[from, repeats, module, args]` — `-1` = previous layer, `[[-1, 2]…]` =
take previous AND layer 2 (that's how skip connections are wired: **a skip IS just a
`Concat` whose `from` list names an earlier layer**).
- **CBAM insertion** = row 10: the baseline's `C2PSA` row replaced textually; nothing
  else in the backbone changes.
- **P2 head insertion** = rows 17–19 (one extra upsample-concat-fuse) + row 29 listing
  **[19, 22, 25, 28]** instead of the baseline's three — that single list is what makes
  the head four-scale.
- Channel numbers here are pre-scaling; YOLO11m's width multiplier + 512 cap produce
  the realized counts in report Table 3.1.
- `build_cbam_p2head_yaml()` (505–514) writes this string and **validates it**: asserts
  the Detect row has 4 scales and a CBAM row exists — a guard against silent
  misconfiguration.

---

## 4. Building the model + pretrained-weight transfer (517–530)

```python
register_cbam()                                  # names resolvable (Section 2)
yaml_path = build_cbam_p2head_yaml(...)          # write + validate the YAML
model = YOLO(str(yaml_path))                     # build architecture from YAML
model.load(pretrained_weights)                   # transfer yolo11m.pt weights
```
`model.load()` copies every pretrained tensor whose **name and shape match** the new
architecture: the whole backbone (except CBAM) and most of the neck arrive pretrained;
the CBAM module, the P2 branch (17–19), and the P2 part of the Detect head start from
random init and are learned during the 300-epoch schedule.

**★ The most important implementation Q&A point:** the model is defined *declaratively
in YAML*, NOT by mutating a built model in Python. Why that matters: Ultralytics'
`train()` REBUILDS the model from its config — any module injected after initialization
is silently discarded. That is exactly the fault that invalidated the first Mamba
attempt (it trained plain CBAM+P2 while believing it trained Mamba). The YAML route
survives every rebuild because the config itself contains the modification; after that
incident the protocol also gained **module-presence verification** (count CBAM/Detect
scales after build) and runtime activity checks (Section 6).

Training deltas for this config (script constants): `BATCH_SIZE = 8` — the stride-4 maps
are 4× larger than P3's, so physical batch halves and gradient accumulation holds the
effective batch at 16; optimizer pinned to AdamW lr0 = 0.001 (default SGD diverged on
this architecture); everything else identical to the other three runs.

---

## 5. SAHI — the slicing study (`sahi_tta_cbam_p2_thesis.py`)

Configuration (lines 80–87):
```python
SAHI_CONFIGS = [ slice 256 / 320 / 512 / 640, overlap 0.25–0.30 ]
SAHI_POSTPROCESS = dict(perform_standard_pred=True,       # + one full-image pass
                        postprocess_type="GREEDYNMM",     # greedy non-max MERGING
                        postprocess_match_metric="IOS",   # intersection-over-smaller
                        postprocess_match_threshold=0.5)
```
Execution (lines 599–605):
```python
from sahi.predict import get_sliced_prediction
res = get_sliced_prediction(img_path, sahi_model,
                            slice_height=slice_px, slice_width=slice_px,
                            ..., **SAHI_POSTPROCESS)
```
What each knob is for:
- `slice_height/width` — the tile size in ORIGINAL-image pixels; each tile is resized to
  the model's 640 input, which is what magnifies tiny people (256-px tile → 2.5×).
- `perform_standard_pred=True` — also run the plain full-image pass, so large objects
  that no single tile fully contains aren't lost.
- `GREEDYNMM` + `IOS` @ 0.5 — merging (not suppression) of per-tile boxes mapped back to
  original coordinates; IOS handles tile-border half-boxes (guide §4.6).
- Low confidence floor (`FLOOR_CONF`) at tile level; the per-image protocol scores at
  IoU 0.5 afterwards — SAHI results use per-image matching, which is why the report
  keeps them in a separate table from the COCO-protocol Table VII.
- The script even supports per-tile TTA via a context manager that injects
  `augment=True` into every SAHI tile prediction (line 616) — that's the "SAHI+TTA"
  mode of the qualitative figure.

## 6. TTA — three lines of configuration (same script)

```python
TTA_IMG_SIZES = [640, 832, 1280, 1920]          # 1920 auto-skips on OOM
r = model.predict(img, conf=FLOOR_CONF, imgsz=imgsz, augment=True)   # per-image
official_val(best_pt, data_yaml, sz, augment=True, batch)            # official mAP rows
```
**There is no custom TTA code** — `augment=True` IS Ultralytics' built-in test-time
augmentation: it internally evaluates scales {1.0, 0.83, 0.67} of the given `imgsz`,
each with a horizontal flip, and NMS-merges the union. Our contribution is the
*measurement*: sweeping `imgsz` 640→1920 under the same protocol, finding 1280 optimal
(very-tiny recall 0.850 @ 60 ms) and 1920 degraded — the "collapses beyond 2× training
resolution" finding. TTA rows use official `val(augment=True)` mAP so they're directly
comparable to published numbers; batch shrinks with size (`TTA_VAL_BATCH`: 8→1).

**"Why can a 640-trained model run at 1280?"** — nothing in the network stores an input
size; convolutions slide over any H×W and the Detect head reads whatever grid comes out
(160²→320² at 1280). Only feature *statistics* limit how far you can stretch — hence the
2× ceiling.

## 7. Verification hooks — proving the modules actually run

`cbam_attention_metrics()` (training script 533+): registers forward hooks on every
CBAM module during test images and writes `channel_attention_weights.csv`,
`spatial_attention_entropy.json`, and the attention heat-map overlays — **the real
attention map on your slides comes from this hook** (and the demo's
`make_arch_figure_images.py` uses the same trick on `layers[10].spatial_attention.sigmoid`).
This is the CBAM counterpart of the Mamba scan diagnostics (fwd/rev cosine distance
0.836): after the injection bug, *every* architectural claim is backed by a runtime
activity measurement, not just a printed layer list.

---

## 8. Likely code questions → one-breath answers

- **"How exactly is CBAM inserted?"** → "Declaratively: row 10 of the model YAML says
  CBAM instead of C2PSA; the class is registered into Ultralytics' module namespace so
  the parser resolves it; it lazy-initializes to the incoming 512 channels on first
  forward."
- **"How is the P2 head added?"** → "Three YAML rows — upsample, concat with the
  backbone layer-2 skip, C3k2 fuse — plus listing layer 19 in the Detect row. Based on
  Ultralytics' official yolo11-p2 config, so it's a supported pattern, not a hack."
- **"Did you train from scratch?"** → "No — YOLO(yaml).load(yolo11m.pt) transfers every
  name-and-shape-matching pretrained tensor; only CBAM, the P2 branch, and the P2 detect
  slice start fresh."
- **"How do you know your modules were really in the trained model?"** → "Because our
  first state-space attempt taught us they can silently vanish: train() rebuilds from
  config. Since then: YAML-declared architecture, post-build module-presence assertions,
  and runtime hooks — attention statistics for CBAM, scan-disagreement diagnostics for
  Mamba."
- **"What does GREEDYNMM/IOS do?"** → "Greedy merging of per-tile boxes using
  intersection-over-smaller, threshold 0.5 — IoU under-scores a border-cut half-box
  against its full box (0.5), IOS scores it 1.0, so cut people merge instead of
  duplicating."
- **"Is your TTA custom?"** → "No — Ultralytics' augment=True (3 scales × flip + NMS).
  The contribution is the controlled sweep showing 1280 is optimal and 1920 collapses,
  and that it beats every SAHI setting at a third of the cost."
- **"Why batch 8 for P2 models?"** → "The stride-4 feature map is 4× the area of
  stride-8; VRAM doubles roughly; physical batch dropped to 8 with gradient accumulation
  holding the effective batch at 16, so optimization is unchanged."

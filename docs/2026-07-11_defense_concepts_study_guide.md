# 2026-07-11 — Defense study guide: every concept & formula in the slides, explained

Written for defense Q&A prep. Everything is tied to YOUR numbers (report tables) so you
can quote them. Read top to bottom once, then use as lookup during Q&A prep.

---

## 1. Backbone, Neck, Head — the anatomy of YOLO

```
IMAGE 640×640
   │
   ▼
BACKBONE  (the eyes — extracts features)          what each level "sees"
   Conv stem (stride 2)      → P1 320×320         edges, dots
   Conv + C3k2 (stride 2)    → P2 160×160         textures, blobs      ← skip kept!
   Conv + C3k2 (stride 2)    → P3  80×80          parts, small shapes  ← skip kept!
   Conv + C3k2 (stride 2)    → P4  40×40          objects              ← skip kept!
   Conv + C3k2 + SPPF (s2)   → P5  20×20          whole-scene context
   C2PSA (baseline) / CBAM (ours)                 attention on P5
   │
   ▼
NECK  (the mixer — PAN-FPN)
   FPN top-down:  P5 → upsample → concat P4 skip → C3k2 → upsample → concat P3 skip …
                  (meaning flows DOWN: deep semantics reach the high-res maps)
   ── our extension: one MORE top-down step → concat backbone-P2 skip → 160²×128 map
   PAN bottom-up: downsample and re-concat back up P3 → P4 → P5
                  (details flow back UP: precise localization reaches the deep maps)
   │
   ▼
HEAD  (the decision maker)
   At each scale (ours: strides 4, 8, 16, 32 → grids 160², 80², 40², 20²),
   every grid cell predicts: box offsets (via DFL bins) + a person-confidence score.
   │
   ▼
NMS → final boxes
```

- **Why a neck at all?** Deep features (P5) know *what* things are but are spatially
  coarse; shallow features (P2/P3) know *where* edges are but not what they belong to.
  The neck fuses both so every detection scale gets meaning AND precision.
- **C3k2** — YOLO11's basic building block: a CSP ("split-transform-merge") residual
  block. It splits channels, passes part through small bottleneck convolutions, and
  concatenates back. Cheap, gradient-friendly feature refinement. It appears in both
  backbone and neck.
- **SPPF** (Spatial Pyramid Pooling–Fast) — at the end of the backbone, applies a 5×5
  max-pool three times in sequence and concatenates, gathering context from several
  receptive-field sizes at almost no cost.
- **C2PSA** — YOLO11's attention stage on P5: CSP wrapper around **P**osition-**S**ensitive
  **A**ttention blocks (transformer-style multi-head self-attention + feed-forward, two
  stacked at the m scale). Powerful but heavy: self-attention cost grows with the square
  of the number of positions. **This is what we remove** — replacing it with one CBAM
  saves ≈ 1 M parameters (20.03 M → 19.08 M) and even reduces latency (13.7 → 13.5 ms).

---

## 2. One-stage vs two-stage detectors

- **Two-stage** (R-CNN → Faster R-CNN → Cascade R-CNN): Stage 1, a Region Proposal
  Network scans the image and proposes ~1000 "maybe an object here" boxes. Stage 2, each
  proposal is cropped, resized, and classified + refined individually. Accurate (each
  candidate gets dedicated computation) but slow and heavy — you run the classifier
  hundreds of times per image.
- **One-stage** (SSD, RetinaNet, YOLO): no proposals — a fixed grid over the image, and
  EVERY cell directly predicts "is there an object centered near me, and what box?" One
  forward pass total. Fast and small; the historical accuracy gap closed once focal loss
  (RetinaNet) fixed the foreground/background imbalance of dense grids.
- **Your defense line:** *"We need airborne real-time, so one-stage. And on this
  benchmark the trade is not even a sacrifice — our 19.6 M one-stage model at AP50 0.853
  beats every published two-stage model on C2A: Faster R-CNN 0.634 @ ~41 M, Cascade
  R-CNN 0.735 @ ~69 M."* (Table IX.)

---

## 3. Feature pyramid & strides — what stride actually means

**Stride d = how many input pixels one grid cell covers.** A stride-8 map has one cell
per 8×8 pixel patch. At 640 input: P2 (s4) → 160×160 cells, P3 (s8) → 80×80,
P4 (s16) → 40×40, P5 (s32) → 20×20.

**Coverage formula c_d(s) = s/d** — how many cells a target of side s spans:

| person size s | @P5 (d=32) | @P4 (d=16) | @P3 (d=8) | @P2 (d=4) |
|---|---|---|---|---|
| 6 px (very-tiny) | 0.19 | 0.38 | 0.75 | **1.5** |
| 12 px (tiny) | 0.38 | 0.75 | 1.5 | **3.0** |
| 24 px (small) | 0.75 | 1.5 | 3.0 | 6.0 |

A detector cannot localize what lives inside a *fraction* of one cell — the cell's
features are a summary of the whole patch, and one 6-px person is 1% of a P5 cell's
area. Since **99.6 % of C2A instances are < 32 px**, the standard P3–P5 head is
structurally mismatched to this dataset; stride 4 is the first scale where the bulk of
the targets span ≥ 1 full cell. That's the entire P2 hypothesis in one table.

Why keep P4/P5 at all? Context (the neck pulls their semantics down), the rare 0.4 % of
larger targets, and stability of multi-scale training.

---

## 4. Every formula, with numbers

### 4.1 IoU — and why tiny boxes make it twitchy
IoU(B̂,B) = area(B̂∩B) / area(B̂∪B). 1 = perfect, 0 = disjoint; ≥ 0.5 counts as correct.

**Worked example (10×10-px person, prediction shifted right by k px):**
| shift k | intersection | union | IoU | verdict @0.5 |
|---|---|---|---|---|
| 1 px | 9×10 = 90 | 110 | 0.82 | ✔ |
| 3 px | 7×10 = 70 | 130 | 0.54 | ✔ (barely) |
| 5 px | 5×10 = 50 | 150 | 0.33 | ✘ |

A **3-pixel** error nearly fails a 10-px box. The same 3-px shift on a 100-px box gives
IoU 0.94. **This asymmetry is why:** (a) AP averaged over strict IoU thresholds
(0.5–0.95) is nearly flat across our configs — at this scale it measures annotation
noise; (b) we lead with AP50 and recall instead; (c) the CIoU loss adds centre/shape
terms — pure IoU gradients are unstable for few-px boxes.

### 4.2 Precision, Recall, F1, F2
- P = TP/(TP+FP): *of the boxes I output, what fraction were real people?*
- R = TP/(TP+FN): *of the real people, what fraction did I find?*

**Worked example:** scene has 40 people; model outputs 38 boxes; 33 are correct.
P = 33/38 = 0.868, R = 33/40 = 0.825.
- F1 = 2PR/(P+R) = 2(0.716)/1.693 = **0.846** (balanced blend)
- F2 = 5PR/(4P+R) = 3.580/4.297 = **0.833** — pulled toward R (0.825), the weaker one.

Now flip the situation (P = 0.825, R = 0.868): F1 is identical (0.846), but F2 = 0.859.
**F2 rewards finding more people even at the cost of extra false alarms** — β=2 in
F_β = (1+β²)PR/(β²P+R) puts β² = 4× weight on recall. Domain justification: a missed
person may die; a false alarm costs an operator seconds.

### 4.3 CBAM — the two attention equations, walked through
**Channel attention ("WHICH features matter?"):**
M_c(F) = σ( MLP(AvgPool(F)) + MLP(MaxPool(F)) )

Step by step on our P5 feature (512 channels × 20×20):
1. AvgPool and MaxPool squeeze each channel's whole 20×20 map to ONE number each →
   two 512-length vectors (each channel's average and peak response).
2. Both go through the SAME small MLP: 512 → 32 → 512 (reduction ratio r=16 keeps it
   tiny: ~33 k weights). The MLP learns which channel *combinations* signal "person".
3. Sum the two outputs, sigmoid σ squashes to (0,1) → one weight per channel.

*Toy example with 4 channels:* suppose channel responses (avg) = [0.2, 0.9, 0.1, 0.5]
where ch2 fires on compact vertical blobs (people) and ch3 on water glare. After the
MLP+σ, weights might be [0.4, **0.95**, **0.15**, 0.6] → person-channel amplified,
glare-channel suppressed — *before* any spatial reasoning happens.

**Spatial attention ("WHERE to look?"):**
M_s(F′) = σ( f⁷ˣ⁷([AvgPool(F′); MaxPool(F′)]) )
1. Now pool ACROSS channels at each pixel → two 20×20 maps (mean and max response per
   location).
2. Stack them (2×20×20) and run a single 7×7 convolution — it looks at each location's
   neighbourhood and scores "is something interesting HERE?"
3. σ → one weight per location (the blocky heat-map on your Slide 18 IS this map).

**Application:** F′ = M_c ⊗ F (each channel's whole map multiplied by its weight),
F″ = M_s ⊗ F′ (each location, across all channels, multiplied by its weight).
⊗ = element-wise multiply with broadcasting. Nothing is added — existing responses are
re-weighted: clutter → toward 0, faint people → preserved/relatively amplified.

### 4.4 CBAM vs ECA — and why we still tested ECA
- **ECA** (Efficient Channel Attention): channel attention ONLY, and even cheaper than
  CBAM's channel half — no MLP, just a 1-D convolution of kernel ~5 sliding across the
  channel dimension (a few dozen parameters). Each channel's weight comes from its
  neighbours' pooled responses.
- **What ECA lacks:** any notion of WHERE. It can say "texture channels matter less"
  but cannot say "this corner of the image is rubble, ignore it."
- **Why that matters here:** disaster clutter is spatial — rubble, water, smoke occupy
  *regions*. The spatial map is the half that localizes faint targets against them.
- **The measurement (report Tab. 4.2, earlier protocol):** baseline F2 0.844 /
  small-object recall 0.881 → +ECA 0.844 / 0.884 (nothing) → +CBAM **0.849 / 0.892**.
  ECA is a fine choice when the budget is extreme and clutter is channel-separable —
  on THIS task it bought no recall, so CBAM won. Say: *"we didn't assume, we measured."*

### 4.5 The three losses
**BCE (classification):** L_cls = −[y·log p̂ + (1−y)·log(1−p̂)], one class ("person").
Numbers: true person (y=1) predicted at p̂=0.9 → loss 0.105. Same person at p̂=0.1 →
loss 2.30 — **22× more**. The log punishes *confident wrongness*, which is what trains
calibrated confidence scores (your ECE 0.021 later).

**CIoU (box regression):** L_CIoU = 1 − IoU + ρ²(b,b^gt)/c² + αv — three penalties:
1. `1 − IoU` — overlap quality.
2. `ρ²/c²` — centre misplacement: ρ = distance between the two box centres, c = diagonal
   of the smallest box enclosing both. Dividing by c² makes it scale-invariant.
3. `αv` — aspect-ratio mismatch: v compares width/height ratios (via arctan); α = v/((1−IoU)+v)
   auto-scales it so shape only matters once overlap is decent.

Numbers: pred (0,0)–(10,10), GT (5,0)–(15,10) (same size, 5 px right):
IoU = 50/150 = 0.333 · centres (5,5) vs (10,5) → ρ² = 25 · enclosing box 15×10 →
c² = 15²+10² = 325 → ρ²/c² = 0.077 · same aspect → v = 0.
**L_CIoU = 1 − 0.333 + 0.077 + 0 = 0.744.** Even when IoU goes to 0 (no overlap — where
plain IoU loss has NO gradient), the centre term still tells the box which way to move.
That is why CIoU is the right loss for few-px boxes whose IoU flickers.

**DFL (Distribution Focal Loss):** instead of regressing each box edge as one number,
the head predicts a **probability distribution over discrete offset bins** (0,1,2,…,15
cells). If the true edge offset is 2.3 cells, DFL trains the model to put ~70 % mass on
bin 2 and ~30 % on bin 3; the final edge = expectation over bins = 2.3. Regression as
soft classification → richer gradients and **sub-pixel precision**, which at stride 4 is
sub-*half*-pixel in input space.

**Total:** L = 7.5·L_CIoU + 1.5·L_DFL + 0.5·L_cls — box terms outweigh class 15:1
wait — (7.5+1.5) : 0.5 = 18:1 on localization vs classification. With one class the
question is never *what*, only *where*. Kept **identical for all four configs** so the
ablation compares architectures, not objectives.

### 4.6 IOS (SAHI's merge rule)
IOS(a,b) = area(a∩b) / **min**(area(a), area(b)) ≥ 0.5.
Numbers: a person's full box is 8×20 = 160 px²; a tile border cuts it, so one tile
detects the top half: 8×10 = 80 px². Intersection = 80.
- IoU = 80/160 = 0.50 — borderline, could fail to merge.
- IOS = 80/min(80,160) = 80/80 = **1.0** — merged with certainty.
The min() denominator recognizes "small box entirely inside big box" = same object,
which is exactly the geometry tiling creates.

---

## 5. THE 640-vs-1280 CONFUSION — how SAHI and TTA actually take input

**Key fact 1: the network is *fully convolutional*.** Nothing in a conv layer cares
about image size — kernels slide over whatever they're given. "640" is only the size we
*trained* at. Feed 1280×1280 and everything still works; the grids simply double:
P2 becomes 320×320, P3 160×160, etc. (This is why `imgsz=1280` is a legal inference
flag with the same weights.)

**Key fact 2: your dataset images are NOT 640.** C2A images range from 150×150 up to
3400×3400. The 640×640 the network sees is produced by letterbox-downscaling. **That
downscale is where tiny people die**: a person who is 12 px in a 2000-px-wide original
becomes 12 × (640/2000) ≈ **4 px** in the network's view — below even P2's comfort zone.

Now the two enhancements, seen as two different answers to "how do I stop shrinking
people?":

### SAHI — don't shrink the image; cut it up
1. Take the ORIGINAL full-resolution image (say 2000×1500).
2. Slice it into 256×256-px tiles with 25–30 % overlap (~70 tiles for this size), plus
   ONE standard full-image pass (to catch anything large).
3. Each 256-px tile is **upscaled to 640** and pushed through the detector. The person
   who was 12 px in the original is 12 px in the tile → after upscale he is
   12 × (640/256) = **30 px** in the network's view. Detectable!
4. Each tile's boxes are mapped back to original-image coordinates.
5. Duplicates across overlapping tiles (and border-cut halves) are merged by greedy
   IOS matching (§4.6).
**Cost:** ~70 forward passes instead of 1 → 162 ms/image at 256-px tiles.
**Gain:** very-tiny recall 0.758 → 0.788 (+3.0 pt).

### TTA — shrink the image less, and look six times
1. Letterbox the whole image to **1280** instead of 640 → every person keeps **2×** the
   pixels (the 4-px person from before is now 8 px; a 8-px one becomes 16 px).
2. Evaluate SIX versions: scales {1.0, 0.83, 0.67} of 1280 × {original, horizontal
   flip}. Different scales catch different size bands; the flip catches asymmetric poses.
3. Union all predictions (flipped ones un-flipped first) → standard NMS removes
   duplicates.
**Cost:** 6 full-frame passes = 60 ms — *cheaper than any SAHI setting*, because 6 big
passes beat ~70 small ones.
**Gain:** very-tiny recall 0.758 → **0.850** (+9.2 pt), F2 → 0.854.
**Why it collapses past 2× (1920 px):** the model's learned feature statistics (object
sizes, receptive fields, normalization) match training scale ±2×; beyond that, people
become *larger than anything it trained on* and the DFL offset range / feature scales no
longer fit. Empirically it degrades — say "a model trained at 640 cannot extrapolate to
3× resolution."

**One-line answer if asked "but your input is 640?":** *"The 640 is a training-time
letterbox, not a property of the network — it's fully convolutional. SAHI avoids the
downscale by tiling the original image; TTA halves the downscale by running at 1280.
Both give the smallest people back the pixels the 640 letterbox was taking away."*

---

## 6. NMS — step by step

After decoding, the four grids emit thousands of raw candidates (we keep everything
above conf 0.001 for evaluation). Many are the SAME person seen by neighbouring cells
or scales. NMS (Non-Maximum Suppression):

1. Sort all boxes by confidence, highest first.
2. Take the top box → KEEP it.
3. Delete every remaining box whose IoU with it exceeds the NMS threshold (0.7 in our
   eval protocol).
4. Repeat with the next surviving box, until none remain.

**Mini example** — five boxes on two nearby people:
| box | conf | IoU with A |
|---|---|---|
| A | 0.92 | — |
| B | 0.88 | 0.85 → deleted (duplicate of A) |
| C | 0.81 | 0.10 → survives (different person!) |
| D | 0.60 | 0.75 → deleted |
| E | 0.55 | IoU with C 0.80 → deleted in C's round |
Result: A and C — one box per person.

**Why "clusters of nearby people are kept separate":** two *different* people's boxes
overlap each other only slightly (IoU ≈ 0.1–0.3, below 0.7) → both survive. Duplicates
of the *same* person overlap heavily (IoU > 0.7) → pruned. The threshold is exactly the
dial between "merge duplicates" and "don't merge neighbours" — and dense rubble scenes
are why it matters here.

---

## 7. Break / Integrate / Find — what the scaffold means

It's the story grammar of every methodology step, so the committee always knows where
the novelty is:
- **Break** = what we removed or rejected from the baseline (and that we *dared* to
  remove something — e.g. the two C2PSA blocks, or copy-paste augmentation after testing).
- **Integrate** = what we added/substituted, specified exactly (CBAM r=16 k=7 at layer
  10; P2 branch at neck layers 17–19; C3k2Mamba at 6 neck layers).
- **Find** = what the measurement said — including when the answer was "nothing"
  (Mamba). One change per step is what makes each Find attributable.

---

## 8. Slide-by-slide: the three "in action" slides

### CBAM in action (your slide 18)
- **(a) mechanism, illustrative:** the "before" tile shows what a feature map contains
  mid-network: several strong blobs from background clutter (rubble texture, water
  glare — pink) and one faint blob from a person (blue). CBAM computes channel weights
  (M_c: "glare-ish channels get 0.15") and a location map (M_s: "this region is water,
  weight 0.2"), multiplies them in → the "after" tile: clutter blobs faded, person blob
  now the dominant response (magenta). Nothing was detected yet — the *evidence* was
  re-balanced so the head later has an easier job.
- **(b) real spatial attention:** this is the ACTUAL M_s map read from your trained
  model on a C2A scene, overlaid warm=high. Two honest details to say before being
  asked: it's a 20×20 map upscaled to image size — **blocky is correct**, CBAM sits at
  stride 32; and warm regions align with the scattered people, not the pool/rubble —
  which is the mechanism doing on real data what the schematic promised.

### P2 in action (your slide 19)
- **(a) stride-8 grid:** the same crop with the P3 grid drawn on. Count cells per
  person: several neighbouring people share ONE 8-px cell — the P3 head must emit one
  prediction from features that mix multiple humans → merged/missed boxes.
- **(b) stride-4 grid:** grid twice as fine; each person now owns their cell(s) —
  separable predictions. This is the *spatial* explanation of the per-size table:
  very-tiny recall 0.743 → 0.757 comes precisely from these separations.
- **(c) P2 feature response:** the actual 160×160 P2 feature (mean over its 128
  channels, viridis colormap). The bright specks sit exactly where the tiny people are —
  proof the added branch genuinely fires on the targets it was built for (not a dead
  branch that merely adds parameters).

### Decoding + NMS slide equation strip
`strides {4,8,16,32} → grids 160²·80²·40²·20²` just restates the pyramid at 640 input;
each grid predicts independently, NMS merges (see §6); output = one box + confidence per
recovered person — what the operator reviews.

---

## 9. The three curves in Results & Findings (your slide "Fig. - 20")

### (a) Precision–Recall curve
Built by sweeping the confidence threshold from 1.0 down to ~0: at each threshold,
plot (recall, precision). Reading yours: the curve hugs precision ≈ 1.0 until recall
≈ 0.8 — *the model can find ~80 % of all people while almost never raising a false
alarm*. The cliff at the right end is the residual hard tail (very-tiny, occluded,
crowded) where forcing more recall costs precision quickly. **Area under this curve at
IoU 0.5 = AP50 = 0.853** — that's literally what the headline number is.

### (b) F1/F2 vs confidence threshold
Same sweep, but plotting the two F-scores. Each curve has a peak = its optimal
operating threshold. **F2 (orange) peaks at a LOWER threshold than F1** — a
recall-weighted metric prefers letting more detections through. The dashed line at
**0.16** is where MSA-YOLO deploys (best F2 = 0.844). Also note the F2 plateau is wide:
0.16–0.20 across all four configs → no fragile per-model tuning needed.

### (c) Reliability diagram (calibration)
Bin detections by their predicted confidence (x-axis); for each bin plot the fraction
that were actually correct (empirical precision, y-axis). Perfect calibration = the
diagonal: "when the model says 0.8, it's right 80 % of the time."
- Curve **above** the diagonal = under-confident (scores too modest);
- **below** = over-confident (scores inflated — dangerous for triage).
**ECE = 0.021** is the average |gap| to the diagonal, weighted by bin population —
about two percentage points. Operational meaning: an operator can *trust the numbers*
when triaging detections, and the 0.16 threshold means what it says.

---

## 10. Rapid-fire Q&A lines (memorize these)

- **"Why is aggregate AP flat across your models?"** → At 8-px scale a 1-px shift swings
  IoU by tens of points (see §4.1 table), so strict-IoU AP measures annotation noise;
  the task signal lives in AP50, per-size recall, and F2 — and we predicted this in the
  metrics section *before* showing results.
- **"Why not just use SAHI always?"** → 162 ms vs 60 ms for less gain than TTA on this
  dense benchmark; tiling shines on sparse large frames, C2A targets are everywhere.
- **"Why did Mamba fail?"** → It was genuinely active (scan diagnostics: fwd/rev cosine
  distance 0.836); it adds sequence memory along a scan path, but tiny-target detection
  is a *local resolution* problem, not a long-range dependency problem — the P2 result
  proves resolution was the bottleneck. 2.4 M params and 2.8× latency bought nothing.
- **"Is 0.743 → 0.757 significant?"** → Single seed, so quoted as observed; but it's
  +350 extra people found among 25,072 very-tiny instances, the direction matches the
  geometric prediction made in advance, and the mechanism is visible in the grid figure.
  Multi-seed testing is named future work.
- **"Why AdamW?"** → The framework default SGD diverged on the P2 architecture near
  epoch 50 (val cls loss unbounded); AdamW at lr 0.001 trained all four configs stably —
  pinned identically everywhere, so it can't favour any one row.

---
*Companion files: slide plan `docs/2026-07-09_defense_presentation_slide_plan.md`,
figure staging `Defense/Presentation/slide_figures/README.md`.*

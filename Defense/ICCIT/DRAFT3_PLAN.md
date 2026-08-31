# DRAFT 3 MASTER PLAN — ICCIT 2026 (compaction-proof; RE-READ THIS FIRST, every session)

**Deadline: Aug 31, 2026.** Submit via Microsoft CMT (`cmt3.research.microsoft.com/ICCITconf2026`).
**Working folder:** `Defense/ICCIT/draft3/`. Deliverable: `draft3/main.pdf`.
**Predecessors:** `draft1/` (wrong scope, abandoned), `draft2/` (correct content, 6 pp, builds clean,
KEEP as the fallback). Draft 3 changes FRAMING ONLY; every measured number is identical to draft 2.

---
## 1. THE DECISION THAT DEFINES DRAFT 3 (agreed with the user 2026-08-21)

The user's work has three parts that each look "medium" when presented as three separate
contributions: an architecture that is not novel (CBAM + P2 is a good combination of known parts),
a narrow mechanism, and a measurement. Draft 3 presents them instead as **ONE investigation**.

**The one repeatable sentence (the paper's whole point):**
> For the smallest targets, how supervision is assigned matters more than what is added to the
> network; and on real imagery none of these levers reaches the bottom of the size range.

**The four places we looked for sub-8-pixel recall** (this table IS the paper's spine):

| lever | where it acts | sub-8-px recall effect | cost |
|---|---|---|---|
| Attention (CBAM) | backbone | 0.7427 -> 0.7461 (+0.34 pp) | -0.96 M params, lowest latency |
| Resolution (stride-4 / P2 head) | neck + head | 0.7461 -> 0.7575 (**+1.14 pp**, not 1.4) | +0.50 M params, +1.1 ms |
| Sequence modeling (state-space neck) | neck | 0.7575 -> 0.7567 (-0.08 pp) | +2.44 M params, 2.8x latency |
| **Supervision (label assignment)** | training only | **0.7299 -> 0.7508 (+2.08 pp)** | **zero params, zero inference cost** |
| then: real disaster footage | -- | **0.000, both models, all 54 instances** | -- |

**CRITICAL HONESTY CONSTRAINT (never violate):** the first three rows are measured on the
**official** C2A split; the supervision row is measured on the **scene-disjoint** split. They must
NOT share a single axis or column silently. Each is a matched comparison *within its own protocol*.
State the protocol per group in every figure caption and table that mixes them.

**Why this framing keeps the architecture instead of dropping it:** MSA-YOLO stays the named
detector with its full result (AP50 0.853, 19.6 M params, second among published C2A detectors).
It simply stops carrying a novelty claim it cannot defend and starts carrying one it can: the
best network-side lever, measured under one protocol. The contribution becomes the controlled
comparison, which genuinely is new.

**Title:** MSA-YOLO: Attention, Scale, and Supervision for Tiny-Human Detection in Aerial
Search-and-Rescue Imagery

---
## 2. HARD RULES (all previously agreed; violating any one can sink the paper)

**Venue rules (web-verified 2026-08-17):**
- **6 pages MAX**, IEEE 2-column, **A4** -> `\documentclass[conference,a4paper]{IEEEtran}`.
  (The zips the user downloaded are the US-Letter variants; `a4paper` is how the A4 template is made.)
- **DOUBLE-BLIND.** No names, affiliation, KUET/Khulna/Bangladesh, roll number, email, or
  acknowledgment anywhere. Campus set = "a university campus". Violation = immediate rejection.
- Template rule: **no math or special symbols in title or abstract** -> write "AP50", never `$\mathrm{AP}_{50}$`.
- Table style per the official template: caption ABOVE, inside `\begin{center}`, `|c|` rules with
  `\hline`. **Table notes go BELOW the rules unboxed or into the caption/prose, NEVER as bordered
  `\multicolumn{n}{|l|}` rows** (the user caught this in draft 1; it looked nonstandard because it is).
- Figures: `\centerline{\includegraphics{...}}`, caption BELOW, self-contained and stating what to notice.
- Package set kept to the official template's: cite, amsmath/amssymb/amsfonts, graphicx, textcomp,
  xcolor. `\bm` is NOT available (draft 2 hit this).

**Writing rules:** `research/HUMANIZER.MD` + `research/WRITING_STYLE_JSTARS.md`.
No em/en dashes anywhere. No "Additionally / Moreover / Furthermore / Consequently". Banned
vocabulary: crucial, pivotal, leverage, notably, remarkable, comprehensive, seamless, delve,
testament, showcase, underscore, vital. Numbers attached to every claim, never adjectives.
Varied sentence rhythm. American English (labeled, generalization, normalized, behavior).

**Numbers rule:** every number must trace to `AUDIT_2026-08-21.md` (about 60 independent checks,
recomputed from primary artifacts with fresh code, zero mismatches). Two things are FORBIDDEN:
1. The seed-0 AP_small gain of the assignment pair (+2.00 pp). It did NOT replicate across seeds
   (+2.00 / -1.29 / +0.53). The paper explicitly does not claim it.
2. Any per-frame "N detections vs M labeled" phrasing. Detections are not hits. On the disaster
   set, precision is 0.5133, so about half of all detections are false alarms.

**Tooling rule:** LaTeX edits go through the Edit/Write tools, NOT bash heredocs or sed with
backslashes. Heredocs mangled backslashes three times in this project and once corrupted a chapter.

---
## 3. VERIFIED NUMBERS (inline so no future session must re-derive them)
Source of truth: `AUDIT_2026-08-21.md`. Everything below survived independent recomputation.

**Setup:** RTX 4070 Ti SUPER 16 GB, i7-14700K, 128 GB RAM; Python 3.11.9, PyTorch 2.12,
CUDA 12.6, Ultralytics 8.4.56. AdamW lr0 0.001, cosine, <=300 epochs, early stop on fitness
(patience 50) and F2 (patience 40), AMP, seed 0. Batch 16, reduced to 8 physical with gradient
accumulation for the stride-4 models. Optimizer pinned because the framework default diverged on
the four-scale head.

**C2A:** 10,215 images, ~360,000 person instances, 4 scene types (collapsed building, fire, flood,
traffic). About 47% of people are under 10 px. Official split 6,129 / 2,043 / 2,043. Scene-disjoint
re-split 6,135 / 2,040 / 2,040 (98% of official test scenes also appear in training; re-basing
costs 1.6 pp AP50 and 1.2 pp sub-8-px recall). 99.6% of test instances are below 32 px.
Official-split GT band counts: 25,072 / 20,520 / 26,614 / 317. Scene-split: 24,806 / 20,893 /
26,738 / 410.

**A. Ablation, OFFICIAL split (Table I):**
| config | params M | GFLOPs | AP50 | AP | AR100 | F1 | F2 | lat ms | ECE |
|---|---|---|---|---|---|---|---|---|---|
| YOLO11m | 20.031 | 67.65 | 0.8432 | 0.6151 | 0.6913 | 0.8497 | 0.8399 | 13.68 | 0.0229 |
| + CBAM | 19.076 | 66.86 | 0.8473 | 0.6161 | 0.6923 | 0.8499 | 0.8406 | 13.48 | 0.0206 |
| + CBAM + P2 | 19.571 | 86.66 | 0.8533 | 0.6153 | 0.7030 | 0.8479 | 0.8442 | 14.55 | 0.0214 |
| + SSM neck | 22.008 | 98.44 | 0.8521 | 0.6143 | 0.7044 | 0.8458 | 0.8438 | 41.06 | 0.0197 |

Per-size recall (VT / tiny / small / medium): baseline 0.7427 / 0.8688 / 0.8941 / 0.9085 ·
CBAM 0.7461 / 0.8730 / 0.8953 / 0.9117 · CBAM+P2 0.7575 / 0.8651 / 0.8857 / 0.8076 ·
SSM 0.7567 / 0.8700 / 0.8873 / 0.8107. SSM scan diagnostic: forward-vs-reverse cosine distance 0.836.

**B. Supervision lever, SCENE-DISJOINT split (Table III):**
gamma sweep, 50-epoch pilots: VT recall 0.693 / 0.712 / 0.729 and AP_small 0.560 / 0.553 / 0.505
for gamma = 0 / 0.5 / 1.0. Matched full-protocol pair (identical launcher, init, split, schedule):
| metric | gamma 0 | gamma 0.5 | delta | 95% CI |
|---|---|---|---|---|
| AP50 | 0.8449 | 0.8441 | -0.08 pp | [-0.32, 0.17] |
| F2 | 0.8257 | 0.8284 | +0.27 pp | [0.03, 0.50] |
| recall <8 px | 0.7299 | 0.7508 | **+2.08 pp** | [1.77, 2.38], p<0.001 |
| recall 8-16 px | 0.8343 | 0.8228 | -1.15 pp | [-1.48, -0.84] |
| recall 16-32 px | 0.8504 | 0.8410 | -0.95 pp | [-1.40, -0.50] |
Three-seed replication of the delta: +2.09 / +1.66 / +1.11 = **+1.62 +- 0.49 pp, positive in every
seed**. C = 12.8 (median person size in px). Bootstrap = paired image-level, 1,000 resamples.

**C. Published comparison (Table II), from Nihal et al. on the same official test split:**
Faster R-CNN 0.634 / 0.366 / ~41M · RetinaNet 0.693 / 0.383 / ~37M · RTMDet 0.708 / 0.442 / -- ·
Cascade R-CNN 0.735 / 0.486 / ~69M · DINO 0.789 / 0.471 / ~47M · YOLOv5 0.808 / 0.492 / -- ·
YOLOv9-e 0.893 / 0.688 / 57.3M. Ours: YOLO11m 0.843 / 0.615 / 20.0M; MSA-YOLO 0.853 / 0.615 / 19.6M.
(0.893 - 0.853 = 4 points of AP50 given up for 19.6/57.3 = about a third of the size.)

**D. Inference-time (recommended model, 2,043 test images):**
baseline_640 P 0.8566 R 0.8345 F1 0.8454 F2 0.8388 VT 0.7576 at 15.4 ms, official val mAP50 0.868 ·
sahi256 VT 0.7884 at 161.5 ms · sahi320 VT 0.7823 at 113.3 ms · sahi512 VT 0.7629 at 65.9 ms ·
**tta1280 P 0.7743 R 0.8765 F1 0.8222 F2 0.8540 VT 0.8496 at 59.9 ms, val mAP50 0.8783.**

**E. Real footage.** Sets (frames / people / median px): drone train 101 / 6,313; drone val 14 / 973;
drone test 60 / 3,584 / 44; campus train 87 / 4,510 / 37; campus test 25 / 663 / 37; disaster train
60 / 1,341 / 23; disaster same-events eval 25 / 1,070 / 17; disaster unseen-event eval 23 / 318 / 28.
Total real training frames = **248**. Fine-tune: C2A train split + real frames repeated 20x, 4K
downscaled to 1,280 px long side, lr 5e-4 cosine, <=80 epochs, selected epoch 66.

| set / model | AP50 | F2 | R@.25 | P@.25 | people | found | missed | false alarms |
|---|---|---|---|---|---|---|---|---|
| drone zero-shot | 0.3354 | 0.376 | 0.3365 | 0.5748 | 3,584 | 1,206 | 2,378 | 892 |
| drone drone-tuned | 0.7825 | 0.755 | 0.8281 | 0.5177 | 3,584 | 2,968 | 616 | 2,765 |
| drone all-real | 0.7950 | 0.7708 | 0.8365 | 0.5315 | 3,584 | 2,998 | 586 | 2,643 |
| disaster zero-shot | 0.1326 | 0.2016 | 0.1271 | 0.5211 | 1,070 | 136 | 934 | 125 |
| disaster drone-tuned | 0.1235 | 0.187 | 0.1224 | 0.5647 | 1,070 | 131 | 939 | 101 |
| disaster all-real | 0.4109 | 0.4715 | 0.4505 | 0.5133 | 1,070 | 482 | 588 | 457 |
| campus all-real | 0.6885 | 0.6657 | 0.7179 | 0.5053 | 663 | -- | -- | -- |
| unseen event, base | 0.3957 | 0.4348 | 0.3428 | 0.7315 | 318 | -- | -- | -- |
| unseen event, adapted | 0.4126 | 0.451 | 0.3742 | 0.6010 | 318 | -- | -- | -- |
| C2A after adaptation | 0.8370 | 0.8239 | 0.8211 | 0.8345 | 72,847 | -- | -- | -- |

Unseen-event delta +1.7 pp AP50, bootstrap CI [-3.2, 5.7], **p = 0.47 (not significant)**; precision
falls 13.1 pp (p = 0.018). C2A holds: 0.8441 -> 0.8370 (-0.71 pp), ECE improves 0.0142 -> 0.0111.
**Sub-8-px recall on real disaster frames = 0.000 for every model tested, all 54 instances.**
Both disaster training clips are pillarboxed broadcast footage (41% and 47% of frame is border and
station graphics), shared by the same-events eval frames and absent from the held-out clip.

**LATENCY WARNING:** two different latency numbers exist and must never be mixed. **14.55 ms** =
end-to-end (preprocessing + forward + NMS) from the thesis ablation, on the official split; use
this for the architecture claims. **8.7 ms / ~115 FPS** = a later forward-pass profile of the
adapted model. Draft 3 uses 14.6 ms for architecture statements.

---
## 4. SECTION PLAN (6 pages; page numbers are targets, adjust in the final loop)

- **Fig. 1 (page 1, single column) = THE ANSWER.** Sub-8-px recall gain per lever, grouped by
  protocol, ending in the real-footage zero. This figure is the paper.
- **I. Introduction** (~1 col): SAR need; the stride geometry (a 12-px person spans 1.5 cells at
  stride 8); the question "where does sub-8-px recall actually come from?"; four levers; Fig. 1;
  contributions as 4 bullets.
- **II. Related Work** (~0.6 col, tightened 30% from draft 2): detector families; small-object lines
  (scale, attention, slicing); the tiny-object ASSIGNMENT line (NWD / RFLA / SimD / TOOD) and the
  gap that it has never been put inside YOLO's task-aligned assigner for aerial persons;
  C2A / SARD / HERIDAL; domain randomization.
- **III. Detector and Levers** (~1.2 col): A. MSA-YOLO = CBAM substitution (ONE sentence + cite,
  no equations, they cost a third of a column in draft 2 for a well-known 2018 module) + stride-4
  head with the geometry argument; B. explored state-space neck (2 sentences); C. tiny-aware
  supervision with the Wasserstein equations and Eq. (blend) for gamma. **Fig. 2 = architecture
  (figure*, both columns).**
- **IV. Setup** (~0.6 col): hardware/software, the two protocols and why, the leakage audit,
  metrics (F2 primary, size bands, bootstrap).
- **V. Where the Gain Comes From** (~1.5 col): Table I ablation; per-size reading; Table III
  supervision pair + 3 seeds; Table II published comparison; TTA in 2 sentences.
- **VI. What Happens on Real Footage** (~1.2 col): Table IV sets + pipeline; Table V multi-domain;
  the three bounds as a SHORT BULLET LIST (cross-event not significant, absolute level modest with
  half the detections being false alarms, sub-8-px zero).
- **VII. Conclusion** (~0.3 col): the one sentence, plus what it implies (supervision deserves as
  much attention as architecture; the remaining gap is a data problem).
- References (~0.8 col, 27 entries, all resolving).

**Figure/table budget:** Fig 1 (answer), Fig 2 (architecture, figure*), Tables I-V. If space runs
short, cut in this order: (1) TTA sentences, (2) Table IV shrinks to prose, (3) Related Work again.
NEVER cut: Fig 1, Table I, Table III, the three bounds.

**Optional if the user supplies it:** the qualitative before/after panel (script v2 at
`10-07-2026-Novelty-Lap-3/scripts/29_qualitative_panel.py` is fixed and selftested; the last run
predates the row-crop fix so it needs ONE re-run). Only include it if something else is cut.

---
## 5. LOOP PROTOCOL (per section, in this order)
1. Write the section with the Edit/Write tools.
2. Check every number against Section 3 above.
3. Humanizer + banned-vocabulary grep; double-blind regex (names / institution / contact / roll).
4. Compile: `pdflatex; bibtex; pdflatex; pdflatex` in WSL.
5. Confirm 0 errors, 0 overfull, 0 undefined, page count <= 6.
6. Render the affected page to PNG and LOOK at it.
7. Append one line to the STATE LOG below, then continue.

Final loop: read all 6 pages as rendered; verify page budget, caption self-containment, table-note
style, citation integrity both directions, and that Fig 1 is on page 1.

---
## 6. STATE LOG (append newest last; a fresh session resumes from the first unfinished item)
- 2026-08-21: Plan written. Reframe agreed (one investigation, four levers).
- 2026-08-22 Loop 0: `draft3/` created (class, bst, references.bib, 3 figures copied from draft2).
  **Fig. 1 built**: `draft3/figures/make_fig_answer.py` -> `fig_answer.png`. Two panels, (a) the
  four lever deltas as within-protocol changes with the bootstrap whisker on the supervision bar
  and grey protocol tags on the right, (b) 0.751 on C2A against 0.000 on real disaster frames.
  Regenerate with `python make_fig_answer.py` in that folder.
- 2026-08-22 Loop 1: **`draft3/main.tex` written complete, all seven sections.** Restructured from
  draft2: new title; abstract rewritten around the question; Section III is now "Four Levers on One
  Detector"; Section V is "Where Sub-8-Pixel Recall Comes From" ordered network -> supervision ->
  published comparison; Section VI is "What Survives on Real Footage" with the three bounds as a
  bullet list. CBAM equations CUT to one sentence plus citation. Related Work reorganized as the
  four families. TTA/SAHI compressed into two sentences inside V-A. Intro geometry prose cut.
  Numbers changed from draft2 ONLY where the incremental framing required it: the stride-4 lever
  is now credited CBAM -> CBAM+P2 (AP50 0.847->0.853, AR 0.692->0.703, VT 0.746->0.757,
  F2 0.841->0.844, tiny 0.873->0.865, small 0.895->0.886, medium 0.912->0.808), all from the
  audit table. Gamma-blend sentence corrected to "about half the recall gain for an eighth of the
  precision cost" (was "most of the gain at a tenth", which the sweep does not support).
- 2026-08-22 Loop 2: layout. Title rebalanced to 3 even lines. Architecture `figure*` moved to be
  DECLARED inside Section I (before "This paper treats those options as one question") so it lands
  at the top of page 2; declared any later, the float defers to page 3 and page 2 becomes a text
  wall. **New Fig. 3** (stride geometry) built at `make_fig_stride.py`, replacing the hand-drawn
  `stride_problem.png` whose embedded labels were illegible at column width; orange marks the added
  P2 level, matching Fig. 1's orange. Table I caption shortened. `scale-robust` -> `scale-tolerant`
  (humanizer). **BUILD STATE: 6 pages, A4, 0 errors, 0 overfull, 0 underfull hbox, 0 undefined,
  27 refs, 1.1 MB. All 6 pages render-inspected. Double-blind grep clean; the only `---` are
  table "not reported" cells and the only `--` are numeric ranges, both correct IEEE usage.**
  Every one of the 13 labels is referenced.

## 6b. ORIGINALITY VERIFICATION (2026-08-27, live web + primary sources)
Run because ICCIT is peer-reviewed and the novelty claims had to be checked, not assumed.

**FINDING THAT CHANGED THE PAPER. Close prior art exists for the assignment mechanism.**
YOLOv7-UAV (Zeng, Zhang, He, Zhang; *Electronics* 12(14):3141, 2023;
https://www.mdpi.com/2079-9292/12/14/3141) lists as its own contribution (iv) "utilization of the
weighted normalized Gaussian Wasserstein distance (nwd) and intersection over union (IoU) as
indicators for **positive and negative sample assignments**", evaluated on VisDrone2019 and
**TinyPerson**, at a reported nwd:IoU ratio of **0.5:0.5** for TinyPerson. That is a weighted
NWD/IoU blend, in assignment, on aerial imagery, on tiny people, at our own ratio.
- **What still survives:** YOLOv7 is anchor-based and matches with SimOTA-style lead-head guided
  assignment. We place the blend inside TOOD's **task-aligned assigner**, where the overlap term
  feeds the alignment metric t = s^p u^q and so reshapes the joint classification/localization
  ranking, not just a candidate filter. Different assigner, different detector generation.
- **What does NOT survive:** any claim that nobody has blended NWD with IoU in assignment for
  aerial tiny persons. The old Related Work sentence said exactly that. It is now FIXED: the paper
  cites YOLOv7-UAV explicitly and narrows the unclaimed ground to the comparison in Section V.
- **Why this did not wreck the paper:** the draft-3 reframe had already moved the headline off
  "we invented an assignment method" and onto "supervision beats architecture, measured". The
  mechanism being known prior art is fully compatible with that framing, and arguably supports it.
  Had draft 2 been submitted, this finding would have hit the central claim instead.

**VERIFIED AS STILL TRUE:**
- No published report of C2A scene leakage. Searched the dataset by name against leakage /
  train-test-split / shared-background terms; nothing. This is our most original finding, so it was
  PROMOTED from a Setup paragraph to contribution bullet 4 plus a sentence in the abstract.
- C2A bar unchanged: YOLOv9-e 0.8927 AP50 / 0.6883 AP is still the standing published result
  (Nihal et al., arXiv 2408.04922). No newer C2A benchmark surfaced. "Second among published
  results" holds.
- Prior lap-3 lane-2 verification (2026-07-10 log line 133) called the TAL slot unclaimed. That was
  right about TAL specifically and wrong to imply the broader idea was unclaimed; YOLOv7-UAV had
  been in print since 2023. Lesson: search the MECHANISM, not only the exact module name.

**CAPTION LENGTH, settled with evidence.** Checked the C2A paper's own captions
(arxiv.org/html/2408.04922v2): its Fig. 1 caption runs about 110 words and Fig. 3 about 75, while
every table caption is a single short title ("Performance Evaluation of state-of-the-art Models on
the C2A Dataset"). Our figure captions (55 to 95 words) are in line with the subfield and were
KEPT. Our table captions carried extra sentences and, because IEEEtran sets table captions in small
caps, they looked oversized. All four were cut to short titles; the abbreviation glossary and the
0.25 threshold moved into the running text. Do not re-lengthen them.

## 6c. SUBMITTED TO ICCIT 2026 (2026-08-31)
Uploaded to Microsoft CMT (`ICCITconf2026`, Track 2: Computer Vision) as
`draft3/ICCIT2026_MSA-YOLO.pdf`, 6 pages, 1,164,225 bytes. Deadline was 2026-08-31, extended from
2026-07-31. Notification 2026-10-15, camera-ready 2026-11-15, conference 2026-12-18 to 20,
Cox's Bazar. NOTE: `*.pdf` is gitignored, so the exact submitted binary is NOT in the repo; it is
reproducible from this commit.

**Final title:** MSA-YOLO: Multi-Scale Attention and Tiny-Aware Supervision for Human Detection in
UAV Search-and-Rescue Imagery.

**Compliance verified against the CMT portal's nine stated rules**, not against assumption:
6 pages / IEEE 2-column / A4; no author names, affiliations or acknowledgments; **no page numbers
and no headers or footers** (checked empirically by rendering all six pages at 150 dpi: text ends
at 85.3% of page height, zero ink below it and in the header zone); single PDF. PDF metadata
carries only `Creator: TeX` and `Producer: pdfTeX`, and a raw binary string scan for the author's
name, supervisor, institution, city, roll number and Windows paths returns zero hits.

**Numbers audit (2026-08-27):** 130+ values re-verified folder by folder against primary run
artifacts, not against the md files. Table I 32/32, size bands 20/20, assignment 25/25, gamma sweep
6/6, SOTA 14/14 (checked against the C2A paper itself), multi-domain 15/15, found/missed 12/12
recomputed from raw `*_preds.json`. Five errors were found and fixed: drone train people
6,313 -> **6,315**; the unverifiable "47% under ten pixels" replaced with the exactly-measured
**34.5% below 8 px and 63% below 16 px** (the C2A paper claims 47% by width, but the released data
gives 45.30%, so their Table 3 does not match their own release); C=12.8 corrected from "the median
person size" to Wang et al.'s constant (the C2A median is exactly 12.00, matching the code comment
in `02_nwd_loss_patch.py:9`); the copy-paste result re-attributed to the four-augmentation stack it
actually used; and parameter counts in Table IV re-attributed, since Nihal et al. report none.

**Self-similarity:** 4.5% of the paper's 8-grams also appear in the thesis chapters. Low.

**KNOWN REMAINING DEFECT:** the architecture figure contains two em dashes in its box labels
("FPN — top-down fusion", "PAN — bottom-up fusion"). Text baked into the PNG is invisible to every
grep. Not an ICCIT violation; it breaches the project's own HUMANIZER rule. Fix in PowerPoint
before the camera-ready.

**CAMERA-READY TODO (by 2026-11-15 if accepted):** add author names and affiliations; add the IEEE
AI-disclosure statement to the acknowledgments (IEEE policy requires disclosing AI-generated
content and naming the system; wording to be agreed with the supervisor); fix the figure em dashes;
author list and order must match the CMT record exactly.

## 7. OPEN ITEMS FOR THE USER
1. ~~Optional qualitative panel.~~ **DECIDED 2026-08-27: NOT in the ICCIT paper.** The paper is at
   exactly 6 pages with four figures and five tables, every one load-bearing; the panel carries no
   number and would displace the supervision argument or a bound. `29_qualitative_panel.py` v2 is
   fixed and selftested, so it stays available for the journal version and the thesis report, where
   the page budget is not the constraint. Do not re-raise this for ICCIT.
2. ~~Confirm the 2.08 vs 2.09 seed-0 discrepancy.~~ **RESOLVED 2026-08-27, do not reopen.**
   `27-07-2026-Novelty-Lap-4/D2_SIGNIFICANCE.md` line 81 heads its columns
   `s0 delta | s1 delta | s2 delta | mean+-std | bootstrap (seed0, n=2040)`, so **seed 0 IS the
   matched pair** in Table II. The two figures are one run measured two ways: the per-run eval
   JSONs store recall at 4 dp, so 0.7508 - 0.7299 = 2.09, while the bootstrap and the independent
   audit both recompute from the raw detections and get **2.08**. 2.08 is the accurate one. The
   paper now reads "counting the pair above as the first, the three seeds give 2.08, 1.66 and 1.11
   points", which is both correct and preempts the reviewer question. Summary statistics are
   unchanged either way (mean 1.62, sd 0.49 under both 2.08 and 2.09).
3. `fig_real_samples.jpg` is copied into draft3/figures but NOT used by main.tex. Its labels still
   need `10m` -> `10 m` and a capitalized "(f) Unseen event" if it is ever included.
4. Release of annotations plus the protocol and the scene-disjoint split definition remains the
   highest-value citation lever. Annotations only, never the news frames (copyright).

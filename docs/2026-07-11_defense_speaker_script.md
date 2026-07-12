# 2026-07-11 — Speaker script for the defense deck (57 slides, keyed to the built PPTX)

Deck: `Defense/Presentation/2007074_Mofazzal_Disaster_human.pptx` (read 2026-07-11).
Time budget ≈ 20–22 min + demo videos. Section timings in brackets. Text in quotes =
speak it (adapt words, keep the *last sentence* of each slide verbatim — those are the
punchlines). [square brackets] = gesture/action, don't speak.

**Golden rules:** never read a slide aloud — say what the slide *means*. On table
slides, point at ONE number, not the table. Pause one beat after every "Find". If time
runs short, the skippable slides are marked ⏩.

---

## OPENING [1 min]

**S1 — Title**
"Good morning respected chairman and members of the committee. I am Md Mofazzal Hosen,
roll 2007074, and today I present my thesis: MSA-YOLO — a multi-scale attention
enhancement of YOLO for detecting tiny humans in aerial search-and-rescue imagery,
supervised by Professor Dr. Sk. Md. Masudul Ahsan. In one sentence: this is a compact
detector for people only a few pixels wide in disaster drone footage — built by
measuring one architectural change at a time."

**S2 — Outline**
"I will move from the problem and prior work to my objectives, then walk one image
step-by-step through the proposed architecture, then the results — including one honest
negative result — and close with a live demonstration."
[10 seconds. Do NOT read the list.]

## INTRODUCTION [2 min]

**S3 — Divider** [say nothing, or "Let me begin with why this problem matters."]

**S4 — Introduction**
"When an earthquake or flood strikes, the first hours decide who is found alive. Drones
survey a disaster site far faster than ground teams and risk no rescuer. But the imagery
they return is brutal to search: a person seen from altitude occupies fewer than ten
pixels, buried among rubble, water, and smoke — and one flight produces thousands of
frames. A human operator scanning that stream WILL miss people. So automating this
search is a practical need, and that is the problem this thesis addresses."

**S5 — Introduction (Cont'd)**
"These are the four scene types of the C2A disaster benchmark — collapsed buildings,
fire, flood, traffic incidents. Three facts define the difficulty: forty-seven percent
of the annotated people are under ten pixels — at or below a standard detector's
resolving limit; every scene carries twenty to forty people, many partially buried; and
mature detectors do not transfer here, because these targets are an order of magnitude
smaller than what those models were tuned for."

**S6 — Human Detection Example** ⏩
"This is what the task looks like when it works — every green box is a person the model
recovered. Keep this image in mind; by the end I'll show you how each box is produced."
[5–10 seconds only.]

**S7 — Applications**
"The same capability serves five uses: live victim triage during search and rescue,
damage assessment for relief planning, crowd monitoring, wide-area border patrol — and
because the final model is only nineteen point six million parameters, it fits embedded
hardware on the drone itself."

## RELATED WORKS [3 min]

**S8 — Divider** ["What has been done before, and what was missing."]

**S9 — Nihal et al. (C2A)**
"The foundation of this work is the C2A dataset by Nihal and colleagues: ten thousand
images, three hundred sixty thousand human instances, built by compositing segmented
human poses onto real disaster backgrounds. Their benchmark reports whole models against
whole models — YOLOv9-e wins at 0.688 mAP — but it never tells you WHERE a gain comes
from. That gap is exactly what my thesis fills."

**S10 — Liu et al. (UAVMOT)** ⏩
"Liu et al. addressed UAV multi-object tracking with motion-adaptive filtering — strong
work, but tracking-focused and heavy at twelve frames per second. It told me what NOT to
build: my problem needs lightweight per-frame detection first."

**S11 — Zhu et al. (TPH-YOLOv5)**
"TPH-YOLOv5 is the closest prior work: they added a P2 detection head and CBAM attention
to YOLOv5 for drone scenes. But they added everything at once, on generic urban imagery
— so nobody knows what each piece contributed. Remember this slide: my thesis takes
these same ingredients apart and measures them one at a time on disaster imagery."

**S12 — TABLE I (detector families)**
"Summarizing the field method-wise: two-stage detectors are accurate but too heavy for
airborne use; transformer detectors are competitive only recently and still heavier. We
work in the one-stage YOLO family, because speed and model size decide whether a
detector can fly."

**S13 — TABLE II (small-object techniques)**
"Three technique families exist for tiny targets: higher-resolution detection scales,
feature attention, and inference-time slicing. Notice the limitation column — every one
of them is usually reported ALONE, on generic aerial data. Nobody had compared them
under one protocol on disaster imagery."

**S14 — TABLE III (SAR data & state-space models)**
"Real rescue datasets like SARD exist but are far too small to train a twenty-million-
parameter model, which is why the semi-synthetic C2A is the right benchmark. And the
newest trend — Mamba state-space models — has been applied to detection only as whole-
backbone replacements, never as a controlled test on tiny humans. I test that claim."

**S15 — Research Gap (TABLE IV)**
[Read the RIGHT column only.] "So the gap, line by line: prior work reports techniques
individually — I compare them in one controlled ablation. Attention is usually asserted
— I measured CBAM against ECA on the task. State-space necks were never tested against
an identical baseline — I do, and report the outcome honestly. And it all happens on
disaster imagery with very small humans, not generic scenes."

## PROBLEM & OBJECTIVES [2 min]

**S16 — Problem Statement**
"Formally: given an aerial image, output one box per visible person, correct when the
intersection-over-union with ground truth is at least point five. [point at IoU box]
IoU is simply overlap area divided by combined area — B-hat is my model's box, B is the
annotator's. But the key property of this domain is asymmetry: a missed person may never
get a second chance, while a false alarm costs an operator seconds. [point at F₂ box]
That is why my operational metric is F-two — recall weighted four times over precision."

**S17 — Objectives**
"Four objectives. Build a compact detector through an ADDITIVE ablation — CBAM
attention, a stride-four P2 scale, and a state-space neck, introduced one at a time
under one frozen protocol. Identify which configuration best serves very-tiny recall
within a deployable budget. Honestly test the state-space claim — whichever way it
falls. And raise recall further at inference time without retraining. The novelty is
not any single module — it is the controlled attribution: after this thesis you know
what each component is worth."

## METHODOLOGY [7 min]

**S18 — Divider** ["Now the architecture — I will walk one real image through it, stage
by stage."]

**S19 — Proposed MSA-YOLO Architecture**
"This is the whole pipeline on a real C2A scene. The image enters the YOLO11m backbone,
which extracts features at four scales — see the feature pyramid on the right, from
fine 160-by-160 down to coarse 20-by-20. At the deepest level, my first modification:
CBAM attention replaces the baseline's C2PSA block — that's the red dashed box. The
neck fuses all scales top-down and bottom-up, and — second modification — I extend it
one level further to stride four: the red P2 branch. Four detection heads and NMS
produce the boxes you see on the right. Everything that is not red is unchanged
YOLO11m — deliberately, so every difference we measure is attributable."

**S20 — Step: Input & preprocessing**
"Stage one, the input. This collapsed-building scene is our running example — thirty
people, most under sixteen pixels. It is letterboxed to 640, never stretched — a
stretched eight-pixel person is a destroyed eight-pixel person. One honest note: I
tested aggressive small-object augmentation — copy-paste, mixup — and DISCARDED it as a
negative result; only default augmentations survive."

**S21 — Step: Backbone**
"Stage two, the backbone — untouched, and that's the point: it keeps the ablation
clean. What matters here: layers two, four, and six are kept as SKIP SOURCES — their
high-resolution outputs are saved and wired directly into the neck later. Layer two,
the 160-squared map, holds the finest detail the network ever computes — standard
YOLO11 never uses it. Mine will."

**S22 — Step: CBAM**
"Stage three, the first modification. I BREAK the baseline's two heavy C2PSA
self-attention blocks and INTEGRATE one CBAM module. [point at equations] CBAM asks two
questions in sequence: which feature channels matter — average and max pool each
channel, a small shared MLP scores them, sigmoid gives weights — and then WHERE to look
— pooling across channels, a seven-by-seven convolution, sigmoid again. The features are
multiplied by both answers: clutter toward zero, faint people preserved. And the FIND:
this substitution REMOVES about one million parameters — attention at negative cost,
with the lowest latency of all four configurations."

**S23 — CBAM in action**
"Does it actually do that? Left: the mechanism, illustrative. Right: the REAL spatial
attention map read from my trained model — warm means attended. It concentrates on the
scattered people, not the rubble or the pool. And before you ask — the map is blocky
because CBAM sits at stride 32, a 20-by-20 map upscaled. That is correct behaviour, not
an artifact."

**S24 — Step: P2 branch**
"Stage four, the main event. I INTEGRATE one more top-down step in the neck: upsample
the fused P3 feature, concatenate it with that saved layer-two skip, fuse — giving a
160-squared map read by a stride-FOUR detection head. I BREAK nothing — P3 to P5 are
untouched, the branch is purely additive, which is what lets me attribute its effect.
The FIND, as you'll see: this is the principal driver — very-tiny recall rises from
0.743 to 0.757."

**S25 — P2 in action**
"Why does stride four matter? Same crop, two grids. Left, the stride-eight grid: one
cell covers several neighbouring people — the head must summarize them into one
prediction. Middle, stride four: each person owns their own cells. Right: the actual P2
feature response — the bright specks sit exactly on the tiny people. The added branch
genuinely fires on the targets it was built for."

**S26 — State-space variant**
"Stage five — the explored variant. The literature promotes Mamba state-space models
for detection, but always as backbone replacements, never controlled. So I replace the
neck's C3k2 bottlenecks at six layers with bidirectional local-window selective scans —
backbone and head untouched. [point at recurrence] The scan carries a running memory
whose dynamics depend on the input. Preview of the find: plus 2.4 million parameters,
2.8 times the latency — and no accuracy change. I'll show that evidence in results."

**S27 — Loss design** ⏩ [can compress to 20 s]
"The training loss is standard YOLO11 and — crucially — identical across all four
configurations, so the ablation compares architectures, not objectives. Three terms:
binary cross-entropy on the single person class; complete-IoU for boxes — overlap plus
centre distance plus aspect, because pure IoU is unstable for few-pixel boxes; and
distribution focal loss for sub-pixel edge precision. The weighting is fifteen-to-one
in favour of localization: with one class, the question is never WHAT — it is WHERE."

**S28 — Decoding + NMS**
"Final stage: the head predicts at four strides — grids from 160-squared down to
20-squared — and non-maximum suppression removes duplicates while keeping nearby
DIFFERENT people as separate boxes. The output is what an operator reviews: one box,
one confidence, per recovered person."

**S29 — End-to-end trace** ⏩
"The whole journey on one slide — input, real attention, real P2 response, detections.
Every intermediate here is read from the trained model's activations. The pipeline does
what the diagrams claim."

**S30 — Additive ablation design (TABLE V)**
"And here is the experimental design that makes everything attributable: four
configurations, each differing from the previous by exactly ONE change — same data,
same split, same optimizer, same thresholds. This is why my negative result will be
interpretable: the state-space neck fails under conditions where the P2 head
demonstrably succeeds."

**S31 — SAHI**
"Beyond the architecture, two inference-time modes — no retraining. SAHI refuses to
downscale the original image: it slices it into overlapping tiles, runs the detector on
each tile — so tiny people effectively get BIGGER in the network's view — and merges by
intersection-over-smaller, which correctly re-joins people cut at tile borders. Gain:
three points of very-tiny recall — but at 162 milliseconds per image."

**S32 — TTA**
"Test-time augmentation instead runs the WHOLE image at 1280 pixels — legal because the
network is fully convolutional; 640 was only the training size — under three scales and
a horizontal flip, then merges by NMS. Nine point two points of very-tiny recall for
only sixty milliseconds — cheaper AND better than any slicing configuration. It
collapses beyond twice the training resolution, so 1280 is the sweet spot."

## RESULTS [5–6 min]

**S33 — Divider** ["So — what did the measurements say?"]

**S34 — Experimental setup (TABLE VI)**
"All four models trained on a single RTX 4070 Ti SUPER under this identical
configuration. One setting deserves its reasoning on record: the optimizer is pinned to
AdamW because the framework's default SGD diverged on the P2 architecture; AdamW
trained all four stably. Every run wrote checkpoints, telemetry, and an environment
manifest — every number I show can be traced to the run that produced it."

**S35 — Dataset**
"C2A: ten thousand images, roughly 360 thousand instances, frozen MD5-verified split.
The geometry that drives everything: 99.6 percent of test instances are below 32
pixels. And the stated limitation, stated early: the dataset is semi-synthetic —
transfer to fully real imagery is future work, and my demo will give a first taste of
it."

**S36 — Evaluation metrics**
"Metrics chosen for the task, not by habit. F-two leads because a miss outweighs a
false alarm. AP-fifty carries the localization signal — at eight-pixel scale a
one-pixel shift swings IoU wildly, so the STRICTER thresholds mostly measure annotation
noise; remember that when you see the AP column stay flat. Per-size recall is the
direct hypothesis test. And calibration matters because an operator must be able to
trust the confidence scores."

**S37 — MAIN RESULT (TABLE VII)** [slow down — this is the thesis]
"Here is the thesis in one table. Row by row: CBAM removes a million parameters and is
the fastest — attention at negative cost. Adding P2 lifts AP-fifty from 0.843 to 0.853
and AR from 0.691 to 0.703, for half a million parameters and one millisecond. And the
state-space row: 2.4 million MORE parameters, forty-one milliseconds — nearly triple
the latency — and no accuracy metric improves. Note the aggregate AP column is flat
across all rows, exactly as predicted two slides ago. On this evidence the recommended
model is CBAM plus P2 — MSA-YOLO — and the state-space neck is excluded."

**S38 — Waterfall**
"The same story as a picture: each bar is one component's contribution to AP-fifty.
Green, green — and the Mamba bar goes DOWN while its latency triples. Every gain
attributed to one change; the null displayed, not hidden."

**S39 — Per-size recall (TABLE VIII)**
"Now the hypothesis test. I predicted in the methodology that if a stride-four scale
helps, the gain MUST appear in the very-tiny band. [point] Very-tiny recall: 0.743 to
0.757 — plus one and a half points, exactly where the geometry said it must land. Two
honest notes: the medium-band drop rests on only 317 of seventy-two thousand instances
— noisy; and Mamba tracks CBAM-plus-P2 while its scan diagnostics prove the blocks were
genuinely active — the null is architectural, not a wiring fault."

**S40 — Curves**
"Operating behaviour of the recommended model. The precision-recall curve holds nearly
perfect precision out to eighty percent recall. The F-two curve peaks at confidence
0.16 — and the peak is a plateau across all configurations, so deployment needs no
fragile tuning. The reliability diagram: expected calibration error 0.021, meaning a
reported eighty percent confidence is right about eighty percent of the time —
operators can trust the scores for triage."

**S41 — Qualitative**
"The densest scene in the test set — 156 annotated people. The standard pass recovers
the large majority; SAHI and TTA add detections along the rubble line, where the
smallest and most occluded figures sit. The residual failures are of three kinds:
heavy occlusion, extreme scale, and crowding — all concentrated in the very-tiny band."

**S42 — SOTA comparison (TABLE IX)**
"Against every published detector on this same test split: MSA-YOLO at 0.853 stands
SECOND among all of them — one point of AP-fifty behind YOLOv9-e, a model nearly THREE
times its size. For a detector that must ride on airborne hardware, that is the right
trade."

**S43 — SAHI vs TTA (TABLE X)**
"And when latency permits, TTA at 1280 is the single best inference-time setting —
very-tiny recall 0.850, F-two 0.854, at sixty milliseconds — cheaper than any slicing
configuration. Both remain optional modes; neither touches the trained weights."

## DEMO [1–2 min]

**S44 — Live demo, cross-domain**
"Finally, generalization beyond the benchmark. I shot this footage myself with a DJI
Mini 3S over KUET Central Field at ten to fifty metres altitude, plus news footage from
The Guardian and AP. The model has NEVER seen real footage like this — it trained only
on the semi-synthetic C2A." [play Video-01; while playing:] "Notice it holds up on real
people at altitude — and where it hesitates, those are exactly the domain-gap cases my
future work targets."

**S45 — Live demo, test set**
"And on the C2A test set itself, the behaviour you saw in the tables." [play Video-02,
5–10 s is enough.]

## CLOSING [1.5 min]

**S46 — Timeline** ⏩
"Two thirteen-week terms: first term built the foundation and the baseline benchmark —
which I presented at pre-defense; the second term ran the ablation, the state-space
exploration, the inference-time studies, and this documentation."

**S47 — Limitations**
"Stated plainly: all results are on one semi-synthetic dataset; the medium size band is
too sparse to judge; the ablation ran a single seed per configuration, so small deltas
sit within run noise; and latency is desktop-GPU — no field trial yet."
[NOTE: say "single seed" even though it's not on the slide — see fix list.]

**S48 — Future work**
"Three directions, two already in progress: serving the model behind a ground-station
pipeline — which enables the TTA mode; an enhanced dataset with my own drone imagery —
directly attacking the domain gap you just saw in the demo; and validation on real
rescue collections like SARD and HERIDAL."

**S49 — Conclusion** [slow, four beats]
"To conclude. The gain on C2A comes almost entirely from the P2 detection scale — the
resolution argument was right. CBAM contributes efficiency — attention at negative
parameter cost. The state-space neck contributes nothing at 2.8 times the latency — an
honest, verified null. And the result is MSA-YOLO: AP-fifty 0.853, F-two 0.844,
nineteen point six million parameters, fourteen point six milliseconds — second among
all published detectors on this benchmark at a third of the leader's size. Thank you."

**S50–55 — References** [do not present; leave visible during Q&A if asked]

**S56 — Thank you**
"Thank you — I welcome your questions."

---

## FIX LIST noticed while reading the deck (before defense)

1. **S40 title says "Component Attribution"** — that's S38's kicker. Should be
   "Operating Point & Calibration" (it shows the PR/F/reliability curves).
2. **S47 Limitations**: contains "Edge-device optimization for UAV deployment" — that's
   a future-work item (pre-defense leftover), not a limitation. Replace it with the
   missing **"Single seed per configuration — small deltas sit within run noise"**
   (your strongest honesty point; the committee WILL ask about significance).
3. **S49 Conclusion**: line "Results validate feasibility on disaster datasets" is
   pre-defense wording — weak filler; replace with the headline-metrics line (AP₅₀
   0.853 · F₂ 0.844 · 19.6 M · 14.6 ms) if it's not elsewhere on the slide.
4. **Slide-number text glitches**: S31/S32 show "21/22" again (duplicating S26/S27's
   numbers), references slides all show "15". Fix page numbers or remove them from
   those slides.
5. **S57 is an empty slide** — delete it (an accidental blank after THANK YOU looks
   sloppy if you arrow past the end).
6. **Fig numbering**: S19 caption says "Fig. - 15" but sits before Fig. - 07/08 (S20/21)
   — renumber sequentially once slide order is final.
7. **Outline page numbers (S2)**: currently 01…34 — re-check they match the final
   printed numbers after fixes.

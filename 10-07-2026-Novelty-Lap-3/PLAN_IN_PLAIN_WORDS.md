# The plan in plain words (locked 2026-07-12, Option A)

**One-sentence answer to "what are we doing":** we are building ONE new detector design +
ONE credibility pipeline around it — not just annotating data and training models.

## The problem (what the paper is about)
Find very small (often <16 pixels) and partially hidden people in drone pictures of disasters.
Our main training data (C2A) is FAKE — people cut out and pasted onto disaster photos — so a
model trained on it tends to learn the wrong clues (paste edges) and then fails on real footage.

## The three layers of the paper

### Layer 1 — a new architecture (the "FCCG" composite; the NOVELTY headline)
We modify the inside of YOLO11m with two cooperating custom branches (not plug-and-play blocks):
- **Evidence branch:** a tiny person is a sharp little anomaly against a smooth background.
  This branch amplifies exactly that faint "pop-out" signal at the high-resolution layers,
  where standard YOLO smears it away.
- **Context gate:** amplifying pop-outs alone creates false alarms (grass, rubble specks). A
  second branch looks at the WIDE surrounding scene and outputs a "is a person plausible here?"
  gate that vetoes the evidence branch's mistakes.
- Plus two training-time upgrades: a fairer rule for which pixels get credit for tiny objects
  during training (assignment — the literature says this, not fancier losses, is the real lever),
  and occlusion-aware loss terms.
Why this is novel: each ingredient exists somewhere, but the gated evidence×context pairing,
inside a YOLO, for tiny occluded PEOPLE, driven by OUR measured failure evidence, is unclaimed —
verified against 2025/26 literature (closest rivals DERNet/SRTSOD/SAFE-Net are being read and
differentiated in writing).

### Layer 2 — the sim-to-real pipeline (the CREDIBILITY story; uses YOUR footage)
1. **Seam probe (cheap test):** prove/measure how much our current model cheats on paste edges.
2. **C2A-H:** clean the fake training images (better blending/harmonization) so the model can't
   cheat — published recipe gained +14 mAP50 on exactly this problem.
3. **Self-training:** let the model teach itself on your UNLABELED drone frames.
4. **Few-shot real data:** the 120–240 frames you annotate TODAY join training (small but proven
   to matter).
5. **Final exam:** the 60 frozen test frames = a real-world 3-altitude benchmark nobody else has.

### Layer 3 — honest benchmarking (what makes reviewers trust it)
Reproduce the printed C2A record (YOLOv9-e 0.8927/0.6883) under BOTH the official split and our
leakage-audited clean split, plus modern baselines (YOLO26m, YOLOv12m; D-FINE-M optional on the
A6000). Every number traceable; effect sizes reported honestly.

## So: is it "annotate + train blah blah"?
No. Annotation is ~5% of the story (one afternoon, Layer 2 ingredient). The paper's claim is:
**a measured-failure-driven composite architecture (Layer 1) + a synthetic-to-real protocol with
a real benchmark (Layer 2), proven under honest comparisons (Layer 3).** Any one layer alone is
publishable at a low tier; the three welded together is what targets IEEE JSTARS / WACV 2027
(stretch: Pattern Recognition with the formal-problem framing, TGRS if deltas are strong).

## What happens next (order of operations)
1. TODAY (you): annotate per `ANNOTATION_TODO_2026-07-12.md`. (Me): rival-differentiation notes
   (agent running), then S0 module code + selftests.
2. Seam probe + 50-epoch pilot pairs on PC-4/PC-2 — cheap go/no-go gates; nothing expensive runs
   until a pilot proves its component.
3. Winning combo → full protocol runs on PC-1; anchors (YOLOv9-e repro, YOLO26m, YOLOv12m).
4. Sim-to-real ladder + joint final model → 3-altitude benchmark scores → paper tables.
Decisions locked: Option A · baselines YOLOv9-e+YOLO26m+YOLOv12m mandatory, D-FINE-M optional ·
annotation approved (today's batch) · pose-aux = optional S2 row (audit PASSED 2026-07-12:
10,215 pose files, boxes identical to standard labels, poses balanced — free supervision signal).

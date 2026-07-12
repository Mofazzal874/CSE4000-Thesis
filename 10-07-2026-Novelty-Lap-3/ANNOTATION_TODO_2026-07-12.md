# Annotation TODO — 2026-07-12 (user labels TODAY; agreed under Option A)

## Priority 0 — finish the 60 frozen TEST frames (if not already done)
`Drone Shoot\extracted_v1\test_frames\{10m,30m,50m}` (20 each), already in Roboflow.
These gate the paper's real-world benchmark (S5). Export: COCO JSON. These frames are NEVER
trained on — keep them in their own split/tag.

## Priority 1 — NEW today: the self-training/few-shot pool
Source images (already extracted, test-guard verified via manifest.json):
`Drone Shoot\extracted_v1\selftrain_frames\10m` (80) · `...\30m` (80) · `...\50m` (80)

**Target today: EVERY OTHER frame per folder (sorted by filename) = 40 per altitude = 120 frames.
Stretch goal if energy remains: all 240.** Even 30/altitude helps (few-shot DA evidence:
single-digit→150 labeled target images move the needle; our evidence-backed budget is 50–150+).

**Order: 50 m first → 30 m → 10 m.** (50 m people are ~13–16 px — hardest to label but worth the
most: it's the exact C2A tiny band and the pseudo-label recall danger zone.)

## Labeling rules (same contract as the test frames)
1. Single class: `person`.
2. Box EVERY visible human, including partially visible/occluded — box the VISIBLE extent, tight.
3. Zoom to ≥100% when labeling 50 m frames; sweep the frame systematically (grid pattern) so
   tiny bodies aren't missed.
4. Dark clothing in shade = our known miss class — look twice there.
5. Frames with ZERO people: mark as null/empty and KEEP them (they are valuable negatives against
   our grass/texture false-positive problem — do not delete).
6. Keep this batch in a separate Roboflow tag/split named `selftrain_v1` — never mixed with the
   60 test frames. Export: COCO JSON (same as test).

## What these labels buy (so the effort has a purpose)
- 120–240 labeled real frames = the FEW-SHOT arm of the sim-to-real pillar (S2.5/S5): joint
  training C2A(+H) + SARD + these frames, per-dataset balanced sampling.
- The remaining unlabeled frames stay as self-training fuel (SF-UT ladder).
- The 60 test frames stay untouched as the final 3-altitude real-world benchmark.

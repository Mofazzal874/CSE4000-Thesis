# Disaster-frame annotation guide (SAM3 on Roboflow) — 2026-07-27

**Frames to annotate (on the laptop):**
`d:\Academics\thesis folder\Drone Shoot\extracted_v1\disaster_train_raw\` — 60 jpgs
(extracted from the 2 news clips, deduped, R-set-25 excluded). Do NOT upload `extract_manifest.json`.

## Steps
1. **Roboflow → Create New Project** → Object Detection → class `person` → name `disaster-train-v1`.
2. **Upload** the 60 jpgs → assign to self → annotate.
3. **SAM / Smart-Polygon (Auto)** tool: click each person → confirm → label `person`. Do all ~60.
4. **Policy:**
   - Box EVERY visible person (tiny / partial / occluded / dense crowds — each one).
   - Do NOT box dark voids / rubble holes / windows / debris (leaving them unboxed = the void-FP
     fix). Frames with no people = leave empty (valid background).
   - Single class `person` (ignore any Roboflow 2nd placeholder category — normalized later).
5. **Generate version:** Preprocessing **Resize = OFF**, Augmentation **none** (we do both in training).
6. **Export COCO** → download → extract to
   `d:\Academics\thesis folder\Drone Shoot\extracted_v1\annotations\disaster_train_v1\`
   (same layout as selftrain_v1: `train\_annotations.coco.json` + images).
7. **Then:** 22_coco_singleclass → 23_coco_to_yolo (--img-max-side 1280) → 24 joint retrain
   (C2A + drone + campus + disaster) → eval R-set + void-FP before/after.

## Caveat (record honestly in the paper)
disaster TRAIN (these 60) + R-set EVAL (25) come from the SAME 2 clips (different, deduped frames)
-> measures same-video adaptation, not cross-video generalization. Optional strengthener: add 1-2
more disaster clips for a cross-video split.

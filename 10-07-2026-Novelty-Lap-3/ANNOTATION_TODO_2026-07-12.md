# Annotation TODO — 2026-07-12 (beginner-proof Roboflow walkthrough)
> **SUPERSEDED 2026-07-19 by `ANNOTATION_GUIDE_v2_2026-07-19.md`** (supervisor asked for more
> frames + new sources: lying-down field shoot, campus-road tree occlusion, expanded R-set,
> hard-negative mining for the void-FP problem, and a model-assisted workflow). The Roboflow
> basics below (project setup, boxing rules, export recipe) remain correct and are referenced
> from v2 rather than repeated.

**Zero dependency note:** Claude's next coding step (S0) does NOT wait on this — annotate in
parallel, take your time, accuracy > speed.

## What you are labeling, in one line
120 real drone frames (40 per altitude) that will teach the model what REAL people look like
(few-shot arm of the sim-to-real pillar), plus finishing the 60 frozen TEST frames that are the
paper's final exam.

## STEP 0 — finish the 60 test frames first (existing Roboflow project)
The 60 frames in `Drone Shoot\extracted_v1\test_frames\` are already in your existing Roboflow
project. Finish boxing every person in them (rules in Step 3) and export (Step 4). Do NOT add any
new images to that project — it must stay test-only forever.

## STEP 1 — which frames? Already picked for you.
I copied your exact batch to:
`Drone Shoot\extracted_v1\annotate_batch_v1\`  → subfolders `10m\` `30m\` `50m\` (40 frames each;
every-other frame of the selftrain pool, selection recorded in `selection_manifest.json`).
**Just upload everything in this folder. No choosing needed.**
(Stretch goal later = the remaining 40/altitude still in `selftrain_frames\` — a second batch
another day, NOT today.)

## STEP 2 — Roboflow setup (one-time, ~3 minutes)
1. roboflow.com → sign in → **Create New Project**.
2. Project type: **Object Detection**. Project name: `drone-selftrain-v1`.
   Annotation group / class: `person` (single class — must match the test project spelling).
3. **Why a separate project:** so these training frames can NEVER get mixed into the frozen test
   set. One project = test only; this new project = selftrain only.
4. Click **Upload Data** → drag the three folders (`10m`, `30m`, `50m`) from
   `annotate_batch_v1\` into the upload box → wait for 120 images → **Save and Continue**.
   If it asks about splits, put EVERYTHING in **Train** (we control splits ourselves later;
   0% valid / 0% test in Roboflow).

## STEP 3 — annotating (the actual work; do 50m folder images first — hardest, most valuable)
Open **Annotate** → click the first image. Controls you need:
- Press **B** (or pick the bounding-box tool) → drag a box around a person → it auto-assigns the
  only class `person` → **Enter/confirm**.
- **Zoom: scroll wheel** (or +/−). For 50 m images work at ≥100% zoom and sweep the image in a
  grid pattern (left→right, top→bottom) — people are ~13–16 px, easy to miss.
- Rules per box:
  1. Box every visible human, including partially hidden ones — box only the VISIBLE part, tight.
  2. Dark clothing in shadow is our known blind spot — look twice at shaded areas.
  3. Do NOT box statues/posters/reflections (only real people).
- Image with NO people at all: click **Mark Null** (in the toolbar) — do NOT skip/delete it.
  Empty frames are valuable "nothing here" evidence against false alarms on grass.
- Navigation: arrow keys / Next. Progress bar shows N/120. It autosaves.
Rough time budget: 10 m ≈ 20–30 s/frame, 30 m ≈ 1 min, 50 m ≈ 2–3 min. Total ≈ 2.5–4 h.
If energy runs out: 50m + 30m complete beats everything half-done.

## STEP 4 — export (both projects, same recipe)
1. Project → **Versions** (left sidebar) → **Create New Version / Generate**.
2. Preprocessing: **remove/disable EVERYTHING** except Auto-Orient (Resize must be OFF — we need
   native 3840×2160 pixels).
3. Augmentations: **NONE** (augmentation happens in training, never in the dataset).
4. **Generate** → then **Export Dataset** → format: **COCO** (a.k.a. "COCO JSON") → download zip.
5. Unzip into: `Drone Shoot\extracted_v1\annotations\selftrain_v1\` (new batch) and
   `Drone Shoot\extracted_v1\annotations\test_v1\` (the 60 test frames). Tell Claude when done —
   filenames must match the originals (they will, if Resize stayed OFF).

## Sanity checklist before you stop
- [ ] 60 test frames fully boxed + exported (COCO) to `annotations\test_v1\`
- [ ] 120 batch frames boxed or marked Null + exported (COCO) to `annotations\selftrain_v1\`
- [ ] Nothing was resized/augmented in either export
- [ ] The two Roboflow projects were never mixed

# DATA MAP — every dataset, where it lives, its status (the anti-scatter file)

*Single source of truth for "where is X and is it done". Update the STATUS column whenever a set
advances. Last updated: 2026-07-22. Linked from START_HERE.md.*

## The rule that prevents scatter
- **Raw videos** stay in their source folder (never copied around).
- **Extracted frames** → `<source>\frames_v1\` (or the staged batch folder).
- **Final annotations** → ALWAYS a COCO export under `Drone Shoot\extracted_v1\annotations\<name>\`
  (one place for every export, drone + campus + R-set alike).
- **Eval/test sets are NEVER trained on and NEVER model-labeled** (human-only): the 60 test
  frames, `campus-eval`, and the R-set.
- Roboflow projects: one project per set; **Label Assist OFF** on every eval project.

## Pipeline status (each row: raw → frames → Roboflow project → COCO export → STATUS)

| Set | Raw source | Frames on disk | Roboflow project | COCO export (canonical) | Role | STATUS |
|---|---|---|---|---|---|---|
| **Drone train** | `Drone Shoot\ALL Drone Shots\{10,20,30,40,50}m.MP4` | `extracted_v1\selftrain_frames\` (240) → staged `annotate_batch_v1\` (120) | `drone-selftrain-v1` (assist ON) | `extracted_v1\annotations\selftrain_v1\` (120 imgs, 7538 boxes, 2048px) | TRAIN (sim-to-real few-shot arm) | ✅ DONE (120) |
| **Drone test** | same videos, frozen frames | `extracted_v1\test_frames\{10,30,50}m\` (60) | `drone-sar-test` / job `test_frames_60_frozen` (assist OFF) | `extracted_v1\annotations\test_v1\` (60 imgs, 3584 boxes, 4K) | TEST — frozen benchmark | ✅ DONE (verify miss-hunt noted) |
| **Campus (KUET) train** | `Drone Shoot\ALL Drone Shots\KUET_ROAD_{30,50}m.MP4` | `campus_v1\train_frames\` (88→87) | `campus-train-v1` (assist ON, RF-DETR) | `extracted_v1\annotations\campus_train_v1\train\` (87 imgs, 4510 boxes, 4K, 0 empty) | TRAIN (real occlusion, need N6) | ✅ DONE 2026-07-24 (41×50m + 46×30m) |
| **Campus (KUET) eval** | same KUET videos | `campus_v1\eval_frames\` (25) | `campus-eval-v1` (assist OFF, manual) | `extracted_v1\annotations\campus_eval_v1\test\` (25 imgs, 663 boxes, 4K, 0 empty) | EVAL (occlusion) | ✅ DONE 2026-07-24 (12×50m + 13×30m) |
| **RealDisaster R-set** | `Drone Shoot\Footage From News(Real)\` (Chennai_flood_final, venezuala_final; Turkey dropped) | `RealDisaster\frames_v1\` (36 → curated to 25) | `realdisaster-rset-v1` (assist OFF, manual) | `extracted_v1\annotations\rset_v1\test\` (25 imgs, 1070 boxes, 1080p, 0 empty) | EVAL — real-event benchmark | ✅ DONE 2026-07-23 (18 venezuela + 7 chennai; **45% boxes <16px = genuinely tiny → real-world tiny-object test**) |
| ~~Lying-down poses~~ | — | — | — | — | (was: pose diversity) | ❌ DROPPED 2026-07-23 (user decision; low cost — C2A already has 5 synthetic pose classes, pose-aux #37 unaffected, SARD/R-set cover real non-standing poses) |

## Reusable Roboflow assistant
- **RF-DETR-Small labeler**, trained in project `drone-selftrain-v1` on the 120 corrected drone
  frames (tile 3×3, 512px). Use it as Label Assist for the TRAIN tiers (campus-train, poses).
  Retrain after each big approved batch so it compounds. NEVER use it on an eval project.

## Extraction / split tools (lap-3 scripts)
- `12_rset_extract.py` — generic video→frames (news + campus). `--only <substr>`, `--every-sec`.
- `13_campus_split.py` — temporal train/eval split for single-clip-per-altitude footage.
- `14_gt_overlap_stats.py` — GT crowding / box-overlap stats on any COCO (feeds eval-NMS choice).

## Housekeeping notes
- `campus_v1\frames_v1\` (125) = reproducible pre-split copy; safe to delete for space (re-make
  with scripts 12+13). Kept for now.
- Original 4K drone frames live in `extracted_v1\selftrain_frames\`; the Roboflow train export is
  downscaled to 2048 — re-map to 4K via filename stem if hi-res tiling is ever needed.
- Provenance/quality log for all annotation batches: `extracted_v1\annotations\LABELING_LOG.md`.

# Novelty Lap 4 — disaster sim-to-real + paper finalization (started 2026-07-27)

Branch: `novelty-lap-4` (cut from `novelty-lap-3`). This lap **finishes the paper** — one more
data/experiment push (the disaster pillar) + the baselines the table needs, then WRITE.
Carries forward all lap-3 results (`../05-07-2026-Novelty-Lap/results/MANIFEST.md`).

## Why this lap exists
lap-3 delivered: leakage-audited protocol, D2 tiny-aware assignment (validated), and a STRONG
sim-to-real result on the **drone** benchmark (AP50 0.324→0.725). But the thesis is about
**disaster** imagery, and the real-disaster **R-set is still hard (~0.13)** — drone-tuning does
NOT transfer to it (domain-specific, [[project_d3_simtoreal]]). This lap closes that gap with
real disaster *training* data (SAM3-assisted, no tedious manual boxing) and then finalizes.

## Plan (efficient — then stop experimenting and write)
1. **Disaster pillar (B):** extract diverse frames from the 2 news videos (Chennai flood,
   Venezuela) — dedup, altitude/scene-stratified, **exclude the 25 R-set eval frames**.
   SAM3-assisted annotate ~75–100 on Roboflow → COCO → single-class → YOLO.
2. **Retrain** the joint model on **C2A + drone + campus + disaster** → eval on the held-out
   25 R-set + **void-FP before/after** (disaster frames carry the voids that fix the void-FP bias).
3. **Baselines:** PC-2 matched control (`s3_control_full`, α=0) = clean D2 full-protocol pair;
   **YOLO26m** anchor (mandatory SOTA row).
4. **S5 tables** + write the report + paper.

## Status
- [x] Full public-showcase audit and packaging (2026-09-19): inventoried 101,390 files;
      `../README.md`, `../public-showcase/`, and `../pendrive/README.md` revised.
      43-file release includes thesis, original presentation, two complete video demos,
      corrected gallery, nine CSVs, two bootstrap summaries, diagrams/charts, and checker.
      Portable copy and 56.9 MiB ZIP in `../pendrive/`; all release checks pass.
- [x] Gallery threshold mismatch identified: ranking table conf 0.25 vs overlays 0.16/0.19;
      public captions now use recomputed overlay counts. Original archive/Defense preserved.
- [ ] User release approval: initial public-showcase commit/publication and decision on making
      existing PUBLIC `Mofazzal874/CSE4000-Thesis` private. No remote changes made.
- [x] extract disaster training frames (dedup; exclude R-set 25)
- [x] annotate/normalize disaster data and build the combined training set
- [x] retrain joint (C2A + drone + campus + disaster) and evaluate R-set/void behavior
- [x] complete D2 matched controls, three seeds, and paired-bootstrap significance
- [x] run YOLO26m contextual anchor
- [x] close FCCG as a null architectural experiment
- [x] complete venue/impact audit: `2026-08-17_venue_research.md`
- [x] ICCIT title/name audit (2026-09-13): verified MSA-YOLO collisions; recommended
      assignment-focused title without model branding. See
      `../10-07-2026-Novelty-Lap-3/docs/2026-09-13_iccit_title_and_name_audit.md`.
- [ ] select final ICCIT title to match the submitted manuscript scope (user decision)
- [x] Title follow-up: read full ICCIT draft3 source; superseded assignment-only title with
      attention/scale/supervision framing and documented close YOLOv7-UAV prior art in the audit.
- [ ] write and internally audit the paper (recommended first target: IEEE JSTARS)

## Scripts (reuse lap-3, in `../10-07-2026-Novelty-Lap-3/scripts/`)
12 rset_extract · 22 coco_singleclass · 23 coco→yolo (downscale) · 24 joint_finetune · 18 eval ·
21 yolo26_anchor · 20 assign_patch. Only NEW need: a disaster-frame extractor variant + void-FP metric.

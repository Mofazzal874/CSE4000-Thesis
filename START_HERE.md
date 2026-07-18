# START HERE — map of this whole research project (plain words)

*Last updated: 2026-07-12 by Claude. Rule: this file's "RIGHT NOW" block gets refreshed at the
end of every working session, so it is always the truthful current state.*

## What this project is (one paragraph)
Undergrad thesis being upgraded into a journal/conference paper: detecting tiny (often <16 px)
and partially hidden PEOPLE in drone images of disasters. Base model: YOLO11m. Data: C2A (big
but synthetic — people pasted onto disaster photos), SARD (small but real), and our own 4K drone
videos at 10/30/50 m. The paper we are building = a new "context-gated evidence" architecture
(FCCG) + a synthetic-to-real pipeline proven on our own real footage. Full plain-words plan:
`10-07-2026-Novelty-Lap-3\PLAN_IN_PLAIN_WORDS.md`.

## RIGHT NOW (2026-07-19)
- **Annotation plan v2 is live:** `10-07-2026-Novelty-Lap-3\ANNOTATION_GUIDE_v2_2026-07-19.md` —
  supervisor-expanded scope (more frames + lying-down field shoot + tree-occluded campus road +
  R-set incl. Gaza-with-ethics-rules + void-FP hard negatives) and the SETTLED workflow: tiered
  hybrid (eval sets 100% manual; train tiers model-assisted via Roboflow Label Assist/Instant;
  50 m tier = own-model tiled pre-labels + miss-hunt pass; novel poses/occlusion = our-model ∪
  SAM 3 dual pre-label). Evidence: `docs\2026-07-19_autolabel_landscape.md` (web-verified).
- Drop folders ready: `Drone Shoot\field_poses_v1\raw_videos\` (lying-down shoot, capture spec in
  guide §1B) and `Drone Shoot\campus_v1\raw_videos\` (campus occlusion footage).
- Void-FP problem (2026-07-18 demo finding): fix path = tier-N hard negatives (guide §1E) +
  S2.5 fine-tune; FCCG context gate is the architectural half.
- PC-4 is DOWN → S0 smoke moved to PC-2 GPU1 (`scripts\SMOKE_CHECKLIST_PC2.md`) — still pending.
- Uncommitted changes awaiting user review (no-self-commit rule in force).

## PREVIOUS (2026-07-12, evening)
- **Decision locked: Option A** (composite architecture + sim-to-real pillar). Branch: `novelty-lap-3`.
- **User is annotating** per `10-07-2026-Novelty-Lap-3\ANNOTATION_TODO_2026-07-12.md`
  (60 test frames finish first, then the staged 120-frame batch in
  `Drone Shoot\extracted_v1\annotate_batch_v1\`). Exports land in
  `Drone Shoot\extracted_v1\annotations\{test_v1,selftrain_v1}\`.
- **S0 code is WRITTEN and locally green:** `10-07-2026-Novelty-Lap-3\scripts\`
  `10_fccg_modules.py` (FCCGFuse context-gated evidence @P2/P3 seams + FFLUp adaptive
  upsampler; 15/15 CPU selftests PASS; +0.52M params) + `yolo11m_fccg_p2.yaml` +
  `11_fccg_smoke.py` + `SMOKE_CHECKLIST_PC4.md`.
- **NEXT ACTION: run the S0 smoke on PC-4** (checklist above: selftest → --check-load →
  2-epoch smoke, --no-amp, batch 4). Pass ⇒ S1 paired pilots (ranking doc §5).
- Then P1 seam probe (PC-4, eval-only ~2h) can run in the same PC-4 session.
- Still running remotely: PC-1 G1 baseline retrain (do not disturb PC-1 until it finishes).

## If you are lost, read in this order
1. This file (you are here).
2. `10-07-2026-Novelty-Lap-3\PLAN_IN_PLAIN_WORDS.md` — what we're doing and why, no jargon.
3. `10-07-2026-Novelty-Lap-3\README.md` — current lap status checklist.
4. `10-07-2026-Novelty-Lap-3\docs\2026-07-10_ranking_and_publication_targets.md` — the full plan
   with gates (§5) and venue targets (§4).

## Folder map (what is ACTIVE vs history — safe to ignore history day-to-day)
| Folder | Status | What it is |
|---|---|---|
| `10-07-2026-Novelty-Lap-3\` | **ACTIVE — the current lap** | Verified idea catalog, ranking, protocols, annotation guide, plain-words plan |
| `Drone Shoot\` | **ACTIVE — our own data** | 3 videos (10/30/50m) + `extracted_v1\`: `test_frames` (60, FROZEN — never train), `selftrain_frames` (240 pool), `annotate_batch_v1` (120 staged for labeling), `annotations\` (exports land here) |
| `RealDisaster\` | ACTIVE — R-set (S5, EVAL-ONLY) | Real-event benchmark: drop disaster videos in `raw_videos\`, fill the PROVENANCE table in `README_PROVENANCE.md` (mandatory), extract with lap-3 script 12, hand-curate `frames_v1\`, annotate in a third Roboflow project. NEVER trained on. |
| `05-07-2026-Novelty-Lap\` | ACTIVE as infrastructure | Lap 1: the reusable scripts 00–07 (split builder, eval harness, frame extractor…) and **`results\` = THE RESULTS INBOX** (every remote-PC result + `MANIFEST.md` index) |
| `c2a\C2A_Dataset\` | ACTIVE as data | `new_dataset3` (official C2A, leaky split) + `new_dataset3_scenesplit_v1` (our clean re-split) + `All labels with Pose information` (bonus pose labels, audited OK) |
| `08-07-2026-Novelty-Lap-2\` | Reference | Lap 2: the FCCG architecture proposal + needs analysis N1–N8 + SOTA research |
| `Last Month\` | Reference | Thesis-era results: ablation runs, the joint C2A+SARD model (epoch125.pt), **`system_spec_thesis.md` §6 + `system_spec.md` §11 = the METRIC CONTRACT every eval must follow** (lap-3 mapping: `10-07-2026-Novelty-Lap-3\docs\2026-07-12_metric_contract_reference.md`) |
| `Defense\` | FROZEN | Defense report (XeLaTeX) + `demo\` (gitignored demo bundle). Don't touch. |
| `01-02-2026…`, `01-03-2026…`, `31-03-26…` | History | Early ablations/Mamba era. Ignore. |
| `docs\` | Reference | Older dated research/decision docs (pre-lap-3) |

## The machines (details: `10-07-2026-Novelty-Lap-3\PC_RUN_CONFIG.md`)
PC-1 4070TiS 16GB = protocol machine (comparable runs) · PC-2 A6000 48GB (GPU1 only) = heavy
pilots · PC-3 = spare · PC-4 4070 12GB (fp32, --no-amp) = smokes/pilots · laptop = code/labels/git
only (its old ultralytics can't load YOLO11 checkpoints).

## House rules that never change
1. Results from any PC land in `05-07-2026-Novelty-Lap\results\<pc>\<date>_<gate>_<name>\` + one
   line in `results\MANIFEST.md`.
2. The 60 test frames and anything derived from them NEVER enter training.
3. Every external method/paper is web-verified against primary sources BEFORE we act on it
   (`10-07-2026-Novelty-Lap-3\RESEARCH_PROTOCOL.md`).
4. Optimizer pinned AdamW lr0=0.001; PATIENCE=50; seeds 0–4 for ablation rows; metric contract
   from `Last Month\system_spec*.md` via lap-1 script 04.
5. Dated md files for every finding; branch per lap; nothing trains before its cheap gate passes.

## Session resume protocol
- **Claude, at session start:** CLAUDE.md loads automatically → read this file's RIGHT NOW block →
  read the active lap README → continue from its first unchecked box.
- **Claude, at session end:** update RIGHT NOW here + the lap README checklist + commit.
- **Human:** if Claude ever seems lost, paste it this file's path and say "resume".

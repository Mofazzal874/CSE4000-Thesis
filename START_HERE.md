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

## RIGHT NOW (2026-07-12)
- **Decision locked: Option A** (composite architecture + sim-to-real pillar). Branch: `novelty-lap-3`.
- **User is annotating** per `10-07-2026-Novelty-Lap-3\ANNOTATION_TODO_2026-07-12.md`
  (60 test frames finish first, then the staged 120-frame batch in
  `Drone Shoot\extracted_v1\annotate_batch_v1\`).
- **Claude's next step: S0** — write the FCCG modules + selftests + smoke run
  (design constraints in `10-07-2026-Novelty-Lap-3\docs\2026-07-12_P0_rival_differentiation.md`).
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
| `05-07-2026-Novelty-Lap\` | ACTIVE as infrastructure | Lap 1: the reusable scripts 00–07 (split builder, eval harness, frame extractor…) and **`results\` = THE RESULTS INBOX** (every remote-PC result + `MANIFEST.md` index) |
| `c2a\C2A_Dataset\` | ACTIVE as data | `new_dataset3` (official C2A, leaky split) + `new_dataset3_scenesplit_v1` (our clean re-split) + `All labels with Pose information` (bonus pose labels, audited OK) |
| `08-07-2026-Novelty-Lap-2\` | Reference | Lap 2: the FCCG architecture proposal + needs analysis N1–N8 + SOTA research |
| `Last Month\` | Reference | Thesis-era results: ablation runs, the joint C2A+SARD model (epoch125.pt), **`system_spec*.md` = the METRIC CONTRACT every eval must follow** |
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

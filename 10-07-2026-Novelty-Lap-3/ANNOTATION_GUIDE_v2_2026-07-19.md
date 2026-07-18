# ANNOTATION GUIDE v2 — all sources, tiered workflow (2026-07-19)

Supersedes `ANNOTATION_TODO_2026-07-12.md` (its Roboflow basics — project setup, boxing rules,
COCO-export recipe — still apply and are referenced, not repeated). Drivers of v2: supervisor
wants MORE frames; new sources (lying-down field shoot, tree-occluded campus road, expanded
real-disaster set incl. Gaza); the void-FP problem found during demo inferencing (2026-07-18)
needs hard negatives; and the workflow question (manual vs model-assisted) is now settled on
verified 2026 evidence: `docs\2026-07-19_autolabel_landscape.md` (all claims sourced there).

## 0. The workflow decision (answered)
**Question:** annotate every frame by hand, or annotate 100–200 seeds → train a labeler → let it
label the rest?
**Answer: a TIERED HYBRID — and the "train a small seed labeler" variant is rejected** because it
is dominated twice over: Roboflow's Instant Models already auto-train an assist model from your
first approved batch and improve with every next batch (the seed→assist loop is a built-in
platform feature now), and our full CBAM+P2/epoch125 detectors — trained on 360k boxes — beat any
200-frame model as a pre-labeler anyway. What survives from the idea is its spirit: **each
approved batch makes the next batch cheaper.**

| Tier | Data | Method | Why (verified) |
|---|---|---|---|
| **E — evaluation sets** (60 drone test · R-set · campus-eval clips) | **100% MANUAL, Label Assist OFF, never pre-labeled by our model lineage** | Circularity/anchoring bias inflates scores (arXiv 2106.12417; protocols require annotators not see model outputs). SAM click-to-tighten AFTER the human finds a person is allowed (human finds, tool only tightens). Two passes: annotate, then a separate-day self-review. |
| **T1 — easy train** (10 m/30 m drone, people 25–55 px) | Roboflow **Label Assist with OUR OWN model** → correct & approve | 40–70% time cut is the documented realistic range (treat as 2–3×, not "10×"). Instant auto-retrains per approved batch → assist keeps improving. |
| **T2 — tiny train** (50 m drone, <16 px) | Pre-label with OUR model at LOW conf, run **tiled** (SAHI-style) on the 4K frame; then a **dedicated second pass hunting MISSES** | Zero-shot foundation models documentedly degrade on small aerial objects; our C2A-trained model (63% of its training boxes <16 px) is the stronger pre-labeler here. Missed-tiny-person = silent label hole = the killer failure mode; hence the explicit miss-hunt pass at ≥100% zoom, grid sweep. |
| **T3 — novel-appearance train** (lying-down shoot · tree-occluded campus) | **Dual pre-label: our model ∪ SAM 3** (text prompt "person", open checkpoints, runs locally), human adjudicates the merged proposals | These poses/occlusions are OUTSIDE our model's training distribution (it will undershoot); SAM 3 (released 2025-11, promptable concept segmentation) finds person-instances from text. These corrected frames are the highest-value fine-tuning data → most careful review lives here. |
| **N — hard negatives** (void-FP fix fuel) | Curate + **Mark Null** only — no boxes, minutes of work | Demo inferencing showed FPs on dark voids (holes/pits/windows). C2A has ~0 empty images → the model never learned "nothing here". Every null-marked real frame with voids/shadows/windows is a targeted antidote. Target 100–200 across all sources. |

**Documentation duty (paper fuel):** for every batch record — which model pre-labeled, confidence,
tiled or not, who corrected, date. RealDroneVision (WACV 2026) is the citable precedent for
"semi-automatic train pipeline + fully-manual test set"; we describe ours the same way.

## 1. Source-by-source plan

### A. Drone selftrain pool (expand per supervisor)
- Finish the current 120 (`annotate_batch_v1`) — these become the FIRST approved batch that turns
  Roboflow Instant assist on for everything after.
- Then the remaining 120 of `selftrain_frames` (assisted, T1/T2 tiers by altitude).
- If supervisor wants still more: lap-1 script 05 can extract further alternate blocks from the
  3 videos (test-frame guard stays); do this only AFTER the 240 are done — more distinct SCENES
  (new sources B/C) beat more frames of the same scenes.

### B. NEW lying-down field shoot — CAPTURE SPEC (read before flying)
Purpose: real pose diversity matching C2A's 5 pose classes (bent/kneeling/lying/sitting/upright)
— currently our real data has almost no lying/kneeling people.
1. Same protocol as before: 4K, 10/30/50 m altitudes, steady hover segments.
2. Subjects: 3–6 people cycling through poses — LYING prone, lying supine, curled, kneeling,
   sitting, standing — scattered, not clustered only.
3. Vary: clothing incl. DARK colors (our known miss class), surfaces (grass/soil/concrete/road),
   partial occlusion (half-under a tree line, beside benches/vehicles), some in shadow.
4. **Fly separate clips for TEST vs TRAIN** — decide at capture time, never split one clip across
   both. Suggested: per altitude, 1 short test clip + 1–2 train clips.
5. Also record 1–2 people-free passes (negative fuel, tier N).
6. Drop videos: `Drone Shoot\field_poses_v1\raw_videos\` → extract with lap-3 script 12
   (`--every-sec 1 --max-per-video 80`) → curate → annotate (T3 tier; test clips = tier E).

### C. Campus-road tree-occlusion footage (you already have it)
Purpose: real occlusion evidence (need N6) — people partially hidden by tree canopy.
1. Drop videos: `Drone Shoot\campus_v1\raw_videos\`.
2. **Split at the CLIP level first**: pick 1–2 clips as EVAL-ONLY (tier E), rest = train (T3).
   Never both from one clip.
3. Extract with script 12 (`--every-sec 2 --max-per-video 60`), curate (delete unusable), then:
   train clips → Roboflow project `campus-train-v1` (dual pre-label T3);
   eval clips → project `campus-eval-v1` (MANUAL, assist off).
4. Boxing occluded people: box the VISIBLE extent, tight; if a person is fully hidden this frame,
   no box (we detect, not track). A trunk/canopy splitting a person visually is still ONE box
   over the visible parts' extent if the parts obviously belong together.
5. Targets: train 150–250 frames, eval 40–60 frames, plus null frames with heavy shadow/voids (N).

### D. RealDisaster R-set (Venezuela earthquake · Gaza · floods) — EVAL-ONLY
Full workflow + provenance table + **Gaza ethics rules** in `RealDisaster\README_PROVENANCE.md`
(updated 2026-07-19). Non-negotiables: 100% manual (tier E — never machine-proposed), aerial
viewpoint only, provenance row per video BEFORE extraction, 50–150 curated frames, Gaza only if
non-graphic and marked sensitive; earthquake+flood alone already carry the claim.

### E. Hard negatives for the void-FP problem (tier N — cheap, start immediately)
While annotating ANY source: when you hit a frame with dark voids (open manholes/pits, dark
windows, deep shadows, water holes) and NO people → **Mark Null and mentally tag it a keeper**.
Additionally: pull 30–60 frames from the demo-day inference footage where the model fired on
voids (we have the FP locations from counts/demo outputs) — those exact scenes, null-labeled,
are the most surgical training antidote available. These nulls flow into the S2.5 fine-tune mix.

## 2. Roboflow project layout (hard walls against contamination)
| Project | Content | Label Assist |
|---|---|---|
| existing test project | 60 drone test frames | **OFF forever** |
| `drone-selftrain-v1` (exists) | A: 240 selftrain frames | ON (own model / Instant) |
| `field-poses-v1` | B: lying-shoot train clips | ON (dual T3) |
| `campus-train-v1` | C: campus train clips | ON (dual T3) |
| `campus-eval-v1` | C: campus eval clips | **OFF** |
| `realdisaster-rset-v1` | D: R-set | **OFF** |
Export recipe for ALL: COCO JSON, Resize OFF, augmentations NONE (v1 guide Step 4). Exports land
under the source folder's `annotations\` subdir; tell Claude after each export.

## 3. Order of work (each step makes the next cheaper)
1. Finish 60 test frames (E, manual) — unblocks S5 scoring. ~today.
2. Finish drone 120 batch (mostly manual; approving it switches Instant assist ON). ~today/tomorrow.
3. Drone remaining 120 with assist (T1 for 10/30 m; T2 miss-hunt protocol for 50 m).
4. Campus extraction + clip split → campus-train (T3) + campus-eval (E). Null-sweep as you go.
5. Fly the lying-down shoot when convenient → same pipeline as campus.
6. R-set curation + manual annotation (independent of everything — any time).
7. Continuous: hard-negative nulls (E of section 1) — no dedicated session needed.

## 4. SAM 3 dual pre-label — how it actually runs (for step 4/5, T3 tiers)
Two options, pick per convenience: (a) X-AnyLabeling desktop (free, local, ships SAM 3 ONNX with
text-grounded labeling as of Apr 2026) — load frames, prompt "person", export, then merge/adjudicate
in Roboflow; or (b) Claude writes a short local script (SAM 3 checkpoints are open) that emits
COCO pre-labels for upload. Decide when we reach step 4 — if (b), that becomes lap-3 script 13.
Our-model pre-labels for the same frames come from the standard predict pass (low conf, tiled for
50 m). The human adjudicates the UNION (accept/fix/delete), never trusts either source blindly.

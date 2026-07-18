# RealDisaster R-set — real-event evaluation benchmark (created 2026-07-12, updated 2026-07-19)

**What this is:** a small, carefully curated, **EVALUATION-ONLY** set of real disaster imagery
(recent Venezuela earthquake, Bangladesh floods, Gaza — see ethics note, …) to close the last
credibility gap: real people inside real disaster scenes (C2A = fake people/real-ish scenes;
SARD = real people/no disaster; own drone = real sensor/no disaster). Used ONLY in the S5
evaluation grid — **never trained on, never used for self-training, never mixed into any Roboflow
training project, and NEVER labeled by any model (human-only annotation — eval-set purity rule).**

## ⚠ Conflict-zone imagery (Gaza) — special handling
Natural-disaster footage (earthquake, flood) is the R-set CORE and already carries the paper's
claim. Conflict-zone imagery is different: it frequently shows casualties, and its use in a
detection paper invites ethical review, reviewer discomfort, and politicization of a SAR-framed
contribution. Rules if Gaza clips are used at all:
1. AERIAL viewpoint only; scenes of structural damage with visible but NON-graphic people
   (walking, searching, standing on rubble). NO casualties, no bodies, no close-ups of suffering.
2. Frame it in the paper strictly as "urban structural-collapse search-and-rescue imagery".
3. Every Gaza row in the provenance table gets `sensitive: yes` in notes; if in doubt about a
   frame, DROP it. The earthquake+flood sets alone are sufficient for the real-event claim —
   Gaza is optional enrichment, never a dependency.

## Workflow
1. Drop the videos into `raw_videos\` (any names; mp4/mov/avi/mkv).
2. **Fill the provenance table below — one row per video, BEFORE extraction.** No row = the video
   does not get used. This is what makes the set publishable.
3. Extract frames (laptop, ~minutes):
   ```powershell
   cd "d:\Academics\thesis folder\10-07-2026-Novelty-Lap-3\scripts"
   python .\12_rset_extract.py --videos-dir "..\..\RealDisaster\raw_videos" --out "..\..\RealDisaster\frames_v1" --every-sec 2 --max-per-video 60
   ```
4. **Curate** `frames_v1\` by hand — DELETE frames that are: ground-level/non-aerial viewpoint,
   people-free AND context-free, duplicates, or ethically problematic (identifiable faces of
   victims in distress, bodies). Target: 50–150 keepers, aerial/high-oblique, people visible.
5. Annotate the keepers in a **new, third Roboflow project `realdisaster-rset-v1`**
   (Object Detection, single class `person`, everything in Train split inside Roboflow — we treat
   the export as test). Same boxing rules as the drone frames (visible-extent tight boxes, include
   partials, Mark Null on empty). Export: COCO JSON, **Resize OFF, augmentations NONE** →
   unzip to `RealDisaster\annotations\rset_v1\`.
6. Tell Claude — it enters the S5 eval grid as the "RealDisaster-mini" row.

## Publication rules (why the ceremony)
- Eval-only + per-image provenance = the accepted pattern for news-sourced benchmarks: the PAPER
  publishes the URL list + annotations, not the images themselves.
- Downloaded footage: record source + license/terms; prefer official agency/creator uploads.
- Own footage: mark `own` — no restrictions.
- Skip/crop imagery showing identifiable victims in degrading conditions. When in doubt, drop the
  frame — 80 clean images beat 150 questionable ones.

## PROVENANCE TABLE (fill me — one row per video in raw_videos\)
| file | event (what/where/when) | source (URL or "own drone"/"own phone") | license/terms | viewpoint (aerial/oblique/ground) | notes |
|---|---|---|---|---|---|
| (example) flood_dhaka_01.mp4 | Bangladesh floods, Sylhet, 2026-06 | https://... | YouTube-CC-BY / news-fair-use / own | aerial | drone journalism clip |
|  |  |  |  |  |  |

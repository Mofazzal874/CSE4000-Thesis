# Novelty Lap 3 — verified-idea triage & direction lock (started 2026-07-10)

Branch: `novelty-lap-3` (cut from `novelty-lap-2`). Folder keeps the repo's dated-prefix convention
(user asked for "novelty-lap-3"; this is it).

## Why this lap exists
The user brought a long Google/Gemini conversation containing 50+ 2025/26 ideas (YOLO26 / YOLOv12 /
YOLOv13 components, custom C2A modules like "CG-DSA"/"BP-FFP"/"DAMA", sim-to-real fixes). Some of it
is real, some is garbled, and some is suspected AI fabrication — and blindly pasting its code is one
reason integrations kept failing. Lap-3's job:
1. **Audit** every idea from that conversation against primary sources (3 web-search agents:
   YOLO-lineage / tiny-occluded SOTA / sim-to-real).
2. **Catalog** them: verdict, canonical paper, fetch-verified code link, fit to our measured needs
   (N1–N8, lap-2 doc A).
3. **Rank** by implementability × novelty defensibility × publishability; honest venue targets.
4. **Lock the direction**: lap-2's FCCG-YOLO core (independently re-validated by the convergent
   "Fourier boundary" suggestion in the Gemini convo) + a possible **sim-to-real pillar** exploiting
   our own unlabeled drone footage — the one genuinely new axis the Gemini convo surfaced.

## Contents
| File | What |
|---|---|
| `RESEARCH_PROTOCOL.md` | Standing verify-before-propose procedure (user rule, 2026-07-10) |
| `PC_RUN_CONFIG.md` | PC fleet, venvs, kill-safe rules, what runs where in lap 3 |
| `docs/2026-07-10_idea_catalog_and_audit.md` | Every idea from the Gemini convo: verdict + papers + code + fit |
| `docs/2026-07-10_ranking_and_publication_targets.md` | Scored ranking, tiers, top picks, venue reality check, first experiments |
| `docs/2026-07-10_web_verification_log.md` | Raw agent verification reports (traceability for every claim) |

## Status
- [x] Branch + folder + protocol scaffolding (2026-07-10)
- [x] Lanes 1–3 (YOLO lineage / tiny-occluded SOTA / sim-to-real) verified & logged (2026-07-10)
- [x] Idea catalog §A–G written — 58 items audited (2026-07-10)
- [x] Lanes 4–5 (top-conference + top-journal mining, user task 2) → catalog §H + lane files
      (first spawns quota-killed; respawns ran kill-safe with file checkpoints) (2026-07-10)
- [x] Ranking + venue targets written: top-3 = D3 sim-to-real / D2 TAL-assignment / D1 FCCG core;
      venue ladder JSTARS → PR(reframed) → TGRS stretch; WACV 2027 + AERO-HPR CVPRW 2027;
      CVPR-main/ICML ruled out with receipts (2026-07-10)
- [x] **DECISION LOCKED (2026-07-12): Option A** — composite FCCG + FreqDA. Baselines:
      YOLOv9-e repro + YOLO26m + YOLOv12m mandatory; D-FINE-M optional (A6000). Annotation
      approved: today's batch per `ANNOTATION_TODO_2026-07-12.md` (120 target / 240 stretch;
      60 test frames finish first). Plain-language plan: `PLAN_IN_PLAIN_WORDS.md`.
- [x] P0(b) pose-label audit PASSED (2026-07-12): 10,215 files, 0 malformed in sample, boxes
      identical to standard labels, poses balanced → pose-aux viable as optional S2 row.
- [x] Git hygiene: `Defense/demo/` (4k+ files, defense-day bundle) gitignored — kept on disk,
      out of novelty branches.
- [x] P0(a) rival differentiation COMPLETE (2026-07-12) → `docs/2026-07-12_P0_rival_differentiation.md`.
      Verdict: none of the 4 rivals does cross-scale context→evidence gating; all leave assignment
      untouched; none does occlusion or sim-to-real. Positioning locked: lead with
      **"context-gated evidence"** (not "frequency"); forbidden claims recorded per rival.
      OPEN ITEM: SAFE-Net PDF still "coming soon" on CVF — re-poll before paper writing.
- [ ] S0 (NEXT SESSION): FCCG module code + selftests + 2-ep smoke → gates per ranking doc §5.
      Read `docs/2026-07-12_P0_rival_differentiation.md` FIRST — module design must respect the
      recorded deltas (learnable evidence bank vs fixed bases; coarse→fine gate direction).

## Headline audit results (details in docs/)
- FABRICATED by the search AI: YOLOv12 "Ghost+Swin/transformer head" · YOLOv13 "YOLO-TCM" &
  "Dynamic Task Routing" · "EFEM-YOLO" · "July 2026 photogrammetry 88→95% study" · "YOLOE-26"
  (official). VE-DINO suspicion REFUTED — it is real (Smart Cities 2025, ground-level, no code).
- C2A bar CONFIRMED: YOLOv9-e 0.8927/0.6883, official (leaky) split, ICPR 2024 — UNCHALLENGED in
  print; only other number = LightSeek AP_small 0.478 (we're at 0.6156).
- C2A ships UNUSED official pose labels (5 classes) — pose-prior aux supervision feasible.
- Closest rival to FCCG idea: DERNet arXiv 2606.23825 (2026-06-22, no code) — read before S0.
- Sim-to-real consensus 2025/26 = harmonization (CFHA +14.1 mAP50) + SF-UT self-training ladder,
  NOT 2021-era GRL; label budget evidence → 50–150 drone frames.

## Pointers (don't duplicate, reference)
- Lap-2 architecture proposal: `..\08-07-2026-Novelty-Lap-2\ARCHITECTURE_PROPOSAL.md` (FCCG-YOLO, gates S0–S5)
- Measured needs N1–N8: `..\08-07-2026-Novelty-Lap-2\docs\2026-07-08_A_image_needs_analysis.md`
- Lap-2 verified SOTA/mechanisms: `..\08-07-2026-Novelty-Lap-2\docs\2026-07-08_B_sota_research.md`
- RESULTS INBOX stays single: `..\05-07-2026-Novelty-Lap\results\` + `MANIFEST.md` (every remote-PC result lands there)
- Metric contract: `..\Last Month\system_spec*.md` §6/§11 · Eval harness: lap-1 script 04

## Local ground truth to keep in mind (verified this session)
- Our C2A label space is **single class `person`** (`new_dataset3_scenesplit_v1\data.yaml`: nc=1) —
  any "5 pose classes" idea needs label re-derivation first.
- MuSGD already **diverged** on our P2 config (that's why final runs are pinned AdamW lr0=0.001) —
  do not re-adopt because a search AI hyped it.

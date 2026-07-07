# REPORT OVERHAUL PLAN — master state file (2026-07-07)

> **Purpose:** compaction-proof memory for the long-running Ch IV (+ technical chapters)
> overhaul. If context is lost, RE-READ THIS FILE FIRST, then continue from the first
> unchecked box. Update checkboxes + notes as work completes.

## Project anchors (do not re-derive)
- Report: `Defense/draft1_30_6_26/` (XeLaTeX + biber, compile in WSL:
  `cd /mnt/d/Academics/thesis\ folder/Defense/draft1_30_6_26 && xelatex main && biber main && xelatex main && xelatex main`)
- Verified numbers source of truth: `Defense/FINDINGS_AND_SOURCES.md`
- New SAHI/TTA run (CBAM+P2, 2026-07-07): `Last Month/24_01_26- Benchmarking YOLOs/CBAM_P2Head/runs_sahi_tta/20260707_062217_sahi_tta_cbam_p2/metrics/grand_summary.md`
- Canonical training runs (per-model metrics/summary.json):
  - baseline `.../Yolo11m/runs/20260615_230315_yolo11m_baseline_s0_nogit/`
  - cbam `.../CBAM/runs/20260601_232929_yolo11m_cbam_s0_nogit/`
  - cbam+p2 `.../CBAM_P2Head/runs/20260602_063759_yolo11m_cbam_p2head_s0_nogit/`
  - mamba `Last Month/02-06-26-Mamba_CBAM_P2Head/runs/20260609_205717_mamba_cbam_p2head_s0_nogit/`
- Spec of monitored metrics: find + read `system_spec_thesis.md` (user says everything
  monitored during training is catalogued there)
- Weekly presentations: search `weekly report*/` or similar folders for tables
- Sample thesis structure: `Defense/sample_efty.pdf` (Ch IV = 4.1 Introduction, 4.2
  Experimental Setup, 4.3 Evaluation Metrics, 4.4 Datasets, 4.5 Experimentation, 4.6
  Quantitative Results, 4.7 Qualitative Results, 4.9 Ablation Study, 4.10 Analysis of
  the Results, 4.11 Objectives Achieved, 4.12 Financial Analysis and Budget)
- HUMANIZER rules: `Defense/HUMANIZER.MD` — no prose dashes, no glue transitions
  (Additionally/Moreover), no AI vocab (crucial/robust/comprehensive/leverage/...),
  varied rhythm, concrete anchors, plain verbs. Apply to ALL new prose.

## EXCLUSIONS (user-mandated)
- NO deployable-model content, NO cross-dataset SARD (future work only).
- NO YOLOv8→11 family-selection details in body: user will add an appendix summary
  table of ALL tried models (v8→v11, cbam, p2, mamba, atrous); body only MENTIONS the
  appendix. (Detector Family Benchmark section in Ch IV shrinks to a mention.)
- `31-03-26(Mamba-ViT-CNN)` and onward "model trying" folders contain WRONG data — use
  for IDEAS/structure only, never numbers.
- NO "Discussion" section anywhere (supervisor rule 9).

## SUPERVISOR RULES (scope: methodology, implementation-results + technical chapters)
1. Objectives: only 3–4 items in Ch I (undergrad scale). DONE means Ch I list trimmed.
2. "Objective Achieved" must cite the exact sections/tables/figures where each
   objective was met (e.g. "as shown in Section 4.6, Table 4.5").
3. ALL figure/graph text ≥ 10pt when printed. Tiny-font images (e.g.
   cbam_p2_training_panel.png) must be replaced or regenerated.
4. Abstract: end with to-the-point numbers ("achieves AP50 0.853 ... 19.6M params").
5. References must fill pages (expand toward 40+; add reputable entries only).
6. Same-trend line charts: show ONE chart only (the worst-performing config) and
   explain the others in prose. (Training curves: one panel, not four.)
7. Any error map / heatmap needs a SCALE (colorbar) beside it. cbam_overlay.png and
   p2_featuremap.png lack colorbars → regenerate script-side or annotate.
8. Visual + writing consistency across the whole report.
9. No discussion section.
- Layout: combine related figures (input + GT + prediction) into unified grids, not
  scattered pages. Budget: ONE definitive workstation estimate, not a range.

## TOC FORMAT CHANGE (global)
- Remove bullet (•) and arrow (➤) markers from TOC. Show numeric labels: sections
  X.Y, subsections X.Y.Z. NO dot leaders (plain right-aligned page numbers stay).
- Requires: `secnumdepth=2` (number subsections in body too) + titletoc formats
  showing `\thecontentslabel` for section/subsection instead of bullet/ding.
- LoT/LoF header + PAGE header stay as-is.

## FIGURE 1.1 (intro scene)
- Reference given by user: AGIIndia, "Life-Saving Insights in Disaster Zones with
  Drones," Apr 30, 2024. [Online]. Available:
  https://agiindia.com/life-saving-insights-in-disaster-zones-with-drones/
  [Accessed: Dec 14, 2025]. → bib key `agiindia2024drones` (@online/@misc).
- Caption cites it; user pastes the aerial flood image as
  `figures/fig_intro_scene.png`.

## TASK LIST (tick as done)
### A. Structural quick wins
- [ ] A1. This plan file written
- [ ] A2. `references.bib`: add `agiindia2024drones`; cite in Fig 1.1 caption
- [ ] A3. Ch I objectives 5 → 3–4 (merge ablation+quantify; keep select+justify;
        honest SSM test; inference-time modes → fold into others)
- [ ] A4. thesisstyle.sty TOC: numbers instead of bullets/arrows, secnumdepth=2,
        subsection numbering X.Y.Z; verify no over-deep numbering elsewhere
- [ ] A5. Abstract: concrete closing numbers (AP50 0.853, VT recall 0.757, 19.6M,
        TTA VT 0.850 @60ms; 2.8× Mamba null)
- [ ] A6. Budget (Ch IV): single workstation estimate. Compute: RTX 4070 Ti SUPER
        workstation ≈ BDT figure + electricity ~10–15 kWh → one number, not range.
### B. Resource deep-dive (read before writing Ch IV)
- [ ] B1. Find + read `system_spec_thesis.md` (metric catalog; what was monitored)
- [ ] B2. Inventory canonical run outputs: metrics/*.csv, plots/, architecture/,
        summary.json for all 4 models (baseline/cbam/cbam+p2/mamba)
- [ ] B3. Weekly presentation tables: locate (weekly report folders / pptx / md)
- [ ] B4. Re-read sample_efty Ch IV text (lines 1580–2949 of extracted text) for
        section flow + how they explain metrics WHY
- [ ] B5. Check `01-02-2026- ablation study/` + `docs/` for ECA-vs-CBAM table data
### C. Chapter IV rewrite (sample-aligned skeleton)
- [ ] C1. 4.1 Introduction (roadmap)
- [ ] C2. 4.2 Experimental Setup (hardware/software table; training config table;
        WHY these settings; single budget-ready workstation spec)
- [ ] C3. 4.3 Evaluation Metrics (WHY each metric for tiny-target SAR: AP50 vs AP,
        per-size recall, F2 over F1, latency; equations already in Ch I → reference)
- [ ] C4. 4.4 Dataset (C2A description + samples grid + size-distribution TikZ;
        composited caveat; WHY this dataset)
- [ ] C5. 4.5 Experimentation (protocol: frozen split, seeds, early stop, OOM
        ladder, AMP; smoke tests; monitoring per system_spec)
- [ ] C6. 4.6 Quantitative Results (main ablation table + waterfall + per-size +
        curves; ONE training-dynamics chart (worst config) per rule 6)
- [ ] C7. 4.7 Qualitative Results and Error Analysis (unified grids: input|GT|pred;
        detgrid pair; failure modes; colorbar note on heatmaps)
- [ ] C8. 4.8 Comparison with State-of-the-Art (Nihal table; mention appendix for
        full model-selection history)
- [ ] C9. 4.9 Inference-Time Enhancement (SAHI/TTA new numbers; per-size bars fig)
- [ ] C10. 4.10 Objectives Achieved (cross-refs to exact sections/tables per rule 2)
- [ ] C11. 4.11 Financial Analysis and Budget (single estimate)
- [ ] C12. 4.12 Conclusion. NO discussion section.
### D. Figure hygiene
- [ ] D1. Replace cbam_p2_training_panel.png (6 tiny subplots) with ONE readable
        loss/metric chart (worst-performing config per rule 6) — regenerate via
        results.csv + matplotlib (fonts ≥10pt at print size) or TikZ/pgfplots
- [ ] D2. Heatmap colorbars: regenerate cbam_overlay/p2_featuremap with colorbar
        (script change; PC1 rerun) OR grayscale-note in caption meanwhile
- [ ] D3. Unified qualitative grid: input | GT | prediction (use qualitative/
        outputs from sahi_tta run: *_COMPARE.jpg exist!)
- [ ] D4. Check every included figure at print size for <10pt text
### E. Verification
- [ ] E1. Full compile chain (xelatex ×2 + biber + xelatex ×2), 0 errors/undef
- [ ] E2. Render + view every changed page
- [ ] E3. Humanizer scan of all new prose (dashes, glue words, AI vocab)
- [ ] E4. Update FINDINGS_AND_SOURCES.md §3 with new SAHI/TTA numbers
- [ ] E5. References count ≥ 40 (rule 5) — add only real, checkable entries

## KEY VERIFIED NUMBERS (for Ch IV prose; sources noted)
- Ablation (COCO test): AP50 .8432/.8473/.8533/.8521; AP .615/.616/.615/.614;
  AR100 .691/.692/.703/.704; F1 .850/.850/.848/.846; F2 .840/.841/.844/.844;
  params 20.03/19.08/19.57/22.01M; GFLOPs 67.7/66.9/86.7/98.4;
  latency 13.7/13.5/14.6/41.1 ms (baseline/cbam/cbam+p2/mamba)
- Per-size recall (test): baseline .743/.869/.894/.909; cbam .746/.873/.895/.912;
  cbam+p2 .757/.865/.886/.808; mamba .757/.870/.887/.811 (VT/tiny/small/medium)
- GT counts: VT 25,072 / tiny 20,520 / small 26,614 / medium 317 / large 0
  (99.6% < 32px)
- SAHI/TTA (new run 20260707, CBAM+P2): baseline_640 P.857 R.835 F1.845 F2.839
  VT.758 15ms | sahi256 VT.788 162ms | sahi320 F1.852 113ms | sahi512 VT.763 66ms |
  sahi640 VT.756 54ms | TTA1280 R.877 F2.854 VT.850 60ms, val mAP50 .8783,
  mAP50-95 .6765 (baseline val .868/.6317) | TTA640 .8407 | TTA832 .8732 |
  TTA1920 .8507 (degrades)
- ECA-vs-CBAM (older protocol, cite as preliminary): baseline F2 .8441/SOR .8807;
  +ECA .8435/.8835; +CBAM .8491/.8920
- Mamba diagnostics: fwd-vs-rev scan cosine distance 0.836, 6 blocks, d_state 4
- Training: AdamW lr0 1e-3 (MuSGD diverged), cos LR lrf .01, ≤300 ep, patience 50,
  F2-patience 40, batch 8 phys/nbs 16, imgsz 640, AMP, workers 4, cache ram,
  close_mosaic 10, 1 seed (s0) ablation scope
- Wallclock ≈ 7–30 h/config (mamba 29.5h) on 4070 Ti SUPER, ~180W avg
- C2A: 10,215 imgs, ~360k instances, 47% <10px, 20–40 inst/img, 4 scene types,
  train 6,129 / val 2,043 / test 2,043 (frozen, md5)
- SOTA (Nihal ICPR 2024): FasterRCNN .634/.366; RetinaNet .693/.383; RTMDet
  .708/.442; Cascade .735/.486; DINO .789/.471; YOLOv5 .808/.492; YOLOv9e
  .893/.688 @57.3M — ours CBAM+P2 .853/.615 @19.6M

## STATE LOG (append entries; newest last)
- 2026-07-07: Plan created. Ch I rewritten earlier today (objectives still 5 → trim).
  Ch III rewritten + verified. New PNG exports installed. SAHI/TTA numbers refreshed
  in Ch IV §SAHI already; rest of Ch IV still old structure.
- 2026-07-07 (later): ALL of A + B + C + most of D/E DONE. Details:
  - A2 bib `agiindia2024drones` + Fig 1.1 cite ✔; A3 objectives → 4 ✔; A4 TOC numeric
    X.Y/X.Y.Z no bullets/leaders, secnumdepth=2 ✔ (verified render); A5 abstract closes
    with AP50 .853 / F2 .844 / 19.6M / 14.6ms + TTA .850 ✔; A6 budget table BDT 285,000
    definitive + ~20 kWh electricity ✔.
  - Ch IV fully rewritten: 4.1 intro / 4.2 setup (+AdamW-divergence + patience WHY) /
    4.3 metrics with per-metric WHY subsections / 4.4 dataset (+why C2A) / 4.5
    experimentation (smoke, OOM ladder, resume, sampler, wallclock 7.3/7.0/10.2/29.5h,
    early-stop events: cbam ep239 F2-stop, cbam+p2 ep218 F2-stop, baseline ran to 300
    cap best@270, mamba fitness-stop ep154 best@149) / 4.6 quantitative (eca, main,
    waterfall, per-size + Mamba scan-diagnostic 0.836, NEW tab:thrcal thresholds+ECE,
    ONE training chart = mamba incl. epoch-28 collapse val-cls .61→4.5 documented) /
    4.7 qualitative (NEW fig:qualgrid 2×2 GT/640/SAHI/SAHI+TTA from
    collapsed_building_image0463_0, panels in figures/qual_*.png; detgrid; taxonomy) /
    4.8 SOTA + appendix mention / 4.9 SAHI-TTA / 4.10 objectives⇄sections cross-refs /
    4.11 budget / 4.12 conclusion. NO discussion section. Detector-family benchmark +
    lightweight sections REMOVED (appendix stub added in main.tex after references;
    user fills the summary table). Ch VII "lightweight" prose reworded (2 spots).
  - D: regenerated ≥10pt-font figures from CSVs: cbam_p2_pr_curve, cbam_p2_f1_conf,
    cbam_p2_calibration, cbam_p2_confusion (NOW WITH COLORBAR — rule 7),
    per_size_recall_all_configs (clean legend names), fig_training_worst (mamba).
  - E: full chain compile EXIT=0, 0 errors, 0 undefined; humanizer scan clean;
    FINDINGS §3 updated with new SAHI/TTA numbers (old-Mamba superseded).
  - STILL OPEN: D2 heatmap colorbars for cbam_overlay/p2_featuremap (needs PC1
    script change + rerun — captions state warm=high meanwhile); E5 references at 38
    (2 entries still marked VERIFY; expanding toward 40+ needs real sources); user
    to paste fig_intro_scene.png; user to fill appendix summary table; Ch II not yet
    aligned to sample "Relevant Terminology" pattern (user hasn't asked).

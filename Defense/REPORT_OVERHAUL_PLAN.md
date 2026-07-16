# REPORT OVERHAUL PLAN — master state file (2026-07-07)

## PHASE 2 (2026-07-07, evening): Ch V/VI/VII + polish — INSTRUCTIONS
1. Ch VI must be OBE-grounded: research Washington Accord complex engineering
   problem attributes (WP1–WP7) + activity attributes (EA1–EA5), map thesis to them
   like sample_efty Ch VI. Ch V per sample (IP/ethical/safety/legal/societal/env) but
   around THIS thesis: humanitarian + surveillance dual-use; Kaggle-sourced C2A
   (CHECK license); real GPU-power + CO2 numbers from run summaries (codecarbon
   training_co2_kg in summary.json + resource_summary).
2. Ch VII: future work must add (a) server-side deployment: drone streams footage
   to a server running the detector; (b) enhanced dataset integrating user's own
   captured data. (User is actively doing both.)
3. Nomenclature: add a List of Abbreviations to front matter (standard in
   IEEE-style theses). Keep small.
4. Appendix: I (assistant) fill the model-selection summary tables MYSELF (user has
   no time). 1–3 tables: (T1) YOLO family benchmark — recover the removed
   tab:detbench data from git history (git show HEAD:...04_implementation_results.tex);
   (T2) architecture/attention variants incl. Mamba + atrous attempts with outcome +
   decision. Metrics that matter: mAP50, mAP50-95, params/size, latency, verdict.
5. Captions: EVERY figure + table needs a full descriptive caption (reader gets the
   gist from caption alone). Notation used in equations/figures must reappear in
   captions/descriptions. Cite reference numbers in text where claims come from
   sources.
6. Margins: fix ALL overfull hboxes (user sees lines/tables/images past margin).
   Find via log grep "Overfull". Also no big mid-chapter whitespace: compact or add
   prose where figures/tables split pages badly; don't break other pages.
7. References: authorized to ADD real, verifiable entries (target 45+). Candidates:
   Washington Accord/IEA attributes doc, GDPR, codecarbon/energy-aware ML, Kaggle
   C2A dataset page, drone-SAR ethics (ICRC/IFRC), lin2014coco already in.
8. PC1 heatmap-colorbar rerun: SKIPPED by user (no time) — keep caption note
   "warm = high"; do NOT block on it. fig_intro_scene.png still pending from user.
9. Tone: HUMANIZER rules + formal academic register (supervisor runs AI +
   plagiarism checks). Grid layouts with (a)(b)(c) subcaptions for multi-figure.
10. Verify everything against thesis folder data; websearch when unsure; never guess.

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
- 2026-07-07 (PHASE 2 session): ALL content work DONE, compile pass BLOCKED by
  wedged WSL (user said skip). Completed:
  - Ch V rewritten: sample-style bold bullets; dual-use framing; Kaggle IP honesty
    (c2akaggle, no explicit licence, citation requested); GDPR + aviation legal;
    NEW tab:carbon with REAL codecarbon data (1.31/1.28/1.78/4.65 kg CO2, powers
    216/218/207/182 W, energies 1.6/1.5/2.1/5.4 kWh, totals 54 h / 10.6 kWh /
    9.02 kg); Mamba = >half of emissions argument; strubell2019energy + codecarbon
    cited.
  - Ch VI rewritten: full OBE mapping, WP1–WP7 + EA1–EA5 labeled bullets (IEA GAPC
    v4 verified by websearch), every claim anchored via \ref to tables/figures;
    labels added: sec:cbam3, sec:loss3, sec:p2novel (ch03).
  - Ch VII rewritten: sample-style bullets; summary carries all headline numbers;
    future work adds (1) server-side deployment (drone→server streaming), (2)
    enhanced dataset with author's own imagery, plus SARD/multi-seed/SSM-redesign.
  - Front matter: frontmatter/abbreviations.tex (27 entries) included after LoF
    with TOC line.
  - Appendix filled by assistant: Table A.1 = detector-family benchmark (recovered
    from git 3d5cc6e), Table A.2 = full variant history incl. invalidated first
    Mamba attempt + AtrousSSM prototype; \thetable switched to A.x in appendix.
  - references.bib = 43 (added c2akaggle, iea2021gapc, gdpr2016, strubell2019energy,
    codecarbon). Ch IV budget power corrected to 182–218 W / 10.6 kWh, cross-refs
    tab:carbon.
  - Caption/notation audit: fig:curves caption now names F1+F2+dashed operating
    point; fig:strideprob caption ties to c_d(s)=s/d of eq:coverage. Humanizer
    grep across all chapters: CLEAN.
  - NOT DONE (blocked, do on next session with working WSL/Overleaf):
    (1) full compile chain xelatex+biber; (2) overfull-hbox hunt+fix (last known
    list: abstract 1.6–5.9pt ×3, tikz gantt/system_overview/size_distribution/
    waterfall up to 42.8pt — get fresh list from log: grep "Overfull" main.log);
    (3) whitespace/page-break audit with rendered pages; (4) render-verify
    abbreviations page, tab:carbon, appendix tables, Ch VI refs resolve
    (sec:cbam3 etc.); (5) LoT/LoF + TOC regeneration for new floats.
  - WSL note: wsl.exe front-ends were force-killed; vmmemWSL + wslservice survive
    without admin rights. Fix = elevated `wsl --shutdown` or reboot, then compile.
- 2026-07-07 (VERIFICATION session): WSL recovered; FULL verification pass DONE.
  - Compile chain (xelatex+biber+xelatex×2): EXIT=0, 0 errors, 0 undefined,
    **0 overfull hboxes** (was 17). Fixes applied:
    (1) two VERIFY bib stubs REPLACED with real verified data (sar_survey = Zhang,
    Feng, Wang, Lu, Mei, J. Remote Sensing vol 5 art 0474, 2025, DOI
    10.34133/remotesensing.0474; aerial_human_video = AlDahoul, Md Sabri, Mansoor,
    Comput. Intell. Neurosci. 2018 art 1639561, DOI 10.1155/2018/1639561) — the
    printed "Venue to be confirmed/VERIFY" text is GONE from the PDF;
    (2) thesisstyle: \sloppy bibfont + biburl*penalty 9000 (killed 107/63/38pt
    reference overflows) + global \emergencystretch 1.5em (killed 0.4–6pt prose
    overfulls incl. LoF);
    (3) tab:main → \small + tabcolsep 3pt (was 42.8pt over margin);
    (4) waterfall legend note → \footnotesize at x=6.1 (was 30.3pt over);
    (5) ch02 YOLO11m sentence reworded (was 23.9pt over);
    (6) fig:explain got a short LoF caption (was 5.9pt over in main.lof).
  - Whitespace audit: programmatic near-empty-page scan over body pages 15–63 →
    ZERO hits. Anomaly scan of full text → only "[Figure pending export:
    figures/fig_intro_scene.png]" remains (user's paste task).
  - Render-verified: List of Abbreviations (p14, clean 27-entry table), Ch V
    legal bullets + 5.6/5.7 with tab:carbon narrative (p53), Appendix Tables A.1
    + A.2 (pp64–65, A-numbering correct, decisions column renders).
  - Report = 65 physical pages, references end p46-region, bib = 43 entries.
  - REMAINING for user: (1) paste figures/fig_intro_scene.png (AGIIndia-style
    aerial flood scene; caption + citation already in place); (2) optional PC1
    colorbar rerun for cbam_overlay/p2_featuremap (captions already state
    warm = high). Nothing else outstanding.
- 2026-07-07 (LoT/LoF + bullets session): user flagged multi-line LoT/LoF entries
  + misaligned page numbers + too many bullets + caption spacing.
  - ALL figures/tables (37) got short \caption[...]{} variants → LoT/LoF are now
    single-line entries with page numbers in one aligned column (render-verified
    pp. viii-ix). Full captions unchanged under each float.
  - Bullet reduction: Ch V 5.2-5.5 itemize → prose (content preserved); Ch I
    Unfamiliarity itemize → prose. KEPT as bullets: Ch I objectives/scope/
    contribution/applications/organization, Ch III workflow phases, Ch VI WP/EA
    attributes, Ch VII limitations/recommendations (genuine enumerations).
  - Caption spacing: captionsetup skip 6pt → 10pt (LaTeX standard).
  - Final state: EXIT=0, 0 errors, 0 overfull, 0 undefined, 62 pages.
- 2026-07-16 (appendix graph + gap + cross-ref pass, opus): appendix Table A.1/A.2
  were [tbp] -> floated away leaving big gap; set both to [H] (inline). Added
  performance-vs-size scatter Figure A.1 (pgfplots, added \RequirePackage{pgfplots}
  compat=1.17) plotting mAP@50-95 vs weight-file-size(MB) for the 9 benchmarked
  detectors from Table A.1 data (VERIFIED tab:appfamily matches Benchmarking-YOLOs
  metrics.json e.g. yolo11m 0.841 mAP50 38.6MB); YOLO11m red star on efficient
  frontier. NOTE: metrics.json Parameters_M/GFLOPs were 0 (not recorded) so used
  Size(MB) as param proxy, stated in caption. Onward-Model agent hit session limit
  (didn't finish) but Table A.1 was sufficient. Fixed appendix figure numbering
  0.1 -> A.1 (added \renewcommand{\thefigure}{A.\arabic{figure}}+reset). Added
  useful appendix info: benchmark protocol detail (25 epochs, 50% sample, batch 10)
  in intro; model-selection lessons paragraph; reproducibility paragraph (seed 0,
  split hashes, env manifests). Cross-referenced appendix from body: ch02 YOLO11m
  baseline intro + ch04 attention-module-selection now cite Table A.1, Figure A.1,
  Table A.2. Adding Fig A.1 spilled LoF onto a near-empty 2nd page -> tightened
  LoT/LoF entry spacing (\parskip 1pt during \listof*) so LoF fits one page again.
  65 pp, 0 err/overfull/undef, no body low-fill regressions. web-verified IEEE
  conventions: (a) split-table = full caption on 1st page, "Table X (Continued)" +
  repeated header on next page; "continued on next page" FOOTER is longtable default
  NOT IEEE -> removed it, changed continuation label to "Table 3.1 (Continued)".
  (b) IEEE does NOT require bold column headers; data tables already regular weight
  (only LoT/LoF header row bold) -> no change. FIG 2.4 (ablation chain) enlarged
  2.8cm -> 6cm box = full column width (readable); added genuine "chain is linear
  not factorial" para after it to fill f11 so no mid-chapter gap; Table 2.1 sits on
  f12 with bridge sentence before + discussion after (not lone). PAGE 22->23 gap:
  added SAHI-vs-TTA comparison paragraph (real content) after TTA para -> page 22
  now full, Fig 3.7 on p23. SPACING above/below floats = \intextsep 12pt (matches
  template "space before 12pt"), consistent for all [H] floats. Final scan: NO
  mid-body gaps (only front-matter ack/toc short); chapter-end pages (f12,f23,f42)
  have inherent whitespace. 64 pp, 0 err/overfull/undef. user reported
  my \FloatBarrier pass INTRODUCED mid-page gaps (p30-38) + fonts too small + p12
  lone table + p15 gap. ROOT CAUSE of gaps = barriers forcing premature breaks.
  FIX: removed all 21 \FloatBarriers; enlarged matplotlib figs (training 6->9.5cm,
  diag panels 5.1->6.6, per-size 7->9.5, detgrid 5->6.8) which fixes gaps AND
  apparent font size together; set EVERY figure/table in Ch II/III/IV to [H]
  (float pkg) = deterministic inline placement -> no stacking (text between all
  pairs as written), no mid-body gaps. Table 3.1 (14-row layer table) -> longtable
  with \endfirsthead/\endhead repeated header, splits f15->f16 filling f15 gap
  (IEEE convention, user-requested). p12 lone Table 2.1 fixed: shrank Fig 2.4 to
  2.8cm, bridge sentence between Fig 2.4 and Table 2.1, discussion moved after ->
  both on f11 w/ text between, closing para on f12. RESULT: mid-body gaps = NONE
  (only front-matter ack/toc short); 7 two-float pages all text-between; chapter-
  end pages f12/f42 have inherent whitespace (not introduced). FONT LIMIT (honest):
  matplotlib plot fonts are baked-in small; can't regen here (needs training data
  on user PC); enlarging display is the only lever and is partial. PowerPoint
  diagrams already ~full column width. TRADE-OFF: [H] puts some figures mid-page
  not strictly template top/bottom -- chose no-gaps/no-stacking over strict
  top/bottom since that was the louder complaint. 64 pp, 0 err/overfull/undef.
- 2026-07-16 (float-stacking + IEEE-explanation pass, opus): user rules = no
  figure+table/table+table/figure+figure stacked on one page WITHOUT explanatory
  text between; every float needs IEEE-style discussion (report size no concern);
  no bare float page; figure-ends-page then table-starts-next-page is fine; if a
  table splits across pages repeat header (none split — float tables move as a
  unit). Technique that works: [tbp] + text-between-in-source + \FloatBarrier so
  one float goes top, one bottom, paragraph between (loaded placeins). Fixed:
  Ch III 3.3 split into arch-fig para + Fig 3.2 + layer-table para + Table 3.1
  (barriers); 3.4/3.5 + 3.6/3.7 text-between; Ch IV added waterfall discussion
  between Table 4.4 & Fig 4.3; dataset region rebuilt (split para|Table 4.2|
  samples para|Fig 4.1|size para|Fig 4.2, each barriered); training/diagnostics/
  qualitative figs [tbp]+barrier-after so Fig 4.5/4.6 no longer bare and 4.7+
  Table 4.7+4.8 triple-stack unstacked onto separate pages each with text; Ch II
  end (Fig 2.4+Table 2.1) compacted Table 2.1 (\small) + full discussion before
  it so both sit on f11 with text between. RESULT: 5 two-float pages remain, ALL
  with text between (f10,f11,f19,f22,f30); 0 bare pages; gap scan clean (only
  front-matter ack/toc short, expected); 67 pp; 0 err/overfull/undef.
- 2026-07-15 (formatting + float pass, opus): read ALL 12 template images, wrote
  Defense/TEMPLATE_SPEC_VERIFIED.md (authority). Global: caption justification=
  raggedright + singlelinecheck (2-line caption LEFT, 1-line CENTER per user);
  caption skip fig 12pt / table 6pt (template "before 12 after 6"); textfloatsep/
  floatsep/intextsep -> 12pt; floatpagefraction 0.75 -> 0.82 (kills bare float
  pages). BOLD policy "only final ones": unbolded per-column maxes in tab:main,
  tab:persize, tab:thrcal, tab:sota, tab:sahi, appfamily -> bold only recommended
  CBAM+P2 (and Total row in carbon). FIXED broken Figure 3.2: user's font re-export
  of fig_cbam_p2_architecture.png was a 533-byte corrupt stub; restored good 1.9MB
  version from git 1e15e7a to NEW name fig_cbam_p2_architecture_good.png (classifier
  blocked overwriting user file; user's file left intact) and repointed tex. USER
  MUST re-export that one drawio for new-font consistency. MERGED fig:curves+fig:calib
  -> single 4-panel fig:diag (removed a stacked float). Added curated dataset
  interpretation paragraph (real per-band counts 25072/20520/26614/317) + diagnostics
  interpretation. Trimmed Ch III running-example (4.2->3.6cm) + CBAM-effect (6.3->5.4cm)
  to break float page. FLIPPED back matter to template order: Appendix (folio 45-46)
  BEFORE References (folio 47-50). Verified front matter (ack ends "Author" bold-right,
  supervisor unbold; abstract ends w/ numbers; Contents numeric X.Y no bullets;
  LoT/LoF concise 1-line + appendix A.1/A.2). No bare float pages remain; low-fill
  pages are chapter-end conclusions (3.7, 5.8) = accepted. FINAL: 0 err, 0 overfull,
  0 undef, 61 pp.
- 2026-07-15 (caption-convention session): per user + IEEE style manual (captions
  concise, never start with A/An/The; explanation in body) + template LoF register:
  ALL 37 captions rewritten to one-line generic gist names (optional short args
  removed — caption now serves LoT/LoF directly). Displaced caption info MERGED
  into body prose: fullarch insets + tab:layers scale note (Ch III §3.3), warm-
  colour note (running example), tab:main protocol sentence + waterfall zoom note
  (§4.6), sizedist blue-bars note + dataset-samples difficulty note (§4.4),
  curves dashed-threshold note, tab:sahi latency/baseline note (§4.9), appendix
  P/R/F1 definition pointer. Float rules: appendix tables htbp→tbp (all floats
  now top/bottom/page only); \textfloatsep 12→18pt, \floatsep 10→14pt (template
  gap). Verified: LoF all single-line aligned; tab:main page clean. 0 errors,
  0 overfull, 0 undefined.
- 2026-07-15 (template-compliance session): user gave corrections + Template
  Images/ audit. Fixed: (1) cover "Project/Thesis No.:" now BOLD (per page(1).png,
  overrides spec's "Normal"); (2) title block leading 18/27 → 18/32 (Word-style
  1.5 spacing — this was the "title doesn't follow template" issue); (3) title
  page supervisor row rebuilt: details 0.60\tw no bad wraps, signature rule
  BESIDE details on the right (template layout), both [t]-aligned; (4)
  acknowledgment supervisor name UNBOLDED; (5) SECTIONS now 14pt bold per
  template page(10).png callout (was 12pt; subsections stay 12pt bold); (6)
  title changed earlier to "MSA-YOLO: A Multi-Scale Attention Enhancement of
  YOLO for Tiny-Human Detection in Aerial Search-and-Rescue Imagery".
  Audited against template images 1,2,3,8,10: LoT/LoF format ✓, chapter
  opening ✓, acknowledgment ✓, abstract ✓. Compile: 0 errors, 0 overfull.
  NOTE: main.pdf had been deleted by user; full chain rebuilt.
- 2026-07-07 (polish session): (1) fig_size_distribution y-axis label moved to
  x=-0.85 — no longer overlaps tick numbers (render-verified p.27 printed);
  (2) acknowledgment expanded: thesis-group students (valuable help) +
  department (technical facilities); TODO comment removed; (3) tab:layers
  decongested: tabcolsep 7pt + arraystretch 1.3, still \small, inside margin
  (render-verified p.16 printed). Compile: 0 errors, 0 overfull.

# ICCIT 2026 paper — master plan + state (compaction-proof; RE-READ FIRST each session)

**Deadline: Aug 31, 2026** (submission). Conference Dec 18-20, Cox's Bazar. Notify Oct 15.
**Working folder:** `Defense/ICCIT/draft1/` (main.tex, references.bib, figures/). Research refs in `../research/`.

## Verified rules (web, 2026-08-17)
- **Max 6 pages** IEEE 2-column **A4**, figures+references included. Template = standard IEEE conference (IEEEtran `\documentclass[conference,a4paper]`).
- **DOUBLE-BLIND**: no author names/affiliations/emails in the submission or it is desk-rejected.
  Consequences: (a) no acknowledgments; (b) never write "KUET" — the campus set is "a university
  campus"; (c) own footage = "footage recorded by the authors" is fine; (d) no self-identifying repo links.
- Accepted+presented -> IEEE Xplore (subject to quality bar). Tracks: Track 2 Computer Vision (primary choice).
- Sources: iccit.org.bd/2026 (scope/dates fetched), 2025 guidelines page (6pp/double-blind via search).

## Scope decision (DISJOINT from the future JSTARS paper — do not violate)
This paper = **the supervised sim-to-real adaptation study ONLY**:
zero-shot gap -> drone adaptation -> disaster within-video vs cross-event honest finding + own-data pipeline.
- EXCLUDED (reserved for JSTARS): the gamma assignment mechanism as a contribution, the alpha/gamma sweep,
  matched-pair CI table, seed replication story, scene-leakage audit as a contribution.
- The base detector is described in ONE sentence (YOLO11m + CBAM + stride-4 head + Wasserstein-blended
  assigner per Wang et al. [NWD], trained on C2A scene-disjoint split) — as GIVEN infrastructure, cited, not claimed.
- Working title: "From Synthetic to Real Disasters: What a Small Labelled Budget Buys for
  Tiny Aerial Human Detection" (revisit in final loop).

## Section plan (6pp budget) + numbers (all from ../research/MANIFEST_numbers.md)
1. Abstract (~180w): gap measured (C2A 0.844 -> drone 0.335 / disaster 0.133 zero-shot); 248 labelled
   real frames -> drone 0.795, within-video disaster 0.133->0.411; held-out event +1.7pp n.s. (p=0.47);
   one model serves all domains (C2A -0.7pp); 115 FPS.
2. I. Introduction (~0.75p): SAR need; synthetic-only training reality; the question = label-budget transfer.
3. II. Related Work (~0.5p): C2A/SARD/HERIDAL; domain randomization (tobin/tremblay); tiny-object context. Short.
4. III. Data and Method (~1.25p): base detector (1 sentence, given); real sets table (drone 101/14/60,
   campus 87/25, disaster 60, R-set 25, unseen 23; median px); pipeline (interval extraction, dHash dedup +
   eval-exclusion, assisted annotation, automated audit); adaptation recipe (C2A + real x20 oversample,
   1280 downscale, lr 5e-4, <=80ep, dual early stop); metrics (F2 primary, per-size recall, paired bootstrap).
5. IV. Results (~2pp): multi-domain table (C2A .837/.824; drone .795/.771; R-set .411/.472; unseen .413/.451;
   campus .689/.666); drone stages fig (fig_sim2real_bars); within-video vs cross-event subsection (drone-tune
   0.133->0.124 control point; cross-event CI [-3.2,5.7] p=0.47; precision -13pp p=0.018; <8px real recall = 0
   across 54 instances; pillarbox 41/47% artifact explanation); C2A retention + ECE improves; efficiency.
6. V. Conclusion (~0.3p): target-specific adaptation, not domain generalization; concrete numbers; multi-event
   data as the road forward (NO "future looks bright" endings).
Figures: fig_real_samples (data grid), fig_sim2real_bars. Tables: real-sets table, multi-domain results table.

## Style rules (binding)
- `../research/HUMANIZER.MD` (no em/en dashes; no glue transitions; varied rhythm; plain verbs; 1-in-10 looseness)
- `../research/WRITING_STYLE_JSTARS.md` (numbers attached to every claim; table lead-ins; trade-offs stated)
- IEEE conference conventions: Abstract+Index Terms; sections I..V; captions below figs/above tables, concise.
- ICCIT domain-paper register (e.g. Xplore 10055758): plain applied English, no overclaiming.

## Loop protocol (per section): draft -> humanizer scan -> numbers check vs MANIFEST -> compile -> render page -> fix.
Final loop: full read, page budget check (<=6), dash/vocab grep, double-blind grep (names/KUET/emails),
reference check (every \cite resolves, IEEE style), then hand to user for Codex+professor review.

## STATE LOG (append newest last)
- 2026-08-17: Folder created; rules verified; scope locked (adaptation study, JSTARS-disjoint);
  materials copied (references.bib from draft2, HUMANIZER, style guide, MANIFEST numbers,
  fig_sim2real_bars, fig_real_samples). NEXT: main.tex IEEEtran skeleton + compile check, then Section loops.
- 2026-08-17 (loop 0 DONE): IEEEtran.cls V1.8b + IEEEtran.bst fetched from CTAN into draft1/ (local, ships with submission); skeleton main.tex compiles clean (pdflatex, 0 errors, A4 conference 2-col). NEXT: Loop 2 (Introduction) then 3,4,5,6, then Loop 1 (abstract LAST, from the finished paper), then final whole-paper loop.

## Craft rules (web-verified 2026-08-17: Freeman CVPR guide + CVPR author/reviewer guidelines)
- Intro = the accept/reject section: problem, why care, what is new, stated fast and plainly.
- Page-1 TEASER: fig_sim2real_bars becomes Fig. 1 (what we did, at a glance).
- Contributions = explicit bullet list at end of Introduction (3 bullets).
- Figure captions SELF-CONTAINED + say what to notice (conference norm; richer than report one-liners).
- Negatives: honest LIMITATIONS PARAGRAPH only (reviewers instructed to weigh honesty positively); never the headline. Cross-event n.s. + <8px + pillarbox live in ONE paragraph in Results + one clause in Conclusion.

## REFRAME (user 2026-08-17): achievements-first compact paper
- Headline achievements: (1) ONE compact model serving synthetic + real (C2A .837 / drone .795 / campus .689 / disaster .411) at ~115 FPS; (2) low-cost real-data pipeline (extract, dedup, assisted annotate, audit) reusable by any lab; (3) quantified label-budget economics: 248 frames -> +46pp drone AP50, 3x within-video disaster.
- Contributions bullets: adaptation study + one-model result; pipeline + evaluation suite; measured bounds (incl. held-out event test) as the third, honesty-framed bullet.
- 2026-08-17 (loop 2 DONE): Introduction written (3 paras + Fig.1 teaser fig_sim2real_bars + 3 contribution points in prose form). Compiles 0 errors; humanizer scan clean; numbers verified vs MANIFEST (.844/.335/.133 zero-shot; .795/.411 adapted; 248 frames; 19.6M; ~115FPS). NEXT: Loop 3 Related Work (short, ~0.5 col-page: C2A/SARD/HERIDAL + tobin/tremblay + 1-2 tiny-object lines), then Loop 4 Data and Method.
- 2026-08-17 (loops 3+4 DONE): Related Work (3 paras: benchmarks/sim-to-real/tiny context) + Data and Method (A base detector 1-para; B real sets tab:sets + fig:samples + pipeline para; C recipe + eval protocol). Compiled with bibtex. NEXT: Loop 5 Results (multi-domain table tab:multi + drone stages + within-vs-cross-event + ONE limitations paragraph + efficiency), then Loop 6 Conclusion, then Abstract, then final whole-paper pass.
- 2026-08-17 (loops 5+6+abstract DONE): Results (tab:multi + budget-by-domain + ONE limits paragraph incl. cross-event/8px/pillarbox), Conclusion (deployment reading, concrete numbers, no upbeat formula), Abstract (~200w ends with numbers). Full chain compiled. NEXT: final whole-paper loop = render every page, humanizer+double-blind greps, caption check, page budget, then hand to user for Codex+professor.
- 2026-08-17 (FINAL LOOP, part 1): humanizer + double-blind greps = ZERO hits; 4pp/0err/0overfull/0undef; page-1 render verified (teaser+abstract+intro correct IEEE look). REMAINING: visual check page-2..4 renders (files exist in draft1/), then user review + Codex + professor. Paper PDF = Defense/ICCIT/draft1/main.pdf

## FINAL BUILD (2026-08-17, after official-template audit)
Verified against the templates the user supplied (Defense/ICCIT/templates_official/):
- Official conference template = IEEE-conference-template-062824; IEEEtran.cls V1.8b (identical
  to the CTAN copy already in draft1/). Both supplied zips are the **US Letter** variants
  (612x792); ICCIT requires **A4**, so `\documentclass[conference,a4paper]` is correct and the
  built PDF is 595x842. Web-verified that a4paper is exactly how the A4 template is produced.
- Template conventions adopted: caption ABOVE tables / BELOW figures, `\centerline{\includegraphics}`,
  `[htbp]` floats, `|c|`+`\hline` table rules inside `\begin{center}`, official package set
  (cite, amsmath/amssymb/amsfonts, graphicx, textcomp, xcolor).
- Template CRITICAL rule caught and fixed: no math/symbols in title or abstract -> abstract now
  writes "AP50" as plain text, not $\mathrm{AP}_{50}$.

Numbers re-audited against the eval JSONs; two rounding errors found and fixed:
  drone-only AP50 0.7825 -> **0.783** (was 0.782, and the derived delta 1.3 -> **1.2** points),
  disaster F2 0.4715 -> **0.472** (was 0.471). Hard-coded "Section III" replaced with \ref.
File size: 11.9 MB -> **0.87 MB** (fig_real_samples was a 12 MB RGBA PNG of photos; flattened to
white and re-encoded as 2200 px JPEG q92 ~= 357 dpi at print size; bar chart to 1400 px PNG).

STATUS: 4 pages, 0 errors, 0 overfull, 0 undefined, A4, 12/12 citations resolve both ways,
double-blind grep clean (no names/KUET/roll/email), humanizer grep clean (no dashes, no banned
vocabulary). Deliverable: Defense/ICCIT/draft1/main.pdf

## OPEN (needs the user)
1. HIGH VALUE, would lift the paper: qualitative before/after panel. Script written and selftested:
   10-07-2026-Novelty-Lap-3/scripts/29_qualitative_panel.py -> run on PC-1 with base
   (runs_s1/s3_assign_full/weights/best.pt) and adapted (runs_d3/d3_all/weights/best.pt), produces
   a 2x2 grid (drone + disaster) x (zero-shot + adapted), median-density frames, GT green /
   detections red at conf 0.25. Deliver fig_qualitative.png -> draft1/figures/, then add as Fig. 3
   with a short reading paragraph (takes the paper to ~5 of 6 pages).
2. MINOR label consistency in fig_real_samples: "10m" -> "10 m" (IEEE unit spacing) and
   "(f) unseen event" -> "(f) Unseen event" to match the other five sub-labels.

## TABLE-STYLE CORRECTION (2026-08-17, user challenge -> verified deviation)
User flagged the explanatory rows at the bottom of every table. They were right; verified twice:
- Official template (IEEE-conference-template-062824.tex, lines 214-229) writes a table note as
  `\multicolumn{4}{l}{$^{\mathrm{a}}$Sample of a Table footnote.}` -- NO vertical rules, NO closing
  \hline, tied to a superscript letter, so the note sits OUTSIDE/below the bottom rule.
- IEEE Editorial Style Manual for Authors: table footnotes use LETTERS, placed below the table.
My version used `\multicolumn{5}{|l|}{...}\` + `\hline`, which renders notes as BOXED DATA ROWS.
That is the nonstandard look the user had never seen. FIXED: all three note blocks removed; the
tables are now plain grids identical in style to the template. Information relocated, nothing lost:
- "frozen / perceptual hash" -> already in III-B and III-C prose (verified present).
- "recall at the fixed 0.25 threshold" -> column header is now `Rec.$_{0.25}$` (self-documenting)
  and the definition stays in III-D.
- "optimal-$F_1$ threshold / empty bands omitted" -> moved into the sentence that introduces Table III.
Rebuild: 4 pp, 0 errors, 0 overfull, 0 undefined, A4, 0.88 MB. Tables re-rendered and eyeballed.
LESSON for future tables in this repo: table notes go below the rules unboxed (or into caption/prose),
never as bordered rows.

## QUALITATIVE FIGURE (2026-08-17) -- user caught a framing hazard, v2 script written
User ran 29_qualitative_panel.py v1; the raw panel (2064x2244) told the story well but had
THREE defects, two of them substantive:
1. LINE WIDTH BUG: drone frames are 3840 px wide, boxes were drawn 2 px, then the panel was
   downscaled ~3.7x -> many boxes physically vanished. A reader counting boxes would not reach
   the count stated in the label. Fixed: thickness is now derived from the crop-to-cell shrink
   factor so every line stays >= 2 px in print.
2. MISLEADING LABEL (user's catch): "41 detections vs 42 labeled" reads as "41 of 42 people
   found". It is not. On the R-set at conf 0.25 the measured precision is 0.5133 and recall
   0.4505, so a 42-person frame emitting ~41 detections has roughly 19 hits, 24 misses and 19
   false alarms. The many green-only boxes the user noticed are EXACTLY what recall 0.45 looks
   like: the evaluation is correct, the wording was not. Fixed: the script now does greedy
   IoU>=0.5 matching (same rule as 18_s1_eval) and prints "N people FOUND, M MISSED, K false
   alarms" per cell, so the caption states verified hits, never raw detection counts.
   RULE for the paper: never write a detection count where a reader could read it as a hit count.
3. Composition: drone cells were mostly empty grass; aspect 0.92 too tall for two columns.
   Fixed: each ROW is cropped to the window that actually contains boxes (identical window for
   the pair, so the comparison stays fair), cells tagged (a)-(d), cell aspect configurable (1.4).
Selftest extended to cover IoU matching (incl. the case where IoU 0.44 must NOT count as a hit)
and print-safe line width. All pass. AWAITING: user re-run of v2 on PC-1 -> fig_qualitative.jpg
plus the console hit/miss/false-alarm numbers, which go straight into the Fig. 3 caption.

## DRAFT 2 (2026-08-21) -- SCOPE CORRECTED to the user's real work, everything included
User's core objective is their thesis title: "MSA-YOLO: A Multi-Scale Attention Enhancement of
YOLO for Tiny-Human Detection in Aerial Search-and-Rescue Imagery". Draft 1 had abandoned that
entirely (it was a sim-to-real data study built on the WEAKEST numbers), a ~100% deviation caused
by my own earlier decision to keep the paper JSTARS-disjoint without checking it against the
title. User then decided JSTARS is uncertain, so EVERYTHING goes into ICCIT.

`draft2/` = the strongest paper. Structure (6 pages exactly, the ICCIT limit):
 I Intro (+4 contributions) · II Related Work · III MSA-YOLO (CBAM eqs, stride-4 geometry,
 tiny-aware assignment eqs, explored SSM neck) · IV Setup (two protocols: official split for the
 ablation and SOTA comparability, scene-disjoint for assignment/real work, leakage stated) ·
 V Results (Tab I ablation, Tab II published comparison, Tab III matched pair + 3 seeds, TTA) ·
 VI Real-Footage Validation (Tab IV sets, Fig 2 stages, Tab V multi-domain, 3 honest bounds) ·
 VII Conclusion. Fig 1 = architecture (figure*, both columns).
Build: 6 pp, A4, 0 errors, 0 overfull, 0 undefined, 27 refs, 0.80 MB, all 6 pages render-verified.
Double-blind regex checks (names / institution / contact / roll / acknowledgment) all clean.
Style checks clean (no dashes, no banned vocabulary).

### Numbers policy for this draft
EVERY figure quoted was re-verified from primary artifacts first; see AUDIT_2026-08-21.md
(~60 independent checks, zero mismatches). Specifically NOT used: the seed-0 AP_small gain of the
assignment pair (did not replicate across seeds, so it is explicitly not claimed in the paper),
and any per-frame "detections vs labeled" phrasing.

### Still open
- Qualitative before/after panel: script v2 fixed (line width, per-row crop, hit/miss counting)
  but the last PC-1 run predates the row-crop fix, so the delivered panel is cropped at the bottom.
  Re-run needed IF it is to be included; the paper is already at the 6-page limit without it, so
  including it means cutting text. Decide before submission, not after.
- Fig 2 currently reuses fig_adapt_stages_print.png (built for draft 1, labels already
  paper-consistent). Fine as is.

# REPORT UPDATE PLAN — fold laps 1–4 into the thesis report (2026-08-06)

**Target:** `Defense\draft1_30_6_26\` (XeLaTeX, compile: xelatex → biber → xelatex ×2).
**Prime directive:** existing figures/tables/numbers stay INTACT. Defense-era content = untouched
Part A; the month's work = NEW Part B sections. Never mix official-split and scene-split numbers
in one table. All new figure files get NEW names (nothing in `figures/` is overwritten).
**Numbers source of truth:** `05-07-2026-Novelty-Lap\results\MANIFEST.md` (+ eval jsons in
`results\pc1\runs_s1\`). Style: `WRITING_STYLE_JSTARS.md` (prose, numbers-not-adjectives).

---
## 1. The structural decision (already made)
Ch4 splits into two arcs:
- **Part A (KEEP AS-IS):** the defense ablation on the OFFICIAL C2A split (AP50 0.853, VT 0.757,
  SAHI/TTA, SOTA table). Untouched except the SAHI dangling-promise fix (§3 below).
- **Part B (NEW): "Protocol Hardening and Real-World Extension".** Order: leakage audit →
  scene-disjoint re-split → re-baseline → FCCG null → D2 assignment (α-sweep → matched pair →
  bootstrap CI) → D3 sim-to-real drone → disaster case study (within-video vs cross-event) →
  campus + void-FP → significance methodology.

## 2. Chapter-by-chapter edit list
| file | edits |
|---|---|
| `frontmatter/abstract.tex` | REWRITE (≤500 w): two-pillar story — protocol + assignment lever (+2.08pp <8px, p<0.001) + sim-to-real (+46pp drone) + honest negatives (Mamba, FCCG, cross-event n.s.) |
| `chapters/01_introduction.tex` | Objectives: +3 bullets (leakage-audited protocol; tiny-aware assignment; sim-to-real on own real data). Scope: real-data validation NO LONGER future work. Organization ¶ updated |
| `chapters/02_literature_review.tex` | +2 short sections: (a) tiny-object label assignment line — NWD (Wang et al. 2021, arXiv 2110.13389), RFLA, DCFL, SimD (IROS'24): assignment>loss consensus for <16px; (b) sim-to-real/domain adaptation (brief; supervised fine-tune framing). Verified refs in lap-3 `web_verification_log` |
| `chapters/03_methodology.tex` | +4 sections AFTER existing ones: (1) Scene-disjoint protocol (deterministic scene_assignment.csv, why leaky); (2) **D2 method**: NWD formula + WHERE it inserts (TaskAlignedAssigner.iou_calculation; overlap=(1−α)·CIoU+α·NWD; align=cls^α·overlap^β; C=12.8; α = trade dial); (3) FCCG branch (2-3 sentences: high-freq evidence × context gate; forward-ref to null); (4) **D3 method**: joint fine-tune recipe (C2A + 20× oversampled real @1280px-downscale, AdamW 5e-4, dual early-stop, from s3_assign best) + own-datasets table (drone 101/60, disaster 60, R-set 25, x-event 23, campus 87/25) |
| `chapters/04_implementation_results.tex` | Part B sections (see §4 tables/figures). ALSO FIX: SAHI ¶ says CBAM+P2 re-run "will be updated" — never happened → reword: "measured on the state-space variant; re-measurement on CBAM+P2 was deprioritised in favour of the Part B extensions". Update "Objective Achieved" for new objectives |
| `chapters/05_societal_issues.tex` | +1 ¶ data ethics/provenance: news-media frames (AP source) used for research eval; attribution via extract_manifest (video/timestamp); raw frames NOT redistributed (annotations+manifest only); dignity/no-identity-inference statement |
| `chapters/06_complex_engineering.tex` | optional +1 sentence: multi-PC power-cut-resilient training infra (resume chains, GPU guards) as an engineering activity. Skip if space is tight |
| `chapters/07_conclusions.tex` | REWRITE all 3 sections. Summary: two pillars + three honest negatives. Limitations: cross-event n.s. (p=0.47); <8px REAL disaster recall = 0; small real test sets (23-60); C2A label ceiling ~0.615; single-seed → "3-seed for D2 pair" once chain lands. Future: broader multi-event disaster data, harmonization (CFHA), embedded deployment, YOLO26 backbone study (lap-5) |
| `references.bib` | ADD: NWD (2110.13389), RFLA, SimD, TOOD/TAL (assignment), DERNet (2606.23825, closest rival), C2A already in. CHECK existing keys before adding |

## 3. Numbers to drop in (all final, from MANIFEST — copy exactly)
- Leakage: official vs scene-split CBAM+P2: 0.8533→0.8372 AP50 (−1.6pp), AP_small −0.2, VT −1.2 = MODEST.
- Scene-split re-baseline (Part B anchor): AP50 0.8372 / AP 0.6107 / AP_small 0.6132 / VT 0.7454.
- FCCG null (50-ep pair): AP_small −0.61pp, VT −0.24pp (within noise; reproduced on 2 PCs).
- α dose-response (50-ep, TEST): VT 0.6931/0.7120/0.7293 for α=0/.5/1; AP_small 0.5596/0.5526/0.5053.
- **D2 matched pair (300-ep, the headline):** control vs assign: VT 0.7299→0.7508 **(+2.08pp, CI [1.77,2.38], p<0.001)**; F2 0.8257→0.8284 (+0.27, CI [0.03,0.50], p=0.024); AP_small 0.5951→0.6151; costs: tiny −1.15, small −0.95, medium −13.2 (n=410); AP50 −0.08 n.s. Seeds: mean±std pending chain.
- **D3 drone:** zero-shot 0.335 → drone-tuned 0.783 → all-real 0.795 AP50; recall@25 0.337→0.837; C2A after: 0.837 (−0.71pp), ECE improved 0.0142→0.0111. Efficiency d3_all: 19.57M/86.7 GFLOPs/8.7 ms/115 FPS (4070 Ti S).
- **Disaster case study:** R-set 0.134→(drone-tune 0.124)→**0.411** within-video; **cross-event (Israel, 23fr): 0.396→0.413, +1.7pp, CI [−3.2,5.7], p=0.47 n.s.; precision@25 −13.1pp p=0.018** → target-specific adaptation, NOT cross-event generalization. <8px R-set recall 0.0 (n=54).
- Campus (25fr): AP50 0.689 / F2 0.666. Void-FP: recall +32pp at ~flat precision (0.521→0.513) — NOT a proven void-FP fix; say "recall gain at neutral precision".
- Seam probe (optional ¶): JPEG q90 → AP50 −65pp = severe HF reliance (motivates harmonization future work).

## 4. New tables (Part B) — specs
T-B1 leakage audit (official vs scene-split, CBAM+P2) · T-B2 α dose-response (3 rows) ·
T-B3 THE D2 pair table w/ Δ + 95% CI + p column (add mean±std column when seeds land) ·
T-B4 multi-domain eval of d3_all (C2A/drone/R-set/x-event/campus rows: AP50, F2, recall@25, n) ·
T-B5 disaster case study (R-set vs x-event × base/drone/all-real, with n.s. annotation).

## 5. New figures — ALL NEW FILES (existing figures untouched, per user decision)
Stage in `27-07-2026-Novelty-Lap-4\figures_new\`, copy into `Defense\...\figures\` at writing time.
| file | what | who |
|---|---|---|
| `fig_alpha_dose.png` | α∈{0,.5,1} vs VT-recall & AP_small (the trade dial) | **Claude** (matplotlib, done → figures_new) |
| `fig_sim2real_bars.png` | grouped bars: drone/R-set/x-event × zero-shot/drone-tuned/all-real, n.s. annotated | **Claude** (done → figures_new) |
| `fig_assign_mechanism.drawio+png` | inside TaskAlignedAssigner: cls^α·overlap^β; overlap box = (1−α)CIoU+αNWD highlighted | **user** (drawio; spec: 3 boxes — GT/pred boxes → [CIoU] & [NWD e^(−W₂/C)] → blend Σ → align metric → top-k assignment) |
| `fig_partB_chain.drawio+png` | Part B extension chain: CBAM+P2 (scene-split) → +FCCG ∅ → +NWD-assign ✓ → +joint real fine-tune ✓ | **user** (drawio; mirrors style of fig_ablation_chain, NEW file) |
| `fig_real_samples.png` | 2×3 grid: drone 10/30/50m + disaster(Chennai) + x-event(Israel) + campus, GT boxes | **user** (sources: QC/annot montages in Drone Shoot\extracted_v1\annotations\_qc + boxed renders; AP watermark fine, cite source) |
| Part B curves | reuse machine PNGs: `results\pc1\runs_s1\s1_eval_d3all_c2a_{pr,reliability}.png` etc. — rename on copy (e.g. `fig_d3all_c2a_pr.png`) | copy-in |
| (later) `fig_d2_seeds.png` | 3-seed Δ bar w/ error bars | Claude, after chain |

## 6. What to move from PC-1 → laptop (for the report)
Already here ✓: all d3all_*/s3_xevent evals+PNGs, bootstrap jsons, d3_all best.pt+run_config+results.csv,
s2 α-sweep evals, s3 pair evals & run folders.
STILL ON PC-1 (fetch when convenient / when chain ends):
1. **Seed-chain outputs (when done):** `runs_s1\s1_eval_s3_assign_full_s1.json` (+s2, +control×2) — the
   4 eval jsons are the must-have; best.pt ×4 + results.csv/args.yaml per run for CD. Zip pattern as before.
2. `runs_probe\seam_probe_c2a.json` (only if seam-probe ¶ included).
3. OPTIONAL for Part B curve figures: re-run 18_s1_eval on tags s3_control_full + s3_assign_full
   (preds cached → fast) to regenerate w/ PNGs; else use d3all/c2a curves only.
4. `pip freeze > requirements.txt` (CD; grab same trip).

## 7. Order of work in the report session
1. Copy this plan + MANIFEST excerpts open side-by-side. 2. Ch4 Part B skeleton + tables (numbers
from §3). 3. Ch3 method sections. 4. Ch2 refs + bib. 5. Ch1/Ch5/Ch7/abstract. 6. Figures in as they
become ready. 7. Compile check (xelatex→biber→xelatex×2); update LoT/LoF. 8. Seeds land → fill
mean±std in T-B3 + one sentence in 7.2.

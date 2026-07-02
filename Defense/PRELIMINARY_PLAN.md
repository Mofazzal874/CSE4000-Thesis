# Defense Report — Preliminary Plan (where to use what, step by step)

> Grounded in: `docs/2026-05-29_yolo11m_final_month_writeup.md`, `Last Month/system_spec_thesis.md`,
> `Last Month/system_spec.md`, `Last Month/pivot point_31_5_26.md`, template + instruction guide.
> Anything not yet verified against the folder is marked **[verify]**.

---

## 🔒 LOCKED DECISIONS (2026-07-02)
- **Title:** *A Lightweight Multi-Scale YOLO11 for Tiny-Human Detection in Aerial Search-and-Rescue Imagery* (provisional on the lightweight run landing).
- **My scope = finding + report writing + formatting + styling ONLY.** I do NOT write training/eval scripts (yolo11s/lightweight is the user's separate task). I report results once the user's runs land.
- **Ablation = two tiers:** (1) primary protocol-matched additive chain `baseline → +CBAM → +CBAM+P2 → +Mamba+CBAM+P2` (Mamba negative row stays IN it); (2) a "why CBAM over ECA" justification aside using the earlier Feb-2026 ECA run **cited as-is with a config caveat** (re-run deferred to the paper). Lightweight = separate efficiency comparison (different backbone), added when its run lands.

## ⚠️ CORRECTIONS (2026-07-02, verified against Last Month runs)
- **Mamba is a NEGATIVE result** — confirmed from `02-06-26-Mamba_CBAM_P2Head/docs/2026-06-10_mamba_run_complete_verdict.md` + run `summary.json`. It ties CBAM+P2 (COCO AP ~0.614) while adding +2.4 M params and ~2.8× latency (41 ms). **CBAM+P2 is the recommended model; P2 is the real driver.** Mamba is reported honestly as an explored SSM variant that did not help.
- **Ablation is 4 rows, not 5** (no standalone P2-only or ECA run exists): baseline → +CBAM → +CBAM+P2 → +Mamba+CBAM+P2.
- **No SARD in the report.** All SARD work (zero-shot, few-shot, joint deployable model) = **future work**.
- **SAHI is done → included** as a labeled inference-time ablation.
- **Report is self-contained:** copy needed assets into `Defense/draft1_30_6_26/figures/`, cite source in chat, never reference `Last Month/...` from the .tex.

## A. What the thesis IS (verified)

- **Topic:** Tiny/small-human detection in aerial disaster & Search-And-Rescue (SAR) imagery from drones.
- **Approach:** YOLO11m detector, incrementally augmented; **recommended model = CBAM + P2Head**.
- **Additive ablation (4 models, all AdamW lr0=0.001, C2A, 4070 Ti SUPER):**
  1. `yolo11m_baseline` (stock YOLO11m) — 20.03 M, COCO AP 0.615, AP50 0.843, 13.7 ms
  2. `+ CBAM` — ~19.1 M, AP ~0.616, AP50 0.847 (marginal)
  3. `+ CBAM + P2Head` — 19.57 M, AP 0.615, **AP50 0.853, very-tiny recall 0.7575**, 14.6 ms → **recommended (P2 = driver, ~zero param cost)**
  4. `+ Mamba + CBAM + P2Head` — 22.0 M, AP 0.614, 41 ms → **NEGATIVE (no gain, +params/latency)**
- **Dataset:** **C2A** only (semi-synthetic) for the report.
- **SAHI:** done — reported as a separate, labeled inference-time ablation (not mixed into the primary mAP column).
- **Lightweight direction (NEW — feasibility below §H):** apply the winning CBAM+P2 recipe to a smaller backbone (YOLO11n/s) for edge/drone deployment.
- **Primary metric:** **F2** (SAR favors recall). Full COCO + per-size recall + curves — all present in run `summary.json`.
- **SOTA comparison:** vs **Nihal et al.** (arXiv 2408.04922) — folder `Paper-01- Ragib Amin Nihal(...)`. (2nd of 9 on C2A.)
- **Future-work / NOT reported:** SARD (all of it) + deployable joint model, genuine-novelty SSM redesign, 5-seed significance, energy/CO₂, ONNX/Jetson.
- **Negative results (mention honestly):** Mamba SSM neck (above) and AtrousMamba (`01-03-2026-Onward Model trying/`).
- **Hardware:** RTX 4070 Ti SUPER 16 GB, i7-14700K, 128 GB RAM, Win 11 (per `system_spec_thesis.md`; exact CUDA/torch/ultralytics from each run's `env.json`).

---

## B. Chapter → content → source map

| Chapter | What goes in it | Source material in the folder |
|---|---|---|
| **I Introduction** | SAR motivation, tiny-object problem, objectives, scope, single-author plan, applications, thesis org | `pivot point_31_5_26.md` (motivation/positioning); memory |
| **II Literature Review** | Thematic: classical vs deep-learning small-object detection; **Core Concepts = YOLO11 + CBAM + P2 + Mamba/SSM baselines** (explained HERE, not in Ch III); research-gap table | `Paper-01 Nihal` + refs in `Mofa_thesis_13.4.26.txt`; `01-03-2026-Onward Model trying/mamba-notes` (Mamba) **[verify refs]** |
| **III Methodology** | **Overall framework diagram FIRST**, then: preprocessing, the additive architecture (YOUR modifications only), post-init Mamba injection, running example on one real image | `system_spec_thesis.md` §2–§3; run `architecture/` outputs; code in `Last Month/24_01_26- Benchmarking YOLOs/` |
| **IV Implementation, Results & Discussion** | Exp setup (HW/SW table), problem-specific metrics, dataset (samples+splits+aug), quantitative (tables+curves), qualitative (failure analysis), **Comparison with SOTA (Nihal)**, cross-dataset C2A→SARD, objectives achieved | `runs/<id>/metrics/*.csv`, `plots/*.png`, `ablation_master/`, `common/splits/`; SARD in `common/sard/` **[verify these exist / A6000 vs 4070 run]** |
| **V Societal/Legal/Ethical…** | IP, ethics (SAR privacy/drones), safety, legal (aviation/drone law BD), societal/health/cultural, environment/sustainability | write from principles; cite where possible |
| **VI Complex Engineering** | Map to Washington-Accord attributes (problems + activities) | dept rubric **[verify format]** |
| **VII Conclusions** | 1-para summary, limitations (semi-synthetic C2A, SSM very-tiny trade-off, single-GPU), future work (deployable A6000, SSM redesign, more datasets) | `pivot` §4/§7/§8 |
| **References** | IEEE, DOI, peer-reviewed majority | `Mofa_thesis_13.4.26.txt` bib + Paper folders |

---

## C. Figures & diagrams needed

### C1. Conceptual diagrams — I author these as **draw.io XML** (editable; you regenerate in PPT/Word)
Supervisor rule: diagrams must stay modifiable. I deliver `.drawio.xml`; you open in draw.io / PowerPoint and re-style.

| # | Figure | Chapter |
|---|---|---|
| 1 | **Overall framework / pipeline** (drone image → preprocess → backbone → CBAM → P2 head → Mamba neck → detections) — methodology anchor | III |
| 2 | YOLO11m baseline architecture (backbone/neck/head) | II |
| 3 | CBAM module (channel + spatial attention) | II |
| 4 | P2 detection head (4-scale, where P2 attaches, stride-4) | II/III |
| 5 | C3k2 → **C3k2Mamba** block + selective scan (forward/backward) | II/III |
| 6 | **Additive ablation chain** (baseline → +CBAM → +P2 → +both → +Mamba) | III |
| 7 | Post-init Mamba injection schematic (which neck layers, idx≥11, c_out≤512) | III |
| 8 | SAHI sliced-inference tiling | IV (SAHI ablation) |
| 9 | (opt.) Data-augmentation pipeline | IV |
| 10 | Gantt chart (single-author plan) | I |
| 11 | (if lightweight in scope) lightweight backbone + P2 schematic | III/IV |

### C2. Data plots — regenerated from run CSVs (matplotlib; script = the "editable source")
Most already exist as PNGs under `runs/<id>/plots/` (per `PLOTS_INDEX.md`). For the report I re-plot from CSV at DPI≥300 with consistent per-model colors.

- PR-curve overlay (5 models) · F1-vs-confidence · confidence histogram · confusion matrix · calibration/reliability
- Metric bar charts (P/R/F1/F2/mAP per model) · **params-vs-mAP Pareto** · **improvement waterfall** (baseline→+CBAM→+P2→+Mamba)
- per-size recall bins (very-tiny→large) · training curves (loss/mAP vs epoch) · latency/FPS bars
- SAHI vs no-SAHI comparison (separate, labeled)

### C3. Qualitative image grids — from run outputs / inference
- Detection grid · **failure-case grid (16 worst-FN) + failure taxonomy** · success grid
- CBAM attention-map overlays · P2 per-stride detections (which head catches <8 px)

### C4. Dataset sample grids — must **cover all cases** (supervisor rule)
- C2A samples (all scenario/scale/lighting cases) · SARD samples · before/after augmentation

---

## D. Tables needed (IEEE style, caption above)

1. Hardware/software/platform spec
2. Hyperparameters (imgsz, batch, epochs, patience, optimizer, LR schedule…)
3. Dataset split (train/val/test counts, C2A + SARD) + before/after augmentation
4. **Main comparison** (4 models × {P,R,F1,F2,mAP50,mAP50-95,AP_s/m/l,AR}) — no-TTA/no-SAHI
5. Efficiency (params, GFLOPs, weights size, latency, FPS)
6. Per-size recall table
7. Ablation contribution deltas (what each module adds)
8. **Comparison with SOTA** (vs Nihal et al. + others on C2A)
9. SAHI inference-ablation (separate, labeled)
10. (if in scope) Lightweight vs full CBAM+P2 (params/latency/accuracy trade-off)

---

## E. Step-by-step plan (who does what)

### Phase 0 — Setup (now)
- **[ME]** Finalize skeleton + `thesisstyle.sty` (done) → verify it compiles (Overleaf, see §F).
- **[YOU]** Decide title; drop `kuet_logo.png` into `figures/`; create Overleaf project (§F).
- **[ME]** Build helper docs: this plan, a **figure list**, a **references list** from `Mofa_thesis_13.4.26.txt` + paper folders.

### Phase 1 — Lock content sources
- **[ME]** Read `Mofa_thesis_13.4.26.txt` (old draft) for reusable prose + bib; map every result/number to a run folder; list exactly which CSV/PNG feeds which figure/table. **Verify, not guess.**
- **[YOU]** Confirm **which run set is canonical for the report** (4070 Ti results, not A6000) and point me to the `ablation_master/` / `runs/` paths that hold the final numbers. **[verify]**

### Phase 2 — Diagrams
- **[ME]** Produce draw.io XML for figures C1 #1–#7 first (architecture/methodology core).
- **[YOU]** Open each in draw.io/PowerPoint, restyle to taste, export PNG/PDF into `figures/`.
- **[ME]** Write plot-regeneration scripts (or reuse existing PNGs) for C2/C3.

### Phase 3 — Write chapters (dependency order)
III Methodology → IV Results → II Lit Review → I Intro → VII Conclusions → V → VI. (Technical core first; it anchors everything.)

### Phase 4 — Polish
Cross-refs, caption pass, symbol-consistency pass, IEEE reference pass with DOIs, gap/spacing pass, final compile.

---

## F. Overleaf recommendation — **YES, move to Overleaf (primary), keep local WSL as replica**

Reasons: (1) local WSL can't reliably stream compile output to me here; (2) it's a formatting-critical doc and Overleaf's toolchain is deterministic; (3) real Times New Roman travels with the project (bundled `.ttf`, no install). Plan:
- **You:** create an Overleaf project "Defense" and upload the `draft1_30_6_26/` contents (I'll give an exact file list + a `latexmkrc`).
- Set Overleaf compiler = **XeLaTeX**.
- Local WSL = exact replica for when Overleaf free-tier hits its compile-time limit. I'll add a `build.sh` (`xelatex → biber → xelatex → xelatex`).

---

## G. Open decisions I need from you
1. Final **title** (candidates proposed in chat).
2. **Lightweight model:** report it (needs runs before July 8) or frame as future work? (feasibility §H)
3. Confirm the **Nihal SOTA comparison** stays (C2A leaderboard, 2nd of 9).

## H. Lightweight model — feasibility (answer to your Q2)
**Yes, feasible and well-motivated** (drones need edge-real-time; the 11m CBAM+P2 is 14.6 ms on a 4070 but heavier on a Jetson).
- **Recipe:** take the winning **CBAM+P2** design, drop it onto a smaller backbone — **YOLO11n (~2.6 M)** or **YOLO11s (~9.4 M)** — same P2 insertion + CBAM swap, same C2A pipeline/metrics. Architecturally trivial (change the YAML backbone, reuse the build code).
- **Cost:** training only. 11m baseline took 7.3 h; 11n/s train faster → 1 seed each is affordable before July 8 **if started soon**.
- **Status:** **no lightweight (11n/s) run exists yet** — verified. The A6000 `deployable_model/` scripts use CBAM+P2 on **11m** (+ SARD), which is the *future* deployable model, not a lightweight backbone.
- **Framing options:** (a) if runs land → a "lightweight variant for edge deployment" subsection in Ch IV with a params/latency/accuracy trade-off table + Pareto; (b) if not → strong Ch VII future-work with the recipe + expected trade-off.
- **My recommendation:** decide now — if you want it *reported*, kick off `yolo11s + CBAM + P2` (1 seed) today; 11s is the safer accuracy/size balance for a first lightweight point.

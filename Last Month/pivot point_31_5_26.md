# Pivot Point — 2026-05-31

> Strategic decision record for the final month of the thesis. Captures the
> brainstorm/analysis from the 2026-05-31 session, the decisions taken, and the
> concrete plan that follows from them. This is the "why"; `system_spec_thesis.md`
> is the "how".

---

## 0. TL;DR (the decisions)

1. **Two deliverables, two timelines.** Thesis *report* (≈1 month, course requirement) ≠ journal *paper* (after the course). Stop applying the paper-grade bar to the thesis-month timeline. The full `system_spec.md` is a **journal-grade** protocol; the thesis uses a **trimmed subset** (`system_spec_thesis.md`).
2. **Final/headline model for the thesis = `Mamba + CBAM + P2Head`.** It is the best on the headline accuracy metrics (mAP50, mAP50-95, precision) and keeps the SSM novelty angle the thesis is built around. Its honest ablation is a *complete scientific story* — see §3.
3. **Mamba is NOT a strong novelty driver as-is** (P2 head is the real driver; Mamba is ~parameter-free and slightly hurts very-tiny recall). For the *thesis* this is fine and is reported honestly. For the *paper*, the SSM must be redesigned so it actually serves small objects — see §4.
4. **Retrain the 4-model additive ablation chain at full epochs**, one after another (user decision), then **validate cross-dataset on a real SAR dataset (SARD primary, VisDrone-person secondary).** This is what turns a single-dataset study into a defensible thesis. See §5–§6.
5. **Seed policy: 1 seed for the ablation chain, 3 seeds for the final headline model.** 5 seeds × 4 models is computationally impossible in a month (see §2). Significance testing is deferred to the paper.
6. **Writing is the critical path, not compute.** Start writing now on existing + incoming results; do not block the report on the full re-run.

---

## 1. The core reframe

| | Thesis **report** | Journal **paper** |
|---|---|---|
| Deadline | ~1 month (hard, course) | none (after course) |
| Judged by | committee | anonymous Q1/Q2 reviewers |
| Rigor bar | solid, defensible, complete story | multi-seed, significance, multi-dataset |
| Critical path | **writing** | **novelty + generalization** |

Almost every difficulty this month traces back to conflating these two. The thesis needs a *complete, honest, well-presented* study. The paper needs a *novel, general, statistically-backed* contribution. They share data and code but not scope.

---

## 2. The feasibility math (why the plan is scoped the way it is)

Per `docs/2026-05-29_yolo11m_final_month_writeup.md`, a single full run on the 4070 Ti SUPER is **~30–50 h**.

- 1 model × 5 seeds = **150–250 h = 6–10 GPU-days.**
- 4 models × 5 seeds = **~25–40 GPU-days** — and that's before cross-dataset eval and before writing.

You have ~30 days *and must write the report.* Therefore:

- **Ablation chain (baseline, +P2, +CBAM+P2): 1 seed each.**
- **Headline (Mamba+CBAM+P2): 3 seeds** (gives one honest mean±std for the headline row; full 5-seed + significance is a paper task).
- Budget: 4 single-seed runs (~160 h) + 2 extra headline seeds (~80 h) ≈ **~10 GPU-days of training**, leaving ~20 days for cross-dataset eval + writing. Feasible.

---

## 3. Final model decision — `Mamba + CBAM + P2Head`

**Why this is the right thesis headline despite the novelty caveat:**

- Best **mAP50, mAP50-95, precision** of all variants tried.
- Same parameter count as CBAM+P2 (Mamba injected post-init) — "accuracy gain at zero parameter cost" is a legitimate, citable framing.
- Keeps the SSM angle, which is what distinguishes this thesis from "just another YOLO+attention paper" — and respects the standing preference for **genuine novelty, not plug-and-play**.
- The **additive ablation** (baseline → +P2 → +CBAM+P2 → +Mamba+CBAM+P2) is exactly the structure reviewers/examiners want: one component added at a time, each contribution isolated.

**The honest story to tell (do not hide this — owning it is a strength):**

> "A systematic component study for tiny-human detection in aerial disaster imagery. The **P2 head** is the dominant contributor (high-resolution stride-4 features recover sub-8px humans). **CBAM** adds operational gains (recall, F2, lower latency). The **SSM neck (Mamba)** improves localization (mAP50-95) and precision at *no parameter cost*, with a measured trade-off: a small reduction in very-tiny recall due to sequential-scan feature dilution. Inference-time **SAHI/TTA** further lift very-tiny recall without retraining."

This is a full, defensible thesis. It does **not** overclaim Mamba.

**Why NOT make CBAM+P2 the final model instead:** it is faster and slightly better at very-tiny recall, but it abandons the novelty entirely and reduces the thesis to two off-the-shelf modules — exactly the plug-and-play outcome to avoid. Keep CBAM+P2 as the key ablation row, not the headline.

---

## 4. The novelty problem and the paper-phase fix (post-course)

**The problem a reviewer will raise immediately:** your *novel* component (Mamba neck) is not the driver — a *known* technique (P2) is — and the Mamba block is ~parameter-free and slightly *hurts* very-tiny recall, which is the paper's whole mission. As-is, this is a **Q2/Q3 / good-conference** contribution, not Q1.

**The paper-phase upgrade (architectural advancement) — make the SSM the driver of small-object gains:**

- **Root cause** (from your own findings): the current local-window bidirectional SSM scans sequentially and *dilutes high-frequency small-object features*. So it helps medium objects and localization but not tiny ones.
- **Direction:** redesign the neck SSM as a **cross-scale, small-object-preserving fusion** — e.g. an SSM that takes the high-resolution P2 stream as the *query/anchor* and uses state-space scanning to inject long-range context from P3–P5 *into* P2 without down-mixing P2's detail. The novelty is in the **information-flow pattern** (scale-anchored SSM fusion), not a module swap. If this lifts very-tiny recall *above* P2-only, then the SSM is both novel **and** the driver — that fixes the whole critique.
- This is **out of scope for the thesis month** and is the first paper-phase task.

**Alternative paper framing (lower risk):** reframe the paper as a *systematic study + inference pipeline* for tiny-human SAR detection (P2 + CBAM + SAHI/TTA), with the SSM as a secondary localization-precision finding. Honest, publishable in a solid Q2, but less "novel architecture."

---

## 5. Weak points / loopholes to fix or own

| # | Weakness | Thesis action | Paper action |
|---|---|---|---|
| 1 | **Novelty attribution** — P2, not Mamba, is the driver | Report honestly (§3) | Redesign SSM to drive small-object gains (§4) |
| 2 | **Single, semi-synthetic dataset** — C2A is composited (LSP/MPII poses on AIDER scenes via U²-Net) | **Cross-dataset test on real SARD** (§6) | Add ≥1 more real dataset + train/finetune comparison |
| 3 | **Mixed eval protocols** — headline mAP from TTA, SAHI uses a different per-image protocol | Primary table = no-TTA/no-SAHI, apples-to-apples; TTA & SAHI as *separate* labeled ablations | same, with explicit protocol footnotes |
| 4 | **Unfair baseline** — our TTA model vs Nihal et al. non-TTA | Primary comparison = our *no-TTA* model vs their numbers; TTA shown separately | same |
| 5 | **Hardware-inconsistent latency** — old numbers on T4×2 | Re-measure all latency on the 4070 Ti SUPER (inference pass) | same |
| 6 | **No variance/significance** | 3 seeds on headline → mean±std only | Full 5-seed + paired bootstrap / McNemar |

---

## 6. Cross-dataset (real) generalization plan — the key thesis upgrade

**Primary: SARD (Search And Rescue Dataset).** Real UAV RGB, ~1,981 annotated frames, single class *person* (with pose sub-labels we collapse to `person`), non-urban SAR terrain. Available on Kaggle (`nikolasgegenava/sard-search-and-rescue`) and Roboflow in YOLO format. Directly comparable to C2A's single-class person task — ideal zero-shot transfer target.

**Secondary (optional, if time): VisDrone-person.** Real urban drone imagery; map `pedestrian`+`people` → `person`, ignore other classes. Larger and harder; good stress test.

**Skip for thesis: HIT-UAV (thermal)** — modality gap too large for an RGB-trained model; mention as future work only.

**Protocol (cheap — inference only, no retraining for the headline finding):**
1. **Zero-shot transfer:** take each C2A-trained model's `best.pt`, evaluate on SARD test split with the *same* eval pipeline and metrics. This is the headline generalization result — "does a C2A-trained detector work on real SAR imagery?"
2. Report the **C2A→SARD drop** per model; the model with the smallest drop generalizes best (often not the highest-C2A-mAP model — an interesting finding either way).
3. **Optional adaptability check (if time):** fine-tune the headline model on a small SARD train split and re-evaluate, to show the architecture *adapts* to real data. Cheap-ish (small dataset).
4. Handle the **class-mapping** explicitly (SARD pose labels → single `person`; VisDrone person-like classes → `person`); document it.

This single addition is what most strengthens the thesis: it converts "good numbers on one semi-synthetic dataset" into "validated on real search-and-rescue imagery."

---

## 7. `system_spec.md` — what's required vs deferred for the thesis

The big spec is correct for the *paper*. For the *thesis month*:

**KEEP (cheap, high-value):** consistent metric set across models; COCO AP_small/medium/large + per-size recall (the centerpiece); PR curves, confusion matrices, calibration, confidence histograms; architecture-specific metrics (attention maps, SSM scan stats — they *tell the story*); failure-case grid; basic reproducibility (seed, `env.json`, frozen-split md5); latency on the 4070; smoke/OOM/checkpoint/dynamic-epoch infra (cheap insurance against load-shedding + OOM).

**DEFER to the paper:** 5 seeds for every model (thesis: 1 / headline 3); paired bootstrap / McNemar / Holm-Bonferroni significance; CodeCarbon energy/CO₂; model cards & dataset datasheets; ONNX export; multi-resolution FPS sweep + 4-batch throughput (trim to one number).

---

## 8. Realistic journal tier

- **As-is:** Q2/Q3 or a good conference/workshop (killed at Q1 by weaknesses #1 and #2).
- **Q1 (TGRS / ISPRS / Pattern Recognition):** needs the SSM-as-driver redesign **+** multi-dataset generalization **+** full rigor → 2–4 month post-course project.
- **Realistic "very good" target:** *Remote Sensing* (MDPI), *Drones* (MDPI), IEEE *JSTARS*, *Neurocomputing* — reachable with C2A + 1–2 real datasets + honest ablation + rigor.

---

## 9. The month, week by week

- **Week 1:** re-scope seeds (5→1, headline 3); launch the ablation chain (§5 of `system_spec_thesis.md`); begin writing Intro + Related Work + Methods (no final numbers needed). Acquire & convert SARD to YOLO format.
- **Weeks 2–3:** finish training runs as they complete; run cross-dataset eval; build tables/figures; write Experiments + Results; build failure-case grid + per-size analysis.
- **Week 4:** write Limitations honestly (semi-synthetic data, single primary dataset, Mamba's modest role); polish; defense prep; buffer for load-shedding/re-runs.
- **After the course → paper:** SSM-as-driver redesign + more datasets + full `system_spec.md` rigor.

---

## 10. Questions asked across sessions (throughline)

1. "No plug-and-play" — want genuine architectural novelty (rejected pose multi-task head).
2. "Review/verify feasibility before suggesting" — no ideas that won't run on the compute.
3. AtrousMamba direction (Apr 11) — pivot to SAHI/TTA + neck-level after AtrousSSM failed.
4. Patience values (May 29) — raised 30/20 → 50/40 with literature backing.
5. This session — compile spec → add full metric/significance protocol → strip result numbers → "is re-running the right use of my month?" → "retrain the 4-model chain + cross-dataset test; what's the final model; write the thesis spec."

Pattern: consistent push toward novelty + rigor, repeatedly constrained by compute + time — now at its tightest. This pivot resolves it by splitting thesis-scope from paper-scope.

---

*Sources consulted: MDPI small-object survey (2025); arXiv small-object survey 2503.20516; Tandfonline multi-scale RS small-object (2025) on multi-dataset expectations; SARD dataset (Journal of Remote Sensing / Kaggle / IEEE DataPort); thesis-vs-paper scope (AIJR, APA).*

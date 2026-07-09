# 2026-07-09 — Defense presentation slide plan (MSA-YOLO, CBAM+P2)

**Source of truth:** `Defense/draft1_30_6_26/` (report draft, read in full 2026-07-09).
**All image paths are relative to** `Defense/draft1_30_6_26/figures/` unless stated.
**Model branding:** call the recommended model **MSA-YOLO (YOLO11m + CBAM + P2)** everywhere.

**How to read a slide entry** — every element line ends with its design spec in `[...]`:
`[20 · bold · #1F4E79]` = size pt · weight · color. Bold spans inside bullet text are
marked with `**…**` — only those words are bold, the rest regular. Speaker notes go in
PowerPoint's notes pane, never on the slide.

---

## ONE-TIME SLIDE MASTER SETUP (do this before slide 1, then never think about it again)

Canvas: **16:9** (same as pre-defense). Design → Slide Size → Widescreen.

1. **Theme fonts** (Design → Variants → Fonts → Customize): Headings = **Open Sans Bold**
   (or Open Sauce Bold), Body = **Open Sauce** / Open Sans. Fallback for both: Calibri.
2. **Theme colors** (Design → Variants → Colors → Customize) — name them so they appear in
   every color picker:
   - `TextDark` #1F2933 (body text — never pure black)
   - `Navy` #1F4E79 (titles, table headers, structure)
   - `DiagBlue` fill #DAE8FC / stroke #6C8EBF (baseline diagram blocks — matches report figures)
   - `Green` #82B366 / fill #D5E8D4 (gains, "keep", Find:)
   - `Red` #C0392B / stroke #B85450 / fill #F8CECC (**reserved: OUR modifications + warnings only**)
   - `Grey` #9E9E9E (de-emphasized / not-adopted), caption grey #595959, footer grey #7F7F7F
3. **Layouts to create in Slide Master** (View → Slide Master), fixed positions so nothing
   jumps between slides:
   - Title box on ALL content layouts: x 0.5″, y 0.4″, w 12.3″, h 0.8″ — text 30 pt bold Navy, left.
   - Kicker box above title: x 0.5″, y 0.12″ — 16 pt bold ALL CAPS Red.
   - Content area: y 1.35″ → 6.9″. Footer: y 7.1″, 11 pt #7F7F7F:
     `MSA-YOLO — Md Mofazzal Hosen (2007074)` left + slide number right (not on title/dividers/thank-you).
   - **T2 Divider layout:** full-width Navy band across the vertical middle, 44 pt bold white text.
   - **T6 Architecture layout** (slides M1–M10): pipeline diagram pasted at IDENTICAL
     coordinates every step slide; current stage full color + 2.5 pt Red outline; all other
     stages 40 % transparent greyscale. Advancing slides = the animation.
4. **Reusable boxes** (make once, copy-paste):
   - *Equation box:* white fill, 1 pt #6C8EBF border, rounded 4 pt, 8 pt padding; equation
     rendered ≈ 24–28 pt (PowerPoint equation editor, or crop from the compiled report PDF —
     pick ONE method deck-wide).
   - *Takeaway box:* 10 % Navy tint fill, text 20 pt bold italic #1F2933, bottom of slide.
   - *B/I/F strip:* three lines, lead words 20 pt bold — `Break:` Red, `Integrate:` Navy,
     `Find:` Green — rest of line 20 pt regular TextDark.
5. **Table style** (format the first table, then Format Painter the rest): no vertical
   rules; horizontal rules 1.5 pt top & bottom, 0.75 pt under header; header row Navy fill,
   white 16 pt bold; cells 14–16 pt, numbers right-aligned; recommended-model row = 10 %
   Navy tint + bold; caption ABOVE the table, 14 pt bold, Small Caps (Font dialog → Small
   Caps), centered. Decimals: 3 for metrics (0.853), 1 for ms/points (14.6 ms, +1.5 pt).
6. Embed fonts on save (File → Options → Save → Embed fonts) — the defense PC won't have Open Sauce.

### Export checklist (3 figures have no PNG yet)
| Figure | Source | Action |
|---|---|---|
| C2A size distribution | `fig_size_distribution.drawio` | draw.io → export PNG 300 dpi (or crop report PDF) |
| Ablation waterfall | `fig_ablation_waterfall.drawio` | draw.io → export PNG |
| Gantt timeline | `tikz/fig_gantt.tex` only | crop from compiled report PDF (Ch. I) |

### Continuity with the pre-defense deck (34 slides, HBPA-YOLO proposal — read 2026-07-09)
Port verbatim: title-slide layout (Presented By | Supervised By | `CSE 4000: Project/Thesis`),
outline with per-section page numbers (fill LAST), full-slide section dividers,
`Fig. - 0X :` captions below figures, `TABLE X` captions above tables, "(Cont'd…)" on
continuation slides, IEEE numbered references (keep [1] Nihal, [3] TPH-YOLOv5, [4] AGIIndia
stable). **The committee saw HBPA-YOLO (body-part attention)** — the pivot to MSA-YOLO is
addressed by the Bridge slide (after Objectives) and hidden slide H10.

---

# MAIN DECK — 36 slides + 5 dividers + 10 hidden

**[mergeable]** slides collapse the deck to ~24 if the committee caps you at ~20 min.
Figure numbers `Fig. - 01…25` and table numbers `TABLE I…X` are already assigned inline
below — renumber only if you merge/reorder slides.

---

## PART A — OPENING

### Slide 1 · Title slide (no title bar, no footer)
- KUET logo `kuet_logo.png` top-center, 1.0″ tall.
- Thesis title, centered upper third, ≤ 3 lines `[30–32 · bold · #1F4E79]`:
  *MSA-YOLO: A Multi-Scale Attention Enhancement of YOLO for Tiny-Human Detection in Aerial Search-and-Rescue Imagery*
- Two-column block, bottom half:
  - Left: `Presented By` `[18 · bold]` → `Md Mofazzal Hosen` / `Roll: 2007074` `[20 · regular · #1F2933]`
  - Right: `Supervised By` `[18 · bold]` → `Dr. Sk. Md. Masudul Ahsan` / `Professor` / `Department of Computer Science and Engineering` / `Khulna University of Engineering & Technology` `[20 · regular]`
- Bottom line: `CSE 4000: Project/Thesis` `[16 · regular · #595959]`
- Note: one-breath version to memorize: "a compact detector for people only a few pixels wide in disaster drone imagery, built by measuring one change at a time."

### Slide 2 · "Outline"
- Title `Outline` `[30 · bold · #1F4E79]`.
- Numbered list, two columns if needed `[22 · regular · #1F2933, 1.15 line spacing]`:
  Introduction → Applications → Related Works → Problem Statement → Objectives →
  Methodology → Implementation & Results → Thesis Timeline → Future Work → Conclusion → References.
- Page number per section, right-aligned `[22 · regular · #7F7F7F]` — **fill in LAST**.
- Optional: tiny pipeline-diagram thumbnails beside "Methodology" to preview the step-by-step format.

---

## DIVIDER — `Introduction` [44 · bold · white on Navy band]

### Slide 3 · "Introduction" (motivation)
- Title `Introduction` `[30 · bold · #1F4E79]`.
- Image right 40 %: `fig_intro_scene.png`, 1 pt #BFBFBF border.
  Caption below `[14 · regular · #595959, centered]`: `Fig. - 01 : Aerial drone footage of a flood-affected region [4]`
- Bullets left 55 % `[20–22 · regular · #1F2933]`, bold spans as marked:
  - Earthquake / flood / fire → **UAVs survey the site far faster** than ground teams, with no risk to rescuers.
  - But: a person seen from altitude ≈ **fewer than 10 pixels**, amid rubble, water, smoke.
  - One sortie → **thousands of frames**; an operator scanning that stream *will* miss people.
  - Automating the search is a **practical need** — this thesis addresses it.
- Note: this is the emotional hook; speak the last bullet slowly. The catchy line "the first hours decide who is found alive" is your SPOKEN opening — not on the slide.

### Slide 4 · "Introduction (Cont'd…)" (why it's hard)
- Title `Introduction (Cont'd…)` `[30 · bold · #1F4E79]`.
- Images: 2×2 grid, all four same rendered height (~2.2″), 1 pt #BFBFBF borders:
  `c2a_collapsed.png` · `c2a_fire.png` · `c2a_flood.png` · `c2a_traffic.png`
  Panel labels under each `[14 · bold]`: (a) collapsed building (b) fire (c) flood (d) traffic incident.
  Caption `[14 · regular · #595959]`: `Fig. - 02 : Representative C2A scenes — human targets are tiny and numerous in every scene type [1]`
- Facts strip below (3 lines) `[18–20 · regular]`:
  - ~**47 %** of annotated people in C2A are **< 10 px** — at or below a standard detector's resolving limit.
  - **20–40 people per scene**, many partially buried or occluded.
  - Detection maturity (R-CNN → YOLO → DETR) does **not transfer**: targets are an order of magnitude smaller than these detectors were tuned for.

### Slide 5 · "Applications" **[mergeable → keep items 1, 2, 5 on Slide 3]**
- Title `Applications` `[30 · bold · #1F4E79]`.
- Five numbered items, one small icon each `[20 · regular; lead phrase bold]`:
  1. **Search-and-rescue triage** — drone/laptop flags likely victims live; operators review candidates, not raw frames.
  2. **Disaster damage assessment** — detections aggregated over a flight show planners where people are concentrated.
  3. **Crowd monitoring** — counting tiny people from altitude is the same technical problem.
  4. **Wide-area security** — border/coastline patrol = low-density variant of the task.
  5. **Embedded deployment** — 19.6 M params fits edge GPUs; recipe transfers to smaller YOLO11 backbones.
- Optional corner image: `arch_detections.png` small, caption `[14 · #595959]`: "what the operator sees".

---

## DIVIDER — `Related Works` [44 · bold · white on Navy band]

### Slide 6 · "Related Works" (detector families)
- Title `Related Works` `[30 · bold · #1F4E79]`.
- Caption above table `[14 · bold · Small Caps, centered]`: `TABLE I — DETECTOR FAMILIES AND THEIR TRADE-OFFS`
- Table (header: Navy fill, white 16 bold; cells 14–16):

| Family | Representative works | Key idea | Trade-off for our task |
|---|---|---|---|
| Two-stage | R-CNN (2014), Faster R-CNN (2015), Cascade R-CNN (2018) | propose regions, then classify | accurate but too slow/heavy for airborne use |
| One-stage | SSD (2016), RetinaNet + focal loss (2017), YOLO v1→v11 (2016–2024) | predict boxes in one pass | real-time, compact — **our family** (bold + Navy-tint row) |
| Transformer | DETR (2020), RT-DETR (2024) | set prediction, no NMS/anchors | competitive speed only recently; heavier |

- Takeaway box `[20 · bold italic on 10 % Navy tint]`: *We work in the YOLO family because speed and model size decide airborne deployability.*

### Slide 7 · "Related Works (Cont'd…)" (small-object & aerial methods)
- Title `Related Works (Cont'd…)` `[30 · bold · #1F4E79]`.
- Caption `[14 · bold · Small Caps]`: `TABLE II — TECHNIQUES FOR SMALL/AERIAL TARGETS`
- Table (same style as TABLE I):

| Approach | Representative works | How it helps tiny targets | Limitation in prior work |
|---|---|---|---|
| Higher-resolution scales | FPN (2017), TPH-YOLOv5 (2021), P2/stride-4 heads | detection grid fine enough for few-px targets | usually reported alone, on generic aerial data |
| Feature attention | SE (2018), ECA (2020), CBAM (2018) | re-weights features so faint targets survive clutter | choice asserted, rarely measured on-task |
| Inference-time slicing | SAHI (2022) | overlapping crops enlarge small targets | latency cost; no architecture change |
| Benchmarks | VisDrone (2021) | makes small objects the dominant difficulty | generic scenes, not disaster imagery |

- Takeaway box `[20 · bold italic]`: *All three technique lines are relevant — nobody had compared them under ONE protocol on disaster imagery.*

### Slide 8 · "Related Works (Cont'd…)" (SAR datasets & state-space models)
- Title `Related Works (Cont'd…)` `[30 · bold · #1F4E79]`.
- Caption `[14 · bold · Small Caps]`: `TABLE III — SAR HUMAN DETECTION AND STATE-SPACE MODELS`
- Table:

| Theme | Representative works | Status | Gap |
|---|---|---|---|
| Real SAR datasets | SARD (2021), HERIDAL (2019) | valuable, real imagery | small (~2k frames) — too little to train a 20 M-param detector |
| Synthetic-composite benchmark | **C2A** — Nihal et al. (2024): AIDER backgrounds + U²-Net-segmented humans | 10,215 images, ~360k instances, 4 scene types | whole models vs whole models — no component attribution |
| State-space models (SSM) | S4 (2022) → Mamba (2023) → Vision Mamba / VMamba (2024) → Mamba-YOLO (2025) | linear-time alternative to attention | applied as backbone replacement; never a controlled test on tiny humans |

- Takeaway box `[20 · bold italic]`: *C2A is our benchmark and Nihal et al. our direct comparison; the SSM claim is untested at this scale — we test it.*

### Slide 9 · "Research Gap"
- Title `Research Gap` `[30 · bold · #1F4E79]`.
- Caption `[14 · bold · Small Caps]`: `TABLE IV — POSITION OF THIS THESIS RELATIVE TO PRIOR WORK` (verbatim report Tab. 2.1)
- Table — make the right column visually dominant (right column text bold where marked):

| Aspect | Prior work | This thesis |
|---|---|---|
| Small-object techniques | reported individually | compared in **one controlled ablation** |
| Attention choice | asserted or single option | CBAM vs ECA **measured on the task** |
| State-space neck | proposed as a backbone, generic data | tested **in the neck** vs identical baseline; reported honestly |
| Target domain | mostly generic aerial imagery | disaster/SAR imagery (C2A), very small humans |

- Note: read the right column aloud — this slide IS the transition into objectives.

---

### Slide 10 · "Problem Statement"
- Title `Problem Statement` `[30 · bold · #1F4E79]`.
- Image right 40 %: `qual_gt.png`, 1 pt border.
  Caption `[14 · #595959]`: `Fig. - 03 : One C2A test scene — 156 annotated people`
- Bullets left `[20 · regular]`:
  - Given aerial image *I* → output boxes (x, y, w, h, c), **one per visible person**.
  - Correct when IoU ≥ 0.5 (equation in box below).
  - **Domain asymmetry:** a missed person may never get a second chance; a false alarm costs seconds of review.
- Equation box #1 [white fill, 1 pt #6C8EBF border, eq ≈ 24 pt]:
  IoU(B̂,B) = area(B̂∩B) / area(B̂∪B) ≥ 0.5
- Equation box #2, highlighted (2 pt Navy border) [eq ≈ 28 pt]:
  **F₂ = 5PR / (4P + R)** — label under it `[16 · regular]`: recall weighted 4× over precision — our operational metric.
- Note: close with "which architectural changes raise recall on sub-ten-pixel people, within an airborne parameter/latency budget?"

### Slide 11 · "Objectives"
- Title `Objectives` `[30 · bold · #1F4E79]`.
- Image right 35 %: `fig_ablation_chain.png` (borderless — schematic).
  Caption `[14 · #595959]`: `Fig. - 04 : The additive ablation chain — one change per configuration`
- Objectives, numbered `[20 · regular; bold spans as marked]`:
  1. Build a compact detector via an **additive ablation** — CBAM, P2 scale, SSM neck introduced **one at a time** under one frozen protocol.
  2. Identify the configuration best serving **very-tiny recall** within a deployable budget; show it performs near published SOTA.
  3. **Honestly test** the claim that state-space modules help detection — report the outcome either way.
  4. Raise recall further at inference time **without retraining** (SAHI, TTA).
- Novelty strip, bottom — 3 chips (rounded rectangles, fill #F8CECC, 1 pt #B85450 border, text `[16 · regular; lead word bold]`):
  - **CBAM as substitution** — replaces C2PSA ⇒ *negative* parameter cost.
  - **P2 quantified, not assumed** — additive; P3–P5 untouched ⇒ gain attributable.
  - **Single-variable ablation on C2A** — first of its kind; negative results are first-class.

### Slide 12 · "Hypothesis"
- Title `Hypothesis` `[30 · bold · #1F4E79]`.
- Image left 45 %: `fig_stride_problem.png` (borderless).
  Caption `[14 · #595959]`: `Fig. - 05 : The same ~12-px human under each scale's detection grid`
- Image right 45 %: size-distribution PNG (**export needed**).
  Caption `[14 · #595959]`: `Fig. - 06 : C2A test-set size distribution — 99.6 % of instances < 32 px`
- Equation box, centered between/below [eq ≈ 24 pt]: c_d(s) = s / d
  with the worked numbers beside it `[18 · regular]`: s ≈ 12 px → **0.4** cell @P5 · 0.75 @P4 · 1.5 @P3 · **3 cells @P2 (stride 4)**
- Hypothesis statement in takeaway box `[20 · bold italic on 10 % Navy tint]`:
  *If a stride-4 scale helps, the gain MUST appear in the very-tiny (< 8 px) band — nowhere else can compensate.*
- Note: this equation returns on the per-size results slide — the hypothesis is falsifiable, and Slide 33 shows the test.

### Slide 12-B · "From the Pre-Defense Proposal to MSA-YOLO" (BRIDGE — pivot insurance)
- Title `[28 · bold · #1F4E79]` (28 pt — it's long).
- Left 40 %: screenshot of pre-defense slide 18 (HBPA block diagram), greyed 60 % opacity,
  label over it `[16 · bold · #9E9E9E]`: PRE-DEFENSE PROPOSAL.
- Right 40 %: `fig_ablation_chain.png`, label `[16 · bold · #C0392B]`: FINAL DESIGN.
- Three arrow rows between them `[18 · regular; lead phrase bold]`:
  - **Body-part decomposition** → infeasible to supervise on C2A: targets < 10 px, whole-body boxes only, no part labels at this scale → replaced by a **measurement-first design**. *(⚠ verify wording against actual pre-defense feedback)*
  - **Attention for disaster clutter** → survived, matured: **CBAM as a substitution**, measured against ECA.
  - **Tiny-object focus** → survived, sharpened: **stride-4 P2 scale**, quantified in a single-variable ablation + honest SSM test.
- Takeaway box `[20 · bold italic]`: *The goal — recall on tiny, occluded humans — is unchanged; the route became controlled measurement instead of an unverifiable decomposition.*

---

## DIVIDER — `Methodology` [44 · bold · white on Navy band]

**All step slides (14–22) use the T6 layout:** same pipeline diagram, same coordinates,
current stage full color + 2.5 pt Red outline, rest 40 % greyscale. Kicker top-left
`[16 · bold · ALL CAPS · #C0392B]`. Title is always `Methodology (Cont'd…)` `[30 · bold · #1F4E79]`.
B/I/F strip bottom-right: lead words `[20 · bold]` — `Break:` #C0392B, `Integrate:` #1F4E79,
`Find:` #82B366; rest `[20 · regular · #1F2933]`. Equation box bottom-left [eq ≈ 24 pt].

### Slide 13 · "Methodology" — kicker `SYSTEM OVERVIEW` (BIG IMAGE — no equation)
- Image full width: `fig_overall_framework.png` (borderless), max 5.8″ tall.
  Caption `[14 · #595959]`: `Fig. - 07 : The proposed detection pipeline — red-dashed elements are the two proposed modifications`
- One line under caption `[20 · regular]`: *Everything not red is the YOLO11m baseline — deliberately.*
- Spoken (not written): the 3 phases — ① feature extraction + attention ② multi-scale fusion extended to stride 4 ③ four-scale decoding + NMS. Then announce: "I will now walk one image through this pipeline, stage by stage."

### Slide 14 · "Methodology (Cont'd…)" — kicker `STEP 1 / 7 · INPUT & PREPROCESSING`
- T6 diagram: INPUT block highlighted.
- Image right: `arch_input.png`, 1 pt border.
  Caption `[14 · #595959]`: `Fig. - 08 : Running example — collapsed-building scene, 30+ people, most < 16 px`
- Equation box: letterbox to 640×640 (aspect preserved) · x′ = x/255 ∈ [0,1]
- B/I/F strip:
  - **Break:** aggressive small-object augmentation (copy-paste, mixup) — *tested, discarded: negative result*.
  - **Integrate:** default YOLO augmentations; mosaic off for final 10 epochs.
  - **Find:** a stretched 8-px person is a destroyed 8-px person — letterboxing matters at this scale.

### Slide 15 · "Methodology (Cont'd…)" — kicker `STEP 2 / 7 · BACKBONE`
- T6 diagram: BACKBONE highlighted.
- Image right: crop of the backbone column from `fig_cbam_p2_architecture.png`.
  Caption `[14 · #595959]`: `Fig. - 09 : Backbone feature pyramid P1–P5 (C3k2 stages)`
- Equation box: at 640 input, level P_k = (640/2^k)² → P2 160² · P3 80² · P4 40² · P5 20²
- Bullets `[20 · regular]`:
  - C3k2 stages extract features at strides 2–32; **layers 2/4/6 kept as skip sources**.
  - Layer 2 (160²×256) becomes the **P2 skip** later.
- **Find:** `[20 · Find: in green bold]` backbone untouched — that is what keeps the ablation clean.

### Slide 16 · "Methodology (Cont'd…)" — kicker `STEP 3 / 7 · CBAM ATTENTION`
- T6 diagram: CBAM block (backbone layer 10) highlighted.
- Image right: `fig_cbam_module.png` (borderless).
  Caption `[14 · #595959]`: `Fig. - 10 : The CBAM module — channel attention, then spatial attention`
- Equation box (both, stacked, ≈ 22 pt to fit):
  M_c(F) = σ(MLP(AvgPool(F)) + MLP(MaxPool(F)))
  M_s(F′) = σ(f⁷ˣ⁷[AvgPool(F′); MaxPool(F′)])
  F′ = M_c ⊗ F,  F″ = M_s ⊗ F′
- B/I/F strip:
  - **Break:** remove the baseline's TWO heavy C2PSA self-attention blocks.
  - **Integrate:** ONE CBAM (r=16, 7×7 kernel) at layer 10 — re-weights the 512-ch P5 features.
  - **Find:** ≈ **1 M fewer parameters than baseline**, lowest latency of all four configs (**13.5 ms**), F₂ held/improved — attention at *negative* cost.

### Slide 17 · "Methodology (Cont'd…)" — kicker `CBAM IN ACTION` (BIG IMAGE — no equation)
- Two images side by side, same height:
  - `fig_cbam_effect.png` — label under it `[14 · bold]`: (a) mechanism — *illustrative* (italic!)
  - `cbam_overlay.png`, 1 pt border — label `[14 · bold]`: (b) REAL spatial attention (warm = attended)
  - Caption `[14 · #595959]`: `Fig. - 11 : Clutter suppressed, target amplified — the real attention concentrates on human-shaped regions, not rubble`
- One honesty line `[18 · regular · #595959]`: *Map is 20×20 upscaled — blocky is correct; CBAM sits at stride 32.*
- Note: say "blocky is correct" before they ask.

### Slide 18 · "Methodology (Cont'd…)" — kicker `STEP 4 / 7 · P2 DETECTION SCALE`
- T6 diagram: NECK highlighted, new stride-4 branch in Red.
- Image right: `fig_p2_head.png` (borderless).
  Caption `[14 · #595959]`: `Fig. - 12 : Construction of the P2 branch — upsample, concat backbone skip, fuse`
- Equation box: c_d(s) = s/d → 12-px person = **3 cells** at stride 4; each P2 cell covers 4×4 px → resolves targets down to ≈ 4 px
- B/I/F strip:
  - **Integrate:** top-down pass one level further — upsample fused P3 (80²×256) → concat layer-2 skip (160²×256) → C3k2 → **160²×128 P2 feature**, stride-4 branch (neck layers 17–19).
  - **Break: nothing** — P3–P5 untouched; the branch is purely additive.
  - **Find:** the principal driver: very-tiny recall **0.743 → 0.757**, AP₅₀ **0.843 → 0.853** — for ~0.5 M params, +1 ms.

### Slide 19 · "Methodology (Cont'd…)" — kicker `P2 IN ACTION` (BIG IMAGE — no equation)
- Images: `detgrid_c2a_s8.png` + `detgrid_c2a_s4.png` side by side (same crop!), 1 pt borders;
  panel labels `[14 · bold]`: (a) stride-8 (P3) grid · (b) stride-4 (P2) grid.
  Below or right: `p2_featuremap.png` — label `[14 · bold]`: (c) P2 feature response.
  Caption `[14 · #595959]`: `Fig. - 13 : Stride 8 assigns several neighbours to one cell; stride 4 separates them — the spatial mechanism behind the very-tiny-recall gain`

### Slide 20 · "Methodology (Cont'd…)" — kicker `STEP 5 / 7 · STATE-SPACE VARIANT`
- T6 diagram: six neck C3k2 blocks highlighted in **Grey** (#9E9E9E — exploration, NOT Red: red = adopted changes).
- Image right: `fig_c3k2mamba.png` (borderless).
  Caption `[14 · #595959]`: `Fig. - 14 : The explored C3k2Mamba block — forward + reverse local-window selective scans, fused`
- Equation box (label it "generic selective-SSM recurrence" `[14 · italic]`):
  h_t = Ā h_{t−1} + B̄ x_t,  y_t = C h_t — with input-dependent Ā, B̄; window 6–8, d_state = 4
- B/I/F strip:
  - **Integrate:** C3k2Mamba replaces the C3k2 bottleneck at 6 neck layers (13, 16, 19, 22, 25, 28); backbone + head untouched.
  - **Why test it** `[Navy bold]`**:** literature promotes SSMs for detection — but always as backbone swaps, never controlled.
  - **Find (preview):** +2.4 M params, **2.8× latency, no accuracy change** — a genuine, verified null (details in Results).

### Slide 21 · "Methodology (Cont'd…)" — kicker `STEP 6 / 7 · LOSS DESIGN` **[mergeable: show only CIoU + total]**
- T6 diagram: HEAD highlighted with "training signal" arrows.
- Equation box (four lines, ≈ 20–22 pt to fit; this slide is equation-led, no photo):
  L_cls = −[y log p̂ + (1−y) log(1−p̂)]   (BCE, one *person* class)
  L_CIoU = 1 − IoU + ρ²(b, b^gt)/c² + αv   (overlap + centre + aspect)
  L_DFL: box edges as distributions over binned offsets
  **L = 7.5·L_CIoU + 1.5·L_DFL + 0.5·L_cls**  (total in bold, slightly larger)
- Bullets `[20 · regular]`:
  - **Kept identical across all four configurations** → the ablation compares architectures, not objectives.
  - Heavy box weighting suits the task: with one class the problem is not *what* but **where** — IoU alone is unstable for few-px boxes under sub-pixel shifts.
  - The P2 level joins all three terms exactly like the standard levels — **no re-balancing needed**.

### Slide 22 · "Methodology (Cont'd…)" — kicker `STEP 7 / 7 · DECODING + NMS`
- T6 diagram: full pipeline lit, DETECT + NMS highlighted.
- Image right: `arch_detections.png`, 1 pt border.
  Caption `[14 · #595959]`: `Fig. - 15 : Running-example output — one box + confidence per recovered person`
- Equation box: head predicts at strides {4, 8, 16, 32} → grids 160² · 80² · 40² · 20²; NMS merges overlaps
- **Find:** `[Find: green bold]` clusters of nearby people kept as **separate detections** — what the operator reviews.

### Slide 23 · "Proposed MSA-YOLO Architecture" (BIG IMAGE — the money slide, no equation)
- Title `[28 · bold · #1F4E79]`.
- Image full slide: `fig_cbam_p2_architecture.png`, max size, borderless.
  Caption `[14 · #595959]`: `Fig. - 16 : Layer-level MSA-YOLO — CBAM at layer 10, P2 branch at layers 17–19, four-scale head at layer 29 (insets: real input / attention / P2 response / output)`
- One line `[20 · regular]`: *This exact configuration is the verbatim training configuration.*
- Note: pause here. Hidden slide H4 has the full layer table if asked.

### Slide 24 · "Methodology (Cont'd…)" — kicker `END-TO-END TRACE` **[mergeable → into Slide 23]**
- Images 2×2, same size, 1 pt borders: `arch_input.png` · `cbam_overlay.png` · `p2_featuremap.png` · `arch_detections.png`
  Panel labels `[14 · bold]`: (a) input (b) CBAM spatial attention (c) P2 stride-4 response (d) detections.
  Caption `[14 · #595959]`: `Fig. - 17 : Every intermediate read from the trained model's activations — the pipeline does what the diagrams claim`

### Slide 25 · "The Additive Ablation Design"
- Title `[30 · bold · #1F4E79]`.
- Caption `[14 · bold · Small Caps]`: `TABLE V — THE ADDITIVE ABLATION` (verbatim report Tab. 3.2)
- Table (bold marks the single change per row; CBAM+P2 row = 10 % Navy tint + bold):

| Configuration | Backbone attention | Detection scales | Neck |
|---|---|---|---|
| YOLO11m (baseline) | C2PSA | P3, P4, P5 | C3k2 |
| + CBAM | **CBAM** | P3, P4, P5 | C3k2 |
| **+ CBAM + P2** | CBAM | **P2**, P3, P4, P5 | C3k2 |
| + Mamba + CBAM + P2 | CBAM | P2, P3, P4, P5 | **C3k2Mamba (6 layers)** |

- Bullets `[18–20 · regular]`: one change per row · identical AdamW (lr₀ 0.001, cosine), ≤ 300 epochs, early stopping · same frozen split, same thresholds.
- Takeaway box `[20 · bold italic]`: *Every reported difference is attributable to the architecture alone.*
- Note: "This is why the negative Mamba result is interpretable: it fails where P2 demonstrably succeeds."

### Slide 26 · "Inference-Time Enhancement: SAHI"
- Title `[30 · bold · #1F4E79]`.
- Image top 60 %: `fig_sahi.png` (borderless).
  Real strip below it: `sahi_slice_grid.png` → `sahi_merged_detections.png` (small, 1 pt borders).
  Caption `[14 · #595959]`: `Fig. - 18 : SAHI — slice into overlapping tiles, detect per tile, merge`
- Equation box: IOS(a,b) = area(a∩b) / min(area(a), area(b)) ≥ 0.5 — greedy non-maximum merging; suits boxes cut at tile borders
- B/I/F strip:
  - **Integrate at inference only:** tiles 256–640 px, 25–30 % overlap + one full-image pass; **weights untouched**.
  - **Find (preview):** 256-px slices → very-tiny recall 0.758 → **0.788** (+3.0 pt) — but **162 ms**/image.

### Slide 27 · "Inference-Time Enhancement: TTA"
- Title `[30 · bold · #1F4E79]`.
- Image top: `fig_tta.png` (borderless); side inset: `tta_detections_zoom.png` (1 pt border).
  Caption `[14 · #595959]`: `Fig. - 19 : TTA — three scales × horizontal flip at 1280 px, merged by NMS`
- Equation box: merge over {1.0, 0.83, 0.67} × {id, hflip} at **1280 px** → NMS. Fully-convolutional ⇒ can run at 2× training resolution
- B/I/F strip:
  - **Find (preview):** very-tiny recall 0.758 → **0.850** (+9.2 pt), F₂ → **0.854**, at only **60 ms** — beats every SAHI setting. Collapses beyond 2× training resolution.

---

## DIVIDER — `Implementation & Results` [44 · bold · white on Navy band]

### Slide 28 · "Experimental Setup"
- Title `[30 · bold · #1F4E79]`. Two-column layout.
- Left column, heading `Hardware / Software` `[18 · bold · #1F4E79]`, list `[18 · regular]`:
  RTX 4070 Ti SUPER 16 GB · i7-14700K · 128 GB RAM · Windows 11 · Python 3.11.9 · PyTorch 2.12.0 (CUDA 12.6) · Ultralytics 8.4.56 · single GPU.
- Right column, caption `[14 · bold · Small Caps]`: `TABLE VI — TRAINING CONFIGURATION (IDENTICAL FOR ALL FOUR MODELS)`; mini-table `[14–16]`:
  640² letterbox · AdamW lr₀ 0.001 → cosine · ≤ 300 epochs, patience 50 (fitness) + 40 (F₂) · batch 16 (P2 models: physical 8, effective 16 grad-accum) · AMP · seed 0.
- Honesty bullet `[18 · regular]`: *AdamW pinned because the framework's default SGD diverged on the P2 architecture (~epoch 50); AdamW trained all four stably.*
- Protocol chip strip, bottom `[16 · regular · #595959]`: smoke test before every run · per-epoch safety checkpoints · OOM retry ladder · full GPU telemetry · env manifest per run ⇒ **every number traceable**.

### Slide 29 · "Dataset: C2A"
- Title `[30 · bold · #1F4E79]`.
- Images: scene grid (small, reuse — cite as Fig. - 02, do NOT renumber) + size-distribution PNG (reuse — Fig. - 06).
- Bullets `[20 · regular]`:
  - C2A = AIDER disaster backgrounds + U²-Net-segmented human poses (semi-synthetic); **10,215 images, ~360k instances**, 4 scene types, 20–40 people/image.
  - Frozen split (MD5-verified): **6,129 train / 2,043 val / 2,043 test** — identical for every configuration.
  - Test geometry: **99.6 % < 32 px; > ⅓ below 8 px**.
  - Stated limitation, stated early `[italic]`: *composited origin ⇒ transfer to fully real imagery is future work.*

### Slide 30 · "Evaluation Metrics"
- Title `[30 · bold · #1F4E79]`.
- Equation box (top): P = TP/(TP+FP) · R = TP/(TP+FN) · F_β = (1+β²)PR/(β²P+R), **β = 2 → F₂ leads**
- Why-each list `[18–20 · regular; metric names bold]`:
  - **F₂** — miss ≫ false alarm in SAR; reported at its optimal confidence, and that threshold is reported too.
  - **AP₅₀ carries the signal** at this scale — a 1-px shift on an 8-px box swings IoU wildly; strict-IoU AP mostly measures annotation noise.
  - **Per-size recall** (very-tiny < 8 / tiny 8–16 / small 16–32 / medium 32–96 px) — the direct hypothesis test.
  - **AR₁₀₀** — scenes carry 20–40 people; a 100-detection budget is the realistic regime.
  - **ECE** + params / GFLOPs / end-to-end latency — the deployment claim is stated in those units.

### Slide 31 · "Results & Findings" — kicker `THE ADDITIVE ABLATION`
- Title `Results & Findings` `[30 · bold · #1F4E79]`; kicker `[16 · bold caps · #C0392B]`.
- Caption `[14 · bold · Small Caps]`: `TABLE VII — ADDITIVE ABLATION ON THE C2A TEST SET (COCO PROTOCOL)`
- Table `[cells 14; header 14–16 bold white on Navy]` — MSA-YOLO row: 10 % Navy tint + bold; the 41.1 cell: **Red #C0392B + ⚠**; column-best values bold:

| Model | Params (M) | GFLOPs | AP | AP₅₀ | AR₁₀₀ | F₁ | F₂ | Lat. (ms) |
|---|---|---|---|---|---|---|---|---|
| YOLO11m baseline | 20.03 | 67.7 | **0.615** | 0.843 | 0.691 | **0.850** | 0.840 | 13.7 |
| + CBAM | 19.08 | 66.9 | **0.616** | 0.847 | 0.692 | **0.850** | 0.841 | **13.5** |
| **+ CBAM + P2 (MSA-YOLO)** | 19.57 | 86.7 | 0.615 | **0.853** | **0.703** | 0.848 | **0.844** | 14.6 |
| + Mamba + CBAM + P2 | 22.01 | 98.4 | 0.614 | 0.852 | 0.704 | 0.846 | **0.844** | 41.1 ⚠ |

- Three findings strip `[18 · regular; numbers bold]`: ① P2 is the effective change (**+1.0** AP₅₀ pt, **+1.2** AR pt for 0.5 M / 1 ms) ② CBAM near-free (**−1 M** params, fastest) ③ Mamba: **+2.4 M, 2.8× latency, nothing**.
- Note: AP column is flat by design — predicted on Slide 30; say so.

### Slide 32 · "Results & Findings (Cont'd…)" — kicker `COMPONENT ATTRIBUTION` **[mergeable → right half of Slide 31]**
- Image: waterfall PNG (**export needed**), centered, max height.
  Caption `[14 · #595959]`: `Fig. - 20 : Per-component contribution to AP₅₀ (y-axis zoomed 0.838–0.856); params · latency under each column`
- One line `[20 · regular]`: *Every gain attributed to one change; the Mamba null is displayed, not hidden.*

### Slide 33 · "Results & Findings (Cont'd…)" — kicker `HYPOTHESIS TEST: PER-SIZE RECALL`
- Caption `[14 · bold · Small Caps]`: `TABLE VIII — PER-SIZE RECALL ON THE C2A TEST SET`
- Table (very-tiny column of MSA-YOLO row: bold + Green #82B366):

| Model | very-tiny (<8 px) | tiny (8–16) | small (16–32) | medium (32–96) |
|---|---|---|---|---|
| GT instances | 25,072 | 20,520 | 26,614 | 317 |
| YOLO11m baseline | 0.743 | 0.869 | 0.894 | 0.909 |
| + CBAM | 0.746 | 0.873 | 0.895 | 0.912 |
| **+ CBAM + P2** | **0.757** | 0.865 | 0.886 | 0.808 |
| + Mamba + CBAM + P2 | **0.757** | 0.870 | 0.887 | 0.811 |

- Image right (if space): `cbam_p2_per_size_recall.png` — caption: `Fig. - 21 : Per-size recall of the recommended model`
- Takeaway box `[20 · bold italic]`: *Hypothesis (Slide 12) CONFIRMED — the gain lands exactly in the very-tiny band (+1.5 pt).*
- Two honesty lines `[16 · regular · #595959]`:
  - Medium-band drop is noisy: only **317 of ~72,500** instances.
  - Mamba tracks CBAM+P2; scan diagnostics prove the blocks ran (fwd/rev cosine distance **0.836**) — the null is architectural, not a wiring fault.

### Slide 34 · "Results & Findings (Cont'd…)" — kicker `OPERATING POINT & CALIBRATION` **[mergeable]**
- Images 3-up, same height, 1 pt borders: `cbam_p2_pr_curve.png` · `cbam_p2_f1_conf.png` · `cbam_p2_calibration.png`
  Panel labels `[14 · bold]`: (a) precision–recall (b) F₁/F₂ vs confidence — dashed line at 0.16 (c) reliability diagram.
  Caption `[14 · #595959]`: `Fig. - 22 : MSA-YOLO operating behaviour on the C2A test set`
- Bullets `[18–20]`:
  - F₂-optimal threshold **0.16–0.20 across all configs** ⇒ no per-model tuning; MSA-YOLO deploys at **0.16**.
  - **ECE ≈ 0.021** ⇒ reported 0.8 confidence ≈ 0.8 empirical precision — operators can trust the scores for triage.

### Slide 35 · "Results & Findings (Cont'd…)" — kicker `QUALITATIVE ANALYSIS`
- Images 2×2, same size, 1 pt borders: `qual_gt.png` · `qual_baseline.png` · `qual_sahi256.png` · `qual_sahitta256.png`
  Panel labels `[14 · bold]`: (a) ground truth — 156 people (b) MSA-YOLO @640 px — 209 boxes (c) + SAHI 256 — 217 (d) + SAHI+TTA — 217.
  Caption `[14 · #595959]`: `Fig. - 23 : Dense collapsed-building scene — sliced/augmented modes add detections along the rubble line`
- Failure-modes line `[18 · regular; lead words bold]`: residual errors = **heavy occlusion** · **extreme scale** (few px, no texture) · **crowding** (merged boxes) — concentrated in the very-tiny band.

### Slide 36 · "Comparison with State of the Art"
- Title `[30 · bold · #1F4E79]`.
- Caption `[14 · bold · Small Caps]`: `TABLE IX — PUBLISHED DETECTORS ON THE C2A TEST SET (Nihal et al., same split, no slicing/TTA)`
- Table (ours rows: Navy tint; MSA-YOLO row bold; YOLOv9-e AP₅₀ bold as column-best):

| Model | AP₅₀ | AP | Params |
|---|---|---|---|
| Faster R-CNN | 0.634 | 0.366 | ~41 M |
| RetinaNet | 0.693 | 0.383 | ~37 M |
| RTMDet | 0.708 | 0.442 | — |
| Cascade R-CNN | 0.735 | 0.486 | ~69 M |
| DINO | 0.789 | 0.471 | ~47 M |
| YOLOv5 | 0.808 | 0.492 | — |
| YOLOv9-e | **0.893** | **0.688** | 57.3 M |
| YOLO11m baseline (ours) | 0.843 | 0.615 | 20.1 M |
| **MSA-YOLO (ours)** | 0.853 | 0.615 | **19.6 M** |

- Takeaway box `[20 · bold italic]`: *Second among ALL published detectors on this benchmark — one AP₅₀ point behind a model THREE TIMES our size. For airborne hardware, that is the right trade.*

### Slide 37 · "Inference-Time Study: SAHI vs TTA"
- Title `[28 · bold · #1F4E79]`.
- Caption `[14 · bold · Small Caps]`: `TABLE X — SAHI SWEEP AND TTA (MSA-YOLO, PER-IMAGE MATCHING @ IoU 0.5)` — condensed:

| Setting | Size | R | F₂ | very-tiny R | Lat. (ms) |
|---|---|---|---|---|---|
| none (baseline) | 640 | 0.835 | 0.839 | 0.758 | 15 |
| + SAHI | 256 | 0.846 | 0.848 | 0.788 | 162 |
| + SAHI | 512 | 0.837 | 0.842 | 0.763 | 66 |
| **+ TTA** | **1280** | **0.877** | **0.854** | **0.850** | **60** |

  (TTA row: Green tint #D5E8D4 + bold.)
- Image right: `per_size_recall_all_configs.png` — caption `[14 · #595959]`: `Fig. - 24 : Per-size recall under each inference-time setting — gains concentrate in the very-tiny band`
- Bullets `[18–20]`:
  - **TTA@1280 is the single best setting** — +9.2 pt very-tiny recall, cheaper than any SAHI config; degrades beyond 2× training resolution.
  - Both are **optional modes** — neither touches the trained weights.

### Slide 38 · "Objectives Achieved"
- Title `[30 · bold · #1F4E79]`.
- Four rows: objective `[20 · regular]` → ✔ `[24 · Green #82B366]` → evidence `[18 · regular; refs bold]`:
  1. Additive ablation under one protocol → ✔ **TABLE VII** + waterfall.
  2. Best config within deployable budget, near SOTA → ✔ **TABLE VIII** (best very-tiny recall) + **TABLE IX** (2nd of all published, ⅓ size).
  3. Honest state-space test → ✔ +2.4 M, 2.8× latency, no gain; module verified active.
  4. Inference-time recall without retraining → ✔ TTA **0.758 → 0.850** very-tiny @ 60 ms.

---

## PART G — CLOSING

### Slide 39 · "Thesis Timeline"
- Title `[30 · bold · #1F4E79]`.
- Image: Gantt crop from report PDF (**export needed**), full width.
  Caption `[14 · #595959]`: `Fig. - 25 : Project timeline across two 13-week terms`
- If rebuilding natively: Term-1 bars `[fill #DAE8FC, stroke #6C8EBF]`, Term-2 bars `[fill #D5E8D4, stroke #82B366]`, bar labels `[14 · regular]`:
  T1 — topic selection (w1–3) · literature review (w2–9) · C2A preparation (w6–9) · detector-family benchmark (w9–13) · pre-defense (w11–13).
  T2 — additive ablation CBAM/P2 (w1–5) · state-space exploration (w3–9) · SAHI & TTA (w7–11) · analysis & comparison (w8–12) · documentation & defense (w9–13).

### Slide 40 · "Limitations"
- Title `[30 · bold · #1F4E79]`.
- Four bullets `[20 · regular; lead phrase bold]` — state them before they're asked:
  - **Single, semi-synthetic dataset** (C2A) — transfer to fully real footage untested here.
  - **Single seed per configuration** — small deltas sit within run noise.
  - **Sparse medium band** — only 317 instances; its estimates are noisy.
  - **Desktop-GPU latency** — no on-drone measurement or field trial.

### Slide 41 · "Future Work"
- Title `[30 · bold · #1F4E79]`.
- Five bullets `[20 · regular]`; first two get a chip `IN PROGRESS` `[12 · bold white on Green #82B366, rounded]`:
  1. **Server-side deployment** — drone streams to ground server; detection + flagged frames back to operator; enables TTA@1280.
  2. **Enhanced dataset with own drone imagery** — attacks the composited-data domain gap directly.
  3. Validation on real rescue imagery — SARD, HERIDAL, VisDrone, Okutama (zero-shot + fine-tuned).
  4. Multi-seed significance testing with paired statistics for publication.
  5. A purpose-built state-space design — the null doesn't close the direction; we now have a validated protocol to test it against.

### Slide 42 · "Conclusion"
- Title `[30 · bold · #1F4E79]`.
- Four lines, large type `[24 · bold · #1F2933]` — the ONE slide where full-sentence bold is allowed; generous line spacing (1.3):
  - The gain on C2A comes almost entirely from the **P2 detection scale** — the resolution argument was right.
  - **CBAM** contributes efficiency: attention at negative parameter cost.
  - The **state-space neck contributes nothing** at 2.8× latency — an honest, verified null.
  - **MSA-YOLO: AP₅₀ 0.853 · F₂ 0.844 · 19.6 M params · 14.6 ms** — 2nd of all published C2A detectors at ⅓ the leader's size; +9 pt very-tiny recall via TTA when latency permits.

### Slides 43–44 · "References" / "References (Cont'd…)"
- Title `[30 · bold · #1F4E79]`.
- IEEE numbered entries `[13–14 · regular, hanging indent 0.3″, 9–10 per slide]` — pull exact
  entries from `Defense/draft1_30_6_26/references.bib`. Must include: Nihal 2024 (C2A) ·
  Jocher 2024 (YOLO11) · Woo 2018 (CBAM) · Wang 2020 (ECA) · Akyon 2022 (SAHI) · Gu & Dao
  2023 (Mamba) · Wang 2025 (Mamba-YOLO) · Lin 2017 (FPN, focal) · Zhu 2021 (TPH-YOLOv5,
  VisDrone) · Ren 2015 (Faster R-CNN) · Redmon 2016 (YOLO) · Zhao 2024 (RT-DETR) · Carion
  2020 (DETR) · Lin 2014 (COCO) · Sambolek 2021 (SARD) · Božić-Štulić 2019 (HERIDAL) ·
  Kyrkou 2020 (AIDER) · Qin 2020 (U²-Net) · Hu 2018 (SE) · Cheng 2023 (survey).
- Keep pre-defense numbering stable where possible ([1] Nihal, [3] TPH-YOLOv5, [4] AGIIndia).

### Slide 45 · "THANK YOU"
- `THANK YOU` centered `[44 · bold · #1F4E79]`; `arch_detections.png` faded (30 % opacity) as background; no footer.

---

## HIDDEN SLIDES (after the thank-you slide — Q&A jump targets)
Design: plain title `[28 · bold · #1F4E79]`, body `[18 · regular]`, same table style; no
kickers, no slide numbers.

**H1 — "Why YOLO11m?"** Detector-family benchmark (report appendix Tab. A.1): YOLOv9-s/m/e,
v10-s/m/l, 11-s/m/l — YOLO11m best accuracy-to-size balance (0.841 mAP50 @ 38.6 MB; v9-e
better but 111.8 MB). Label `[14 · italic · #595959]`: "earlier protocol — internally
comparable only."

**H2 — "Why CBAM and not ECA?"** Report Tab. 4.2: baseline F₂ 0.844 / small-R 0.881; +ECA
0.844 / 0.884; +CBAM **0.849 / 0.892**. ECA faster but bought no recall. Same italic
earlier-protocol label.

**H3 — "Is the Mamba null real?"** Image `fig_training_worst.png` (1 pt border). Three beats
`[18; lead words bold]`: ① **first attempt invalidated** — train-time rebuild silently
stripped injected blocks (model trained was CBAM+P2) → protocol gained module-presence
verification. ② **corrected run** — blocks verified at 6 layers; epoch-28 incident (val cls
loss 0.61 → 4.5, mAP collapse, 15-epoch recovery) — only the SSM run shows this fragility.
③ **scan diagnostics** — fwd/rev cosine distance 0.836 ⇒ module genuinely active.

**H4 — Layer configuration.** Report Tab. 3.1 verbatim (layers 0–29; rows 10 / 17–19 / 29
bold + Red text #C0392B) `[cells 12–14 — density okay on a hidden slide]`.

**H5 — Protocol robustness & run record.** Smoke-test gate (24 h) · per-epoch safety
checkpoints + resume · OOM ladder (16→8, effective 16) · 2-s GPU telemetry · env manifest.
Run times **7.3 / 7.0 / 10.2 / 29.5 h**; stop epochs 270 (cap) / 239 / 218 / 154 (best F₂
@ 149). Mamba's 4× wall-clock bought nothing.

**H6 — Confusion & calibration detail.** `cbam_p2_confusion.png` + `cbam_p2_calibration.png`
side by side; per-config ECE 0.023 / 0.021 / 0.021 / 0.020.

**H7 — Cost & carbon.** Replication workstation ≈ **BDT 285,000** (Dhaka mid-2026); ~54
GPU-hours @ 182–218 W ≈ 10.6 kWh GPU, ~20 kWh total ≈ BDT 200 electricity; campaign
carbon-accounted at **9.0 kg CO₂**. No paid data/cloud/licences.

**H8 — "Why is aggregate AP flat?"** All four configs within 0.002 AP — at 8-px scale a
1-px shift swings IoU, so strict-IoU AP measures annotation noise; AP₅₀, per-size recall
and F₂ carry the task signal. Predicted in the metrics section BEFORE results — point back
at Slide 30.

**H9 — SAHI full sweep + visuals.** Full report Tab. 4.7 (incl. 320/640 rows + P column);
`sahi_input_full.png` → `sahi_slice_grid.png` → `sahi_merged_detections_zoom.png`. Why TTA
beats SAHI here: C2A targets are dense and everywhere ⇒ full-frame super-resolution beats
tiling; SAHI shines on sparse large frames.

**H10 — Progress since pre-defense.** Two-column then/now `[18]`: pre-defense = YOLOv8 n/s/m
benchmark @ 50 epochs (best v8m mAP50 0.814) + HBPA proposal; final = family sweep fixing
YOLO11m (→H1), four-config additive ablation at full protocol, MSA-YOLO **AP₅₀ 0.853 /
F₂ 0.844**, honest Mamba null, SAHI/TTA study. Use if asked "what did you do in the second
term?"

---

## BUILD ORDER (practical)

1. Slide Master setup (top of this file) — fonts, colors, layouts, the 3 reusable boxes, table style.
2. Export the 3 missing PNGs (size distribution, waterfall, Gantt).
3. Slides 13–24 first (the progressive architecture build) — paste the pipeline diagram once,
   duplicate the slide per step, recolor one stage each time. Supervisor's priority.
4. Table slides next (6–9, 25, 28, 31–33, 36–37) — format TABLE I fully, then Format Painter.
5. Remaining content slides, dividers, references.
6. Hidden slides last — mostly copy-paste from report tables.
7. Pre-flight: flip through with arrow keys — titles must not jump; red appears ONLY on our
   modifications + warnings; `Fig.`/`TABLE` numbers sequential; fill outline page numbers;
   embed fonts; export PDF backup.

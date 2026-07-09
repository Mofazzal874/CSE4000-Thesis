# 2026-07-09 — Defense presentation slide plan (MSA-YOLO, CBAM+P2)

**Source of truth:** `Defense/draft1_30_6_26/` (report draft, read in full 2026-07-09).
**All image paths below are relative to** `Defense/draft1_30_6_26/figures/` unless stated.
**Model branding:** call the recommended model **MSA-YOLO (YOLO11m + CBAM + P2)** everywhere —
that is the title-page name. Say "CBAM+P2" when contrasting ablation rows.

---

## 0. Deck design rules (apply to every slide)

1. **Supervisor preference — step-by-step architecture.** The methodology section is a
   *progressive build*: the same pipeline diagram appears on each step slide with the
   current stage highlighted (the rest greyed). In PowerPoint: duplicate the slide,
   recolor one block per step — cheap and looks animated.
2. **Equation rule (user's rule):** every methodology slide that is *not* a pure
   architecture/big-image slide carries ONE brief equation tied to the image, plus a
   three-line frame: **What we break/integrate → What happens → What we find.**
3. **IEEE table style:** caption ABOVE the table, centered, `TABLE I` / `TABLE II` … in
   small caps with roman numerals; no vertical rules; only three horizontal rules
   (top, below header, bottom — booktabs look); left-align text columns, right-align numbers.
4. **Numbers are frozen** — every number below is verbatim from the report tables. Do not
   round differently on different slides.
5. Footer on every content slide: `MSA-YOLO — Md Mofazzal Hosen (2007074)` + slide number.

## 0.5 Alignment with the pre-defense deck (read 2026-07-09)

`Defense/Presentation/MSA_YOLO_...pptx` (34 slides) is the **first-term pre-defense**: it
proposed **HBPA-YOLO (hierarchical body-part attention)** and reported the YOLOv8 n/s/m
benchmark. The final deck must feel like the same presenter, matured. Port these
conventions verbatim:

- **Title slide layout:** "Presented By / Roll: 2007074" left, "Supervised By / Dr. Sk. Md.
  Masudul Ahsan, Professor, Dept. of CSE, KUET" right, course line `CSE 4000: Project/Thesis`.
- **Outline slide with page numbers** per section (their slide 2 style).
- **Full-slide section dividers**: Introduction · Related Works · Methodology ·
  Implementation & Results · (Conclusion). Add these; my slide numbering below treats them
  as zero-content dividers — 5 extra slides.
- **Caption style:** `Fig. - 03 : <caption>` under figures; `Table-0X : <caption>` ABOVE
  tables; keep sequential numbering across the whole deck.
- **"(Cont'd…)" suffix** on continuation slides; slide number on every content slide.
- **Related-works summary table columns** (their Table-1): *Authors | Method | Architecture |
  Challenges | Scope for Improvement*, split across slides with "Table-1 (Continued)". My
  Tables I–III below can keep the method-wise grouping the user asked for, but reuse this
  five-column layout so the committee sees continuity.
- **References:** same IEEE numbered format; entries [1] Nihal, [3] TPH-YOLOv5, [4] AGIIndia
  (intro image), [6] HERIDAL already exist in the pre-defense — keep their numbers stable
  where possible.

**⚠ The pivot must be addressed head-on.** The committee saw HBPA-YOLO (body-part
decomposition, per-part spatial attention, hierarchical fusion). The final thesis delivers
MSA-YOLO (CBAM + P2, additive ablation). Expect the question "what happened to body-part
attention?" — answer it BEFORE they ask with the bridge slide below.

**BRIDGE SLIDE (insert between Slide 12 and Slide 13, i.e. end of Objectives):**
- Title: *From the pre-defense proposal to the final design*
- Layout: left = pre-defense HBPA block diagram thumbnail (screenshot their slide 18);
  right = `fig_ablation_chain.png`.
- Content (3 arrows, pre-defense idea → what it became):
  - *Body-part decomposition (head/torso/limbs)* → **infeasible to supervise on C2A**: targets
    are < 10 px and annotations are whole-body boxes only — no part labels exist at this scale
    → replaced by a measurement-first design. *(Verify this wording with the supervisor — state
    the actual reason discussed at pre-defense feedback if different.)*
  - *Attention for disaster clutter* → survived, matured: **CBAM as a substitution** (measured
    against ECA), not a bolt-on.
  - *Tiny-object focus* → survived, sharpened: **stride-4 P2 scale**, quantified in a
    single-variable ablation + an honest state-space test.
- Punchline: *The proposal's goal — recall on tiny, occluded humans — is unchanged; the route
  to it became controlled measurement instead of an unverifiable decomposition.*

### Export checklist (3 figures have no PNG yet)
| Figure | Source | Action |
|---|---|---|
| C2A size distribution | `fig_size_distribution.drawio` (TikZ in report) | draw.io → export PNG (300 dpi), or crop from compiled report PDF |
| Ablation waterfall | `fig_ablation_waterfall.drawio` (TikZ in report) | draw.io → export PNG |
| Gantt timeline | `tikz/fig_gantt.tex` only | crop from compiled report PDF page (Ch. I) |

Everything else already exists as PNG in `figures/`.

---

## 0.75 Design system — fonts, sizes, colors, layout (build this in Slide Master FIRST)

Measured from the pre-defense PPTX for continuity: **16:9 canvas** (13.33 × 7.5 in),
fonts actually used were **Open Sauce / Open Sans Bold** (Arial fallback), body ≈ 22 pt,
titles 32–48 pt. Keep that feel; tighten the system as below.

### Fonts
| Role | Font | Fallback |
|---|---|---|
| Slide titles, dividers, lead words | **Open Sans Bold** (or Open Sauce Bold) | Calibri Bold |
| Body text, tables, captions | **Open Sauce** / Open Sans Regular | Calibri |
| Equations | PowerPoint equation editor (Cambria Math) **or** crops from the compiled report PDF (Latin Modern serif) | — |

Rules: never more than these families on a slide. The serif look of PDF-cropped equations
against sans body is fine — it signals "math", and it guarantees the symbols match the
report exactly. Pick ONE of the two equation methods and use it deck-wide.

### Type scale (pt) — do not go below 16 pt for anything the committee must read
| Element | Size | Weight / style |
|---|---|---|
| Title-slide thesis title | 30–32 | Bold, dark navy, centered, ≤ 3 lines |
| Title-slide name / roll / supervisor block | 20 | Regular; the labels "Presented By" / "Supervised By" 18 Bold |
| Section divider text | 44 | Bold, white on navy band (see T2) |
| Slide title | 30 (shrink to 28 only if it wraps) | Bold, left-aligned, one line |
| Step kicker ("STEP 3 / 7 · CBAM") | 16 | Bold, red accent, all caps, top-left above title |
| Body bullets, level 1 | 20–22 | Regular; one idea per bullet, ≤ 2 lines each |
| Body bullets, level 2 | 18 | Regular |
| Break / Integrate / Find strip | 20 | Lead word Bold + colored (see palette); rest Regular |
| Takeaway / punchline line | 20 | Bold Italic, inside a 10%-navy tinted box, bottom of slide |
| Table caption (`Table-07 : …`) | 14 | Bold, small caps, centered ABOVE table |
| Table header row | 16 | Bold |
| Table body cells | 14–16 | Regular (18 if the table is ≤ 5 rows); numbers right-aligned |
| Figure caption (`Fig. - 03 : …`) | 14 | Regular, dark grey #595959, centered below |
| Panel labels (a)(b)(c) | 14 | Bold, below each panel |
| Display equation | 24–28 rendered height | Centered in equation box |
| Footer (`MSA-YOLO — Md Mofazzal Hosen (2007074)`) + slide number | 11 | Regular, grey #7F7F7F |
| References | 13–14 | Regular, hanging indent 0.3 in, 9–10 entries/slide |

### Color palette (hex — matches the report's figure colors)
| Use | Color |
|---|---|
| Body text | #1F2933 (near-black; never pure black on projectors) |
| Slide titles, structure, table header fill | Navy **#1F4E79** (header text white) |
| Diagram blue (baseline/unchanged blocks) | fill #DAE8FC, stroke #6C8EBF — same as report draw.io figures |
| Green = gains, "keep", checkmarks, **Find:** | #82B366 (fill #D5E8D4) |
| **Red = OUR modifications + warnings ONLY** | #C0392B (stroke #B85450, fill #F8CECC) — CBAM/P2 highlight boxes, red-dashed outlines, the 41.1 ms cell, **Break:** |
| Grey = de-emphasized / not-adopted (Mamba blocks, greyed pipeline stages) | #9E9E9E at 40% transparency |
| Background | Pure white, no gradients, no textures, no stock template art |

Red discipline is the whole design: the committee's eye should learn within two slides
that *red = what we changed*. Never use red decoratively.

### Bold / emphasis rules
- Bold is load-bearing only: claimed numbers (**0.853**, **0.757**, **19.6 M**, **2.8×**),
  model names on first mention per slide, lead words (**Break:/Integrate:/Find:**),
  best-per-column values in tables. Max ~2 bold spans per bullet.
- Never bold a full sentence — sole exception: the four conclusion lines (Slide 42).
- Italic: "et al.", the "illustrative" label on schematic tiles, takeaway lines.
- Underline: never. Text shadows, WordArt, clip-art: never.

### Table styling (IEEE, one saved style reused everywhere)
- No vertical rules. Three horizontal rules: 1.5 pt top and bottom, 0.75 pt under the header.
- Header row: navy fill #1F4E79, white 16 pt bold. Body rows: white / #F2F2F2 alternating
  (optional; skip if ≤ 5 rows).
- **Recommended-model row:** 10% navy tint + bold across the row.
- Numbers: right-aligned, consistent decimals (3 for metrics: 0.853; 1 for ms and points:
  14.6 ms, +1.5 pt). Column-best in bold, exactly as the report tables do.
- Row height ≥ 0.3 in; cell padding 0.05 in; never let a table touch the slide edge.

### Figure styling
- Photos / real model outputs: 1 pt #BFBFBF border. Schematics (white-background draw.io
  exports): no border.
- All C2A photos on one slide = same rendered height. Zoom callouts: 2.25 pt red circle +
  straight red leader line (no elbow connectors).
- Keep sequential `Fig. - 0X` numbering across the deck (pre-defense convention).

### Layout templates (create as Slide Master layouts, then never hand-place)
- **Margins:** 0.5 in all sides. Title baseline fixed at 0.55 in from top on EVERY content
  slide — titles must not jump between slides.
- **T1 Title slide:** KUET logo 1.0 in tall top-center; title block centered at upper third;
  two-column bottom block (Presented By | Supervised By); course line at footer.
- **T2 Section divider:** full-width navy band (#1F4E79) across the vertical middle, 44 pt
  white bold text; section number small (20 pt, white, 60% opacity) above it. No footer.
- **T3 Text + image:** text column left 55%, image right 40%, 0.3 in gutter.
- **T4 Big image:** title + image max 5.8 in tall centered + single 14 pt caption line.
  Nothing else — resist adding bullets to these.
- **T5 Table slide:** caption (14 pt small caps) + table centered + takeaway box bottom.
- **T6 Progressive architecture (slides 14–22):** the pipeline diagram at IDENTICAL
  coordinates on every step slide (paste once, duplicate slide) so nothing shifts when
  advancing; current stage full-color + 2.5 pt red outline, all other stages 40%-transparent
  greyscale; step kicker top-left; equation box bottom-left; Break/Integrate/Find strip
  bottom-right. Advancing through these slides = the "animation".
- **Equation box:** white fill, 1 pt #6C8EBF border, 4 pt rounded corners, 8 pt padding.

### Pre-flight checklist (before exporting to PDF)
1. Title y-position identical on all content slides (flip through with arrow keys — nothing may jump).
2. Footer + slide number on every slide except title, dividers, thank-you.
3. `Fig.` and `Table` numbering sequential with no gaps; decimals consistent per the rule above.
4. Red appears only on: our-modification highlights, Break:, the 41.1 ms cell, warning chips.
5. View every slide at 66% zoom from 2 m away — if you can't read it, the back row can't.
6. Embed fonts on save (File → Options → Save → Embed fonts) — the defense PC will not have Open Sauce.

---

## MAIN DECK (36 slides + 9 hidden)

If the committee caps you at ~20 min, merge the slides marked **[mergeable]** — the deck
collapses to ~24 slides without losing the step-by-step architecture story.

---

### PART A — OPENING

**Slide 1 — Title**
- Content: *MSA-YOLO: A Multi-Scale Attention Enhancement of YOLO for Tiny-Human
  Detection in Aerial Search-and-Rescue Imagery* · Md Mofazzal Hosen, Roll 2007074 ·
  Supervisor: Prof. Sk. Md. Masudul Ahsan · Dept. of CSE, KUET · July 2026.
- Image: `kuet_logo.png` top-center.
- Note: memorize the one-breath version: "a compact detector for people only a few
  pixels wide in disaster drone imagery, built by measuring one change at a time."

**Slide 2 — Outline**
- Content (numbered): Introduction → Applications → Related Works → Objectives →
  Methodology → Implementation & Results → Timeline · Future Work · Conclusion → References.
- Layout: vertical list, the 6 methodology "steps" previewed as small thumbnails of the
  pipeline diagram (sells the step-by-step format up front).

---

### PART B — INTRODUCTION (3 slides)

**Slide 3 — Motivation: the first hours decide who is found alive**
- Image (full-bleed right half): `fig_intro_scene.png` (flood scene).
- Content (left half, 4 bullets):
  - Earthquake / flood / fire → UAVs survey the site far faster than ground teams, with no risk to rescuers.
  - But: a person seen from altitude ≈ **fewer than 10 pixels**, amid rubble, water, smoke.
  - One sortie → **thousands of frames**; a human operator scanning that stream *will* miss people.
  - Automating the search is a practical need — this thesis addresses it.
- Note: this is the emotional hook; speak the last bullet slowly.

**Slide 4 — Why is this hard? Tiny, occluded, cluttered, crowded**
- Images: 2×2 grid — `c2a_collapsed.png`, `c2a_fire.png`, `c2a_flood.png`, `c2a_traffic.png`
  (label each: collapsed building / fire / flood / traffic incident).
- Content (bottom strip, 3 facts):
  - ~47% of annotated people in the C2A benchmark are **< 10 px** — at or below a standard detector's resolving limit.
  - **20–40 people per scene**, many partially buried or occluded.
  - Detection maturity (R-CNN → YOLO → DETR) does **not** transfer: targets are an order of magnitude smaller than what those detectors were tuned for.
- Note: no equations yet — save the geometry argument for the hypothesis slide.

**Slide 5 — Applications**
- Content (5 numbered items, one icon each):
  1. **Search-and-rescue triage** — drone/laptop flags likely victims live; operators review candidates, not raw frames.
  2. **Disaster damage assessment** — detections aggregated over a flight show planners where people are concentrated.
  3. **Crowd monitoring** — counting tiny people from altitude is the same technical problem.
  4. **Wide-area security** — border/coastline patrol = low-density variant of the task.
  5. **Embedded deployment** — 19.6 M params fits edge GPUs; recipe transfers to smaller YOLO11 backbones.
- Image (optional corner): reuse `arch_detections.png` small, captioned "what the operator sees".
- Note **[mergeable]** with Slide 3 if time-limited (keep items 1, 2, 5).

---

### PART C — RELATED WORKS (4 slides — method-wise tables, IEEE format)

**Slide 6 — Related works I: evolution of object detectors**
- Caption: `TABLE I — DETECTOR FAMILIES AND THEIR TRADE-OFFS`

| Family | Representative works | Key idea | Trade-off for our task |
|---|---|---|---|
| Two-stage | R-CNN (2014), Faster R-CNN (2015), Cascade R-CNN (2018) | propose regions, then classify | accurate but too slow/heavy for airborne use |
| One-stage | SSD (2016), RetinaNet + focal loss (2017), YOLO v1→v11 (2016–2024) | predict boxes in one pass | real-time, compact — **our family** |
| Transformer | DETR (2020), RT-DETR (2024) | set prediction, no NMS/anchors | competitive speed only recently; heavier |

- Takeaway line (below table): *We work in the YOLO family because speed and model size decide airborne deployability.*

**Slide 7 — Related works II: small-object & aerial detection methods**
- Caption: `TABLE II — TECHNIQUES FOR SMALL/AERIAL TARGETS`

| Approach | Representative works | How it helps tiny targets | Limitation in prior work |
|---|---|---|---|
| Higher-resolution scales | FPN (2017), TPH-YOLOv5 (2021), P2/stride-4 heads | detection grid fine enough for few-px targets | usually reported alone, on generic aerial data |
| Feature attention | SE (2018), ECA (2020), CBAM (2018) | re-weights features so faint targets survive clutter | choice asserted, rarely measured on-task |
| Inference-time slicing | SAHI (2022) | overlapping crops enlarge small targets | latency cost; no architecture change |
| Benchmarks | VisDrone (2021) | makes small objects the dominant difficulty | generic scenes, not disaster imagery |

- Takeaway: *All three technique lines are relevant — nobody had compared them under ONE protocol on disaster imagery.*

**Slide 8 — Related works III: SAR datasets & state-space models**
- Caption: `TABLE III — SAR HUMAN DETECTION AND STATE-SPACE MODELS`

| Theme | Representative works | Status | Gap |
|---|---|---|---|
| Real SAR datasets | SARD (2021), HERIDAL (2019) | valuable, real imagery | small (~2k frames) — too little to train a 20M-param detector |
| Synthetic-composite benchmark | **C2A** — Nihal et al. (2024): AIDER backgrounds + U²-Net-segmented humans | 10,215 images, ~360k instances, 4 scene types | reports whole models vs whole models — no component attribution |
| State-space models (SSM) | S4 (2022) → Mamba (2023) → Vision Mamba / VMamba (2024) → Mamba-YOLO (2025) | linear-time alternative to attention | applied as backbone replacement; never a controlled test on tiny humans |

- Takeaway: *C2A is our benchmark and Nihal et al. our direct comparison; the SSM claim is untested at this scale — we test it.*

**Slide 9 — The research gap**
- Caption: `TABLE IV — POSITION OF THIS THESIS RELATIVE TO PRIOR WORK` (verbatim report Tab. 2.1)

| Aspect | Prior work | This thesis |
|---|---|---|
| Small-object techniques | reported individually | compared in **one controlled ablation** |
| Attention choice | asserted or single option | CBAM vs ECA **measured on the task** |
| State-space neck | proposed as a backbone, generic data | tested **in the neck** vs identical baseline; reported honestly |
| Target domain | mostly generic aerial imagery | disaster/SAR imagery (C2A), very small humans |

- Note: read the right column aloud — this slide IS the transition into objectives.

---

### PART D — OBJECTIVES (3 slides)

**Slide 10 — Problem statement**
- Image (right): `qual_gt.png` captioned "one C2A test scene: 156 annotated people".
- Content:
  - Given aerial image *I* → output boxes \(\hat B=\{(x,y,w,h,c)\}\), one per visible person.
  - Correct when \(\mathrm{IoU}(\hat B,B)=\frac{\mathrm{area}(\hat B\cap B)}{\mathrm{area}(\hat B\cup B)} \ge 0.5\).
  - **Domain asymmetry:** a missed person may never get a second chance; a false alarm costs seconds of review.
- Equation (highlight box): \(F_2=\dfrac{5PR}{4P+R}\) — recall weighted 4× over precision; our operational metric throughout.
- Note: "The engineering question: which architectural changes raise recall on sub-ten-pixel people, within an airborne parameter/latency budget?"

**Slide 11 — Objectives & our novelty**
- Image (right): `fig_ablation_chain.png` (the 4-config chain).
- Content — Objectives (4, verbatim-condensed from Ch. I):
  1. Build a compact detector via an **additive ablation** — CBAM, P2 scale, SSM neck introduced one at a time under one frozen protocol.
  2. Identify the configuration best serving very-tiny recall within a deployable budget; show it performs near published SOTA.
  3. **Honestly test** the claim that state-space modules help detection — report the outcome either way.
  4. Raise recall further at inference time **without retraining** (SAHI, TTA).
- Novelty strip (bottom, 3 chips):
  - **CBAM as a substitution, not a bolt-on** — replaces C2PSA ⇒ *negative* parameter cost.
  - **P2 head quantified, not assumed** — additive branch; P3–P5 untouched ⇒ gain attributable.
  - **Single-variable ablation on C2A** — does not exist in prior work; negative results are first-class findings.

**Slide 12 — Hypothesis: the stride geometry says P2 must win**
- Image (left): `fig_stride_problem.png` (same 12-px human under 4 grids).
- Image (right): size-distribution PNG (**export needed**, see checklist) — 99.6% of C2A test instances < 32 px.
- Equation: cell coverage \(c_d(s)=\dfrac{s}{d}\) → for s ≈ 12 px: **0.4** cell @P5, 0.75 @P4, 1.5 @P3, **3 cells @P2 (stride 4)**.
- Content:
  - A detector cannot localize what falls inside a fraction of one grid cell.
  - Stride 4 is the first scale where a tiny human spans enough cells to be localized.
  - **Testable hypothesis:** if a stride-4 scale helps, the gain must appear in the very-tiny (< 8 px) band — nowhere else can compensate.
- Note: this equation returns on the per-size results slide — the hypothesis is falsifiable, and slide 27 shows the test.

---

### PART E — METHODOLOGY (13 slides, progressive build)

**Slide 13 — System overview (BIG IMAGE, no equation)**
- Image (full width): `fig_overall_framework.png`.
- Content (3 phase labels only, spoken not written):
  ① Feature extraction + attention (backbone → CBAM at P5) ·
  ② Multi-scale fusion extended to stride 4 (PAN–FPN + P2 branch) ·
  ③ Four-scale decoding + NMS.
- One line under figure: *Red-dashed = the two proposed modifications. Everything else is the YOLO11m baseline — deliberately.*
- Note: announce the format: "I will now walk one image through this pipeline, stage by stage."

**Slide 14 — Step 1 · Input & preprocessing**
- Progressive build: overview diagram with the INPUT block highlighted.
- Image (right): `arch_input.png` — "our running example: collapsed-building scene, 30+ people, most < 16 px".
- Equation: letterbox keeps aspect ratio, pixels normalized \(x' = x/255 \in [0,1]\), input fixed at \(640\times640\).
- Break/integrate/find:
  - **Keep:** default YOLO augmentations (mosaic, scale, HSV), mosaic off for last 10 epochs.
  - **Break:** aggressive small-object augmentation (copy-paste, mixup) — *tested and discarded, negative result*.
  - **Why letterbox matters here:** a stretched 8-px person is a destroyed 8-px person.

**Slide 15 — Step 2 · Backbone: the feature pyramid**
- Progressive build: BACKBONE stage highlighted.
- Image: crop of the backbone column from `fig_cbam_p2_architecture.png` (P1–P5 tiles with resolutions).
- Equation: stride halving — at 640 input, level \(P_k\) has resolution \(\left(\tfrac{640}{2^k}\right)^2\): P2 = 160², P3 = 80², P4 = 40², P5 = 20².
- Content: C3k2 stages produce features at strides 2–32; **layers 2/4/6 are kept as skip sources** — layer 2 (160²×256) becomes the P2 skip later.
- Find: nothing modified here — the backbone is untouched, which is what keeps the ablation clean.

**Slide 16 — Step 3 · CBAM: attention that pays for itself**
- Progressive build: CBAM block highlighted (backbone layer 10).
- Image (right): `fig_cbam_module.png`.
- Equations (the pair, shown small but complete):
  \(\mathbf{M}_c(\mathbf{F})=\sigma(\mathrm{MLP}(\mathrm{AvgPool}(\mathbf F))+\mathrm{MLP}(\mathrm{MaxPool}(\mathbf F)))\)
  \(\mathbf{M}_s(\mathbf F')=\sigma(f^{7\times7}[\mathrm{AvgPool}(\mathbf F');\mathrm{MaxPool}(\mathbf F')])\)
  refined multiplicatively: \(\mathbf F'=\mathbf M_c\otimes\mathbf F,\;\; \mathbf F''=\mathbf M_s\otimes\mathbf F'\).
- Break/integrate/find:
  - **Break:** remove the baseline's TWO heavy C2PSA self-attention blocks.
  - **Integrate:** ONE CBAM (reduction 16, 7×7 spatial kernel) at layer 10 — channel then spatial re-weighting of the 512-ch P5 features.
  - **Find:** ≈ **1 M fewer parameters than the baseline**, lowest latency of all four configs (13.5 ms), F₂ held/improved. Attention arrives at *negative* cost.

**Slide 17 — Step 3 in action (BIG IMAGE, no equation)**
- Images side by side: `fig_cbam_effect.png` (schematic: clutter suppressed, target amplified — label "illustrative") and `cbam_overlay.png` (REAL spatial-attention map from the trained model, warm = attended).
- One caption line: *The real attention concentrates on human-shaped regions, not the rubble. (Map read from the model's activations; it is 20×20 upscaled — blocky is correct, CBAM sits at stride 32.)*
- Note: examiners like the honesty of "blocky is correct" — say it before they ask.

**Slide 18 — Step 4 · The P2 branch: a detection scale where the targets are**
- Progressive build: NECK highlighted with the new stride-4 branch in red.
- Image (right): `fig_p2_head.png`.
- Equation: recall \(c_d(s)=s/d\): a 12-px person = **3 cells** at stride 4 — each P2 cell covers only 4×4 input px, resolving targets down to ≈ 4 px.
- Break/integrate/find:
  - **Integrate:** top-down pass continued ONE level further — upsample fused P3 (80²×256) → concat with backbone-layer-2 skip (160²×256) → C3k2 fuse → **160²×128 P2 feature**, read by a stride-4 detection branch (neck layers 17–19).
  - **Break: nothing** — P3–P5 pathway untouched; the branch is purely additive.
  - **Find (preview):** the principal driver of very-tiny recall: 0.743 → 0.757, AP₅₀ 0.843 → 0.853 — for ~0.5 M params, +1 ms.

**Slide 19 — Step 4 in action (BIG IMAGE, no equation)**
- Images: `detgrid_c2a_s8.png` (stride-8 grid) vs `detgrid_c2a_s4.png` (stride-4 grid) on the SAME real cluster crop; below or right, `p2_featuremap.png` ("the stride-4 feature fires exactly on the smallest scattered figures").
- Caption: *Stride 8 assigns several neighbouring people to one cell; stride 4 separates them. This is the spatial mechanism behind the very-tiny-recall gain.*

**Slide 20 — Step 5 · The explored variant: C3k2Mamba (state-space neck)**
- Progressive build: six neck C3k2 blocks highlighted in a different color (exploration, not recommendation).
- Image (right): `fig_c3k2mamba.png`.
- Equation (generic selective-SSM recurrence, labeled as such): \(h_t=\bar A h_{t-1}+\bar B x_t,\;\; y_t=C h_t\) — with input-dependent (selective) \(\bar A,\bar B\); scans run forward + reverse over a local window (win 6–8, \(d_{state}=4\)) and are fused.
- Break/integrate/find:
  - **Integrate:** C3k2Mamba replaces the C3k2 bottleneck at 6 neck layers (13, 16, 19, 22, 25, 28); backbone and head untouched.
  - **Why test it:** recent literature promotes SSMs for detection — but always as backbone swaps, never controlled.
  - **Find (preview):** +2.4 M params, **2.8× latency, no accuracy change** — a genuine, verified null (details in Results).

**Slide 21 — Step 6 · Loss design: unchanged, on purpose**
- Progressive build: HEAD highlighted, "training signal" arrows.
- Equations (the three terms + total; show compactly):
  - \(\mathcal L_{cls} = -[y\log\hat p+(1-y)\log(1-\hat p)]\) — BCE on the one *person* class.
  - \(\mathcal L_{CIoU}=1-\mathrm{IoU}+\frac{\rho^2(\mathbf b,\mathbf b^{gt})}{c^2}+\alpha v\) — overlap + center distance + aspect ratio: IoU alone is unstable for few-px boxes under sub-pixel shifts.
  - \(\mathcal L_{DFL}\) — box edges as distributions over binned offsets, sharpened around the true offset.
  - \(\mathcal L=7.5\,\mathcal L_{CIoU}+1.5\,\mathcal L_{DFL}+0.5\,\mathcal L_{cls}\).
- Break/integrate/find:
  - **Keep identical across all four configurations** → the ablation compares architectures, not objectives.
  - Heavy box weighting suits the task: with one class the problem is not *what* but **where**.
  - The P2 level joins all three terms exactly like the standard levels — no re-balancing needed.
- Note **[mergeable]**: if short on time show only \(\mathcal L_{CIoU}\) + the total.

**Slide 22 — Step 7 · Four-scale decoding + NMS → output**
- Progressive build: full pipeline lit, DETECT + NMS highlighted.
- Image (right): `arch_detections.png` — the running example's final boxes.
- Equation: head predicts at strides {4, 8, 16, 32} (grids 160², 80², 40², 20²); NMS keeps clusters of nearby people as separate detections.
- Find: output = one box + confidence per recovered person — what the operator reviews.

**Slide 23 — The complete architecture (BIG IMAGE, no equation)**
- Image (full slide): `fig_cbam_p2_architecture.png` — layer-level MSA-YOLO with the real insets (input / attention / P2 response / output).
- One line: *This exact configuration — CBAM at layer 10, P2 branch at layers 17–19, four-scale head at layer 29 — is the verbatim training configuration.*
- Note: pause here; this is the money slide. Hidden slide H4 has the full layer table if asked.

**Slide 24 — The whole trace at a glance (BIG IMAGE, no equation) [mergeable]**
- Images 2×2: `arch_input.png` / `cbam_overlay.png` / `p2_featuremap.png` / `arch_detections.png` with labels (a) input (b) CBAM spatial attention (c) P2 stride-4 response (d) detections.
- Caption: *Every intermediate is read from the trained model's activations on this image — the pipeline does what the diagrams claim.*
- **[mergeable]** into Slide 23 if needed.

**Slide 25 — Methodology capstone: the additive ablation design**
- Caption: `TABLE V — THE ADDITIVE ABLATION` (verbatim report Tab. 3.2):

| Configuration | Backbone attention | Detection scales | Neck |
|---|---|---|---|
| YOLO11m (baseline) | C2PSA | P3, P4, P5 | C3k2 |
| + CBAM | **CBAM** | P3, P4, P5 | C3k2 |
| + CBAM + P2 | CBAM | **P2**, P3, P4, P5 | C3k2 |
| + Mamba + CBAM + P2 | CBAM | P2, P3, P4, P5 | **C3k2Mamba (6 layers)** |

- Content: one change per row; identical AdamW (lr₀ = 0.001, cosine), ≤ 300 epochs, early stopping, same frozen split, same thresholds ⇒ *every reported difference is attributable to the architecture alone.*
- Note: "This design is why the negative Mamba result is interpretable: it fails under conditions where P2 demonstrably succeeds."

**Slide 26 — Inference-time mode 1: SAHI (slicing)**
- Image (top): `fig_sahi.png`; (bottom strip, real): `sahi_slice_grid.png` → `sahi_merged_detections.png`.
- Equation: tiles merged by greedy non-maximum merging with intersection-over-smaller,
  \(\mathrm{IOS}(a,b)=\dfrac{\mathrm{area}(a\cap b)}{\min(\mathrm{area}(a),\mathrm{area}(b))} \ge 0.5\) — suits boxes cut at tile borders.
- Break/integrate/find:
  - **Integrate at inference only:** overlapping tiles (256–640 px, 25–30% overlap) + one full-image pass; weights untouched.
  - **Find (preview):** 256-px slices: very-tiny recall 0.758 → 0.788 (+3.0 pt) — but 162 ms/image.

**Slide 27 — Inference-time mode 2: TTA (multi-scale + flip)**
- Image (top): `fig_tta.png`; (side, real): `tta_detections_zoom.png`.
- Equation: predictions merged over \(\{1.0, 0.83, 0.67\}\times\{\mathrm{id},\mathrm{hflip}\}\) at **1280-px** input, reduced by standard NMS. Fully-convolutional ⇒ can run at 2× training resolution.
- Break/integrate/find:
  - **Find (preview):** very-tiny recall 0.758 → **0.850** (+9.2 pt) and F₂ → 0.854 at only 60 ms — beats every SAHI setting at a fraction of the cost. Collapses beyond 2× training resolution.

---

### PART F — IMPLEMENTATION & RESULTS (8 slides)

**Slide 28 — Experimental setup**
- Two-column layout.
- Left (hardware/software): RTX 4070 Ti SUPER 16 GB · i7-14700K · 128 GB RAM · Windows 11 · Python 3.11.9 · PyTorch 2.12.0 (CUDA 12.6) · Ultralytics 8.4.56 · single GPU.
- Right (training config, IEEE mini-table `TABLE VI`): 640² letterbox · AdamW lr₀ 0.001 → cosine · ≤ 300 epochs, patience 50 (fitness) + 40 (F₂) · batch 16 (P2 models: physical 8, effective 16 via grad-accum) · AMP on · seed 0.
- One honesty bullet: *AdamW is pinned because the framework's default SGD diverged on the P2 architecture (~epoch 50); AdamW trained all four stably.*
- Protocol chip: smoke test before every run · per-epoch safety checkpoints · OOM retry ladder · full GPU telemetry · environment manifest per run ⇒ every number traceable.

**Slide 29 — Dataset: C2A**
- Images: the same 2×2 scene grid (small, reuse from Slide 4) + size-distribution PNG (**export needed**).
- Content:
  - C2A = AIDER disaster backgrounds + U²-Net-segmented human poses, semi-synthetic; 10,215 images, ~360k instances, 4 scene types, 20–40 people/image.
  - Frozen split (MD5-verified): **6,129 train / 2,043 val / 2,043 test** — identical for every configuration.
  - Test-set geometry: **99.6% of instances < 32 px; > ⅓ below 8 px.**
  - Stated limitation, stated early: composited origin ⇒ transfer to fully real imagery is future work.

**Slide 30 — Evaluation metrics: chosen for the task, not by habit**
- Equations: \(P=\frac{TP}{TP+FP}\), \(R=\frac{TP}{TP+FN}\), \(F_\beta=(1+\beta^2)\frac{PR}{\beta^2P+R}\) (β = 2 → F₂ leads).
- Content (why-each, one line apiece):
  - **F₂** — miss ≫ false alarm in SAR; reported at its optimal confidence, and that threshold is reported too.
  - **AP₅₀ carries the signal** at this scale — a 1-px shift on an 8-px box swings IoU wildly, so strict-IoU AP mostly measures annotation noise.
  - **Per-size recall** (very-tiny < 8 px / tiny 8–16 / small 16–32 / medium 32–96) — the direct hypothesis test.
  - **AR₁₀₀** — scenes carry 20–40 people; 100-detection budget is the realistic regime.
  - **ECE** + params / GFLOPs / end-to-end latency — the deployment claim is stated in those units.

**Slide 31 — MAIN RESULT: the additive ablation**
- Caption: `TABLE VII — ADDITIVE ABLATION ON THE C2A TEST SET (COCO PROTOCOL)` (verbatim report Tab. 4.3):

| Model | Params (M) | GFLOPs | AP | AP₅₀ | AR₁₀₀ | F₁ | F₂ | Lat. (ms) |
|---|---|---|---|---|---|---|---|---|
| YOLO11m baseline | 20.03 | 67.7 | **0.615** | 0.843 | 0.691 | **0.850** | 0.840 | 13.7 |
| + CBAM | 19.08 | 66.9 | **0.616** | 0.847 | 0.692 | **0.850** | 0.841 | **13.5** |
| **+ CBAM + P2 (MSA-YOLO)** | 19.57 | 86.7 | 0.615 | **0.853** | **0.703** | 0.848 | **0.844** | 14.6 |
| + Mamba + CBAM + P2 | 22.01 | 98.4 | 0.614 | 0.852 | 0.704 | 0.846 | **0.844** | **41.1** ⚠ |

- Highlight the recommended row (box it); color the 41.1 ms red.
- Three spoken findings: ① P2 is the effective change (+1.0 AP₅₀ pt, +1.2 AR pt for 0.5 M / 1 ms) ② CBAM is near-free (−1 M params, fastest) ③ Mamba: +2.4 M, 2.8× latency, nothing.
- Note: AP column is flat by design — that was predicted on Slide 30; say so.

**Slide 32 — The same story as a picture: ablation waterfall**
- Image: waterfall PNG (**export needed**) — AP₅₀ zoomed 0.838–0.856, green bars +0.0041 (CBAM) and +0.0060 (P2), red bar −0.0012 (Mamba), params·latency under each column.
- One line: *Every gain attributed to one change; the Mamba null is displayed, not hidden.*
- **[mergeable]** into Slide 31 (table left, waterfall right) if time-limited.

**Slide 33 — Hypothesis test: per-size recall**
- Caption: `TABLE VIII — PER-SIZE RECALL ON THE C2A TEST SET` (verbatim report Tab. 4.4):

| Model | very-tiny (<8 px) | tiny (8–16) | small (16–32) | medium (32–96) |
|---|---|---|---|---|
| GT instances | 25,072 | 20,520 | 26,614 | 317 |
| YOLO11m baseline | 0.743 | 0.869 | 0.894 | 0.909 |
| + CBAM | 0.746 | 0.873 | 0.895 | 0.912 |
| **+ CBAM + P2** | **0.757** | 0.865 | 0.886 | 0.808 |
| + Mamba + CBAM + P2 | **0.757** | 0.870 | 0.887 | 0.811 |

- Image (right): `cbam_p2_per_size_recall.png`.
- Content: **hypothesis from Slide 12 confirmed** — the gain lands exactly in the very-tiny band (+1.5 pt). Medium-band drop is noisy: only 317 of ~72,500 instances. Mamba tracks CBAM+P2 ⇒ adds nothing in the regime it was meant to help.
- Honesty line: *scan diagnostics prove the Mamba blocks ran (fwd/rev scan cosine distance 0.836) — the null is architectural, not a wiring fault.*

**Slide 34 — Operating point, calibration, curves [mergeable]**
- Images (3-up): `cbam_p2_pr_curve.png` · `cbam_p2_f1_conf.png` (dashed line at conf 0.16) · `cbam_p2_calibration.png`.
- Content: F₂-optimal threshold 0.16–0.20 across all configs ⇒ no per-model tuning; MSA-YOLO deploys at **0.16**. ECE ≈ 0.021 ⇒ a reported 0.8 confidence ≈ 0.8 empirical precision — operators can trust the scores for triage.

**Slide 35 — Qualitative results & where it still fails**
- Images 2×2: `qual_gt.png` (156 GT) / `qual_baseline.png` (640 px, 209 boxes) / `qual_sahi256.png` (217) / `qual_sahitta256.png` (217).
- Content: sliced/augmented modes add detections along the rubble line where the smallest, most occluded figures sit. Three residual failure types: **heavy occlusion · extreme scale (few px, no texture) · crowding (merged boxes)** — residual error concentrates in the very-tiny band.

**Slide 36 — Comparison with the state of the art**
- Caption: `TABLE IX — PUBLISHED DETECTORS ON THE C2A TEST SET (Nihal et al., same split, no slicing/TTA)`:

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

- Punchline: *second among ALL published detectors on this benchmark — one point of AP₅₀ behind a model **three times our size**. For airborne hardware, that is the right trade.*

**Slide 37 — Inference-time study: SAHI vs TTA**
- Caption: `TABLE X — SAHI SWEEP AND TTA (MSA-YOLO, per-image matching @ IoU 0.5)` (condensed report Tab. 4.7):

| Setting | Size | R | F₂ | very-tiny R | Lat. (ms) |
|---|---|---|---|---|---|
| none (baseline) | 640 | 0.835 | 0.839 | 0.758 | 15 |
| + SAHI | 256 | 0.846 | 0.848 | 0.788 | 162 |
| + SAHI | 512 | 0.837 | 0.842 | 0.763 | 66 |
| **+ TTA** | **1280** | **0.877** | **0.854** | **0.850** | **60** |

- Image (right): `per_size_recall_all_configs.png`.
- Content: **TTA@1280 is the single best inference-time setting** — +9.2 pt very-tiny recall, cheaper than any SAHI config; degrades beyond 2× training resolution. Both are optional modes — neither touches the trained weights.

**Slide 38 — Objectives achieved**
- Layout: 4 rows, each = objective (from Slide 11) → ✔ → the evidence:
  1. Additive ablation under one protocol → ✔ Table VII + waterfall.
  2. Best config within deployable budget, near SOTA → ✔ Table VIII (best very-tiny recall) + Table IX (2nd of all published, ⅓ size).
  3. Honest state-space test → ✔ +2.4 M, 2.8× latency, no gain; module verified active.
  4. Inference-time recall without retraining → ✔ TTA 0.758 → 0.850 very-tiny @ 60 ms.

---

### PART G — CLOSING (5 slides)

**Slide 39 — Thesis timeline**
- Image: Gantt crop from report PDF (**export needed**).
- Content (two terms × 13 weeks): T1 — topic selection (w1–3), literature review (w2–9), C2A preparation (w6–9), detector-family benchmark (w9–13), pre-defense (w11–13). T2 — additive ablation CBAM/P2 (w1–5), state-space exploration (w3–9), SAHI & TTA (w7–11), analysis & comparison (w8–12), documentation & defense (w9–13).

**Slide 40 — Limitations (state them before they're asked)**
- Four bullets (verbatim-condensed from Ch. VII): single semi-synthetic dataset (C2A) — transfer to fully real footage untested here · single seed per configuration — small deltas sit within run noise · medium band has only 317 instances — its estimates are noisy · desktop-GPU latency — no on-drone measurement or field trial.

**Slide 41 — Future work**
- Five bullets, first two flagged **already in progress**:
  1. **Server-side deployment** (in progress) — drone streams to a ground server; detection + flagged frames back to operator; enables the TTA@1280 mode.
  2. **Enhanced dataset with own drone imagery** (in progress) — attacks the composited-data domain gap directly.
  3. Validation on real rescue imagery — SARD, HERIDAL, VisDrone, Okutama (zero-shot + fine-tuned).
  4. Multi-seed significance testing with paired statistics for publication.
  5. A purpose-built state-space design — the null doesn't close the direction; we now have a validated protocol to test it against.

**Slide 42 — Conclusion**
- Four lines, big type:
  - The gain on C2A comes almost entirely from the **P2 detection scale** — the resolution argument was right.
  - **CBAM** contributes efficiency: attention at negative parameter cost.
  - The **state-space neck contributes nothing** at 2.8× latency — an honest, verified null.
  - **MSA-YOLO: AP₅₀ 0.853 · F₂ 0.844 · 19.6 M params · 14.6 ms** — 2nd of all published C2A detectors at ⅓ the leader's size; +9 pt very-tiny recall available via TTA when latency permits.

**Slide 43–44 — References (2 slides)**
- IEEE numbered style, ~9–10 per slide, smallest readable font. Must include: Nihal et al. 2024 (C2A) · Jocher 2024 (YOLO11) · Woo 2018 (CBAM) · Wang 2020 (ECA) · Akyon 2022 (SAHI) · Gu & Dao 2023 (Mamba) · Wang 2025 (Mamba-YOLO) · Lin 2017 (FPN, focal) · Zhu 2021 (TPH-YOLOv5, VisDrone) · Ren 2015 (Faster R-CNN) · Redmon 2016 (YOLO) · Zhao 2024 (RT-DETR) · Carion 2020 (DETR) · Lin 2014 (COCO) · Sambolek 2021 (SARD) · Božić-Štulić 2019 (HERIDAL) · Kyrkou 2020 (AIDER) · Qin 2020 (U²-Net) · Hu 2018 (SE) · Cheng 2023 (small-object survey). Pull exact entries from `Defense/draft1_30_6_26/references.bib`.

**Slide 45 — Thank you / Questions**
- Running-example detections image (`arch_detections.png`) faded as background; contact line.

---

## HIDDEN SLIDES (after Q&A slide — jump targets for anticipated questions)

**H1 — "Why YOLO11m?"** Detector-family benchmark (appendix Tab. A.1): YOLOv9-s/m/e,
v10-s/m/l, 11-s/m/l — YOLO11m best accuracy-to-size balance (0.841 mAP50 @ 38.6 MB;
v9-e better but 111.8 MB). Label "earlier protocol — internally comparable only."

**H2 — "Why CBAM and not ECA?"** Report Tab. 4.2: baseline F₂ 0.844 / small-R 0.881;
+ECA 0.844 / 0.884; +CBAM **0.849 / 0.892**. ECA faster but bought no recall. Earlier
protocol, relative comparison.

**H3 — "Is the Mamba null real?" (deep-dive)** Image `fig_training_worst.png`.
Story in 3 beats: ① first attempt invalidated — the framework's train-time rebuild
silently stripped post-init injected blocks (the model trained was CBAM+P2) → protocol
gained module-presence verification. ② corrected run: blocks verified at 6 neck layers;
epoch-28 incident (val cls loss 0.61 → 4.5, mAP collapse, 15-epoch recovery) — only the
SSM run shows this optimisation fragility. ③ scan diagnostics: fwd/rev cosine distance
0.836 ⇒ module genuinely active. Null stands on measurement.

**H4 — Layer table.** Report Tab. 3.1 verbatim (layers 0–29, bold rows 10 / 17–19 / 29)
for anyone who asks "where exactly?"

**H5 — Protocol robustness & run record.** Smoke-test gate (24 h freshness) · per-epoch
safety checkpoints + resume · OOM ladder (16→8, effective 16) · 2-s GPU telemetry ·
env manifest per run. Run times: 7.3 / 7.0 / 10.2 / 29.5 h; stop epochs 270 (cap) /
239 / 218 / 154 (fitness patience; best F₂ @ 149). Mamba's 4× wall-clock bought nothing.

**H6 — Confusion & calibration detail.** `cbam_p2_confusion.png` + `cbam_p2_calibration.png`;
ECE 0.021 / MCE / Brier available; per-config ECE 0.023 / 0.021 / 0.021 / 0.020.

**H7 — Cost & carbon.** Replication workstation ≈ BDT 285,000 (Dhaka mid-2026);
~54 GPU-hours @ 182–218 W ≈ 10.6 kWh GPU, ~20 kWh total ≈ BDT 200 electricity;
campaign carbon-accounted at 9.0 kg CO₂. No paid data/cloud/licences.

**H8 — "Why is aggregate AP flat / why lead with AP50 and F2?"** All four configs within
0.002 AP — at 8-px scale a 1-px shift swings IoU, so strict-IoU AP measures annotation
noise; AP₅₀, per-size recall and F₂ are where task signal lives. (Predicted in metrics
section BEFORE results — point at Slide 30.)

**H9 — SAHI full sweep + visuals.** Full Tab. 4.7 incl. 320/640 rows and P column;
`sahi_input_full.png` → `sahi_slice_grid.png` → `sahi_merged_detections_zoom.png`.
Why TTA beats SAHI here: C2A targets are dense and everywhere, so full-frame
super-resolution helps more than tiling; SAHI shines on sparse large frames.

**H10 — Progress since pre-defense (continuity insurance).** Two-column then/now:
pre-defense = YOLOv8 n/s/m benchmark at 50 epochs (best: v8m mAP50 0.814) + HBPA proposal;
final = detector-family sweep fixing YOLO11m (H1), four-config additive ablation at full
protocol (≤300 epochs, dual early stopping), MSA-YOLO AP₅₀ 0.853 / F₂ 0.844, honest Mamba
null, SAHI/TTA study. Use if the committee asks "what did you do in the second term?" —
it shows the pre-defense numbers were a warm-up benchmark, not the thesis result.

---

## Build order (practical)

1. Export the 3 missing PNGs (size distribution, waterfall, Gantt).
2. Build Slides 13–24 first (the progressive architecture build) — one master diagram,
   duplicated with per-stage highlights; this is the supervisor's priority.
3. Then tables (6–9, 25, 28–37) — set up ONE IEEE table style and reuse.
4. Equations: screenshot from the compiled report PDF for pixel-perfect match, or retype
   in PowerPoint's equation editor (both acceptable; be consistent).
5. Hidden slides last — mostly copy-paste from report tables.

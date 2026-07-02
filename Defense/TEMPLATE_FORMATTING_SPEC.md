# CSE-4000 Thesis Template — Formatting Specification

> Source: `CSE-4000-Final (Template).pdf` (KUET, Dept. of CSE) + `Thesis_instruction_Masud_Sir.pdf`.
> This is the **authoritative formatting reference**. When writing any chapter, match this exactly.
> Page-by-page screenshots live in `Defense/Template Images/page (1..12).png` — peek there to verify.

---

## 0. Global page setup

| Property | Value |
|---|---|
| Paper | A4 |
| Font (everywhere) | Times New Roman |
| Left margin | 1.2" |
| Right margin | 1.0" |
| Top margin | 1.2" |
| Bottom margin | 1.0" |
| Header | 0.5" |
| Footer | 0.4" from bottom |
| Body line spacing | 1.5 |
| Body alignment | Justified |
| Paragraph spacing | space before 12 pt, after 6 pt |

**Page numbering:**
- Front matter (Title page → List of Figures): lowercase **Roman** (i, ii, iii, …), centered, footer.
- Main body (Chapter I onward): **Arabic** (1, 2, 3, …), centered, footer.

**Case-style rule (from instruction guide):**
- **First/cover page only:** headings in **SMALL CAPS**.
- **All subsequent pages:** Standard Capitalization (Title Case) for headings and body.

---

## 1. Cover Page (page 1, "Top Page") — SMALL CAPS

- Top-left: `Project/Thesis No.:` — 12 pt, Normal, Left.
- `CSE 4000: Thesis/ Project` — 14 pt, Normal, Centered.
- **Title** — Times Roman, **Bold, Centered, SMALL CAPS, 18 pt**, 1.5 spacing.
  - 2 blank lines (size 12, 1.5 spacing) between Title and "By".
- `By` — 14 pt, Normal, Centered.
- 3 blank lines (size 12).
- Author name — 14 pt, **Bold**, Centered; `Roll: XXXXXXX` — 14 pt, Normal, Centered.
  - `&` between multiple authors. (This thesis = single author.)
- **KUET logo** — 1.14" × 1.0", centered.
- 2 blank lines (size 12).
- Footer block (12 pt, **Bold**, Centered, 1.2 spacing, at bottom of page):
  ```
  Department of Computer Science and Engineering
  Khulna University of Engineering & Technology
  Khulna 9203, Bangladesh
  <Month, Year>
  ```

---

## 2. Title Page (page i) — Standard case

⚠️ **INSTRUCTION FLAG:** Masud Sir's guide says *"Check page 2 of your template immediately. A line is known to be missing there; locate and manually restore it."* — This is the Title Page. Likely missing: the **`Roll: <number>` line** or the degree line. Verify against a correct copy before finalizing.

- **Title** — Times Roman, Bold, Centered, **18 pt** (Standard case, NOT small caps), 1.5 spacing.
- 3 blank lines (size 12) between Title and "By".
- `By` — 12 pt, Normal, Centered.
- 2 blank lines (size 12).
- Author name — 12 pt, **Bold**, Centered; `Roll: XXXXXXX` — 12 pt, Normal, Centered.
- `A thesis submitted in partial fulfillment of the requirements for the degree of`
  `"Bachelor of Science in Computer Science & Engineering"` — 12 pt, Normal, Centered.
- 2 blank lines (size 12).
- Left block:
  - `Supervisor:` — 12 pt, **Bold**, Left aligned.
  - Supervisor name — 12 pt, **Bold**, 0.8" left indent.
  - `Professor` / `Department of Computer Science and Engineering` / `Khulna University of Engineering & Technology` — 12 pt, Normal, 0.8" left indent.
- Right block: `Signature` line (underscore + label), right side.
- Footer block (Times Roman, **Bold, Centered, 16 pt**, 1.5 spacing):
  ```
  Department of Computer Science and Engineering
  Khulna University of Engineering & Technology
  Khulna 9203, Bangladesh
  <Month, Year>
  ```
- Page number `i`, footer, 0.4" from bottom.

---

## 3. Acknowledgment (page ii)

- Heading `Acknowledgment` — Bold, Centered (16 pt to match other front-matter headings).
- 2 blank lines (size 12).
- Body — Times Roman, Normal, **justified**, 12 pt, 1.5 spacing.
- `Authors` (or `Author`) — 12 pt, **Bold**, Right aligned, at end.

---

## 4. Abstract (page iii)

- Heading `Abstract` — Times Roman, **Bold, Centered, 16 pt**, 1.5 spacing.
- 2 blank lines (size 12).
- Body — Times Roman, Normal, justified, 12 pt, 1.5 spacing. **≤ 500 words.**

---

## 5. Contents / List of Tables / List of Figures (pages iv–viii)

**Contents (`Contents` heading centered, bold):**
- Chapter-level entries style: 12 pt, **Bold**, Left, 1.5 spacing.
- Within-chapter entries: Times New Roman, Normal, 12 pt, 1.5 spacing, 0.5" left indent.
- Right column header `PAGE`; dotted/aligned page numbers on the right.
- Front-matter entries listed: Title Page (i), Acknowledgment (ii), Abstract (iii), Contents (iv), List of Tables (vii), List of Figures (viii).

**List of Tables (page vii) & List of Figures (page viii):**
- Heading — Times Roman, **Bold, Centered, 16 pt**, 1.5 spacing.
- 2 blank lines (size 12).
- Three columns: `Table No.` / `Figure No.` | `Description` | `Page`. Header row bold.
- Entries — Times Roman, 12 pt, 1.5 spacing.
- Numbering is **hierarchical: <Chapter>.<Sequence>** (e.g., 2.1, 3.1, 4.1).

---

## 6. Chapter opening pages (body, Arabic numbering)

- `CHAPTER I` (roman chapter number) — Times Roman, **Bold, Centered, 14 pt**, 1.5 spacing.
  - 1 blank line (size 12).
- Chapter title, e.g. `Introduction` — Times Roman, **Bold, Centered, 16 pt**, 1.5 spacing.
  - 2 blank lines (size 12).

**Section headings (`1.1`, `1.2`, …):** Times Roman, **Bold, Left**, 1.5 spacing, space before 12 pt / after 6 pt.
- ⚠️ **SIZE DISCREPANCY IN TEMPLATE:** the box next to `1.1 Introduction` says **14 pt**, but the box next to `1.3 The Realization…` says **12 pt**. The repeated/consistent value is **12 pt** — use **12 pt bold** for section headings unless supervisor says otherwise. Flag for confirmation.

**Subsection headings (`1.3.1`, …):** Times Roman, **Bold, Left**, 12 pt (slightly smaller weight than section, but bold).

**Body paragraphs:** Times Roman, Normal, **justified**, 12 pt, 1.5 spacing, space before 12 pt / after 6 pt.

**Text-hierarchy rule (instruction guide):** every chapter/section/subsection — including *Objectives* — must **open with an introductory paragraph**, never start directly with a bullet or numbered list.

---

## 7. Figures & Tables

- **Placement:** strictly **top or bottom** of a page — never mid-paragraph.
- **No outer borders/boxes** around figures or text blocks.
- **Numbering:** `Figure <Chapter>.<Seq>` and `Table <Chapter>.<Seq>` (e.g., Figure 3.2 = 2nd figure of Ch. 3). **Never** `Figure 3.2.2`.
- **Caption style:** Times Roman, Normal, **Centered, 11 pt**, 1.5 spacing, space before 12 pt / after 6 pt.
  - **Figure caption → BELOW the figure.**
  - **Table caption → ABOVE the table.**
  - Captions must be **descriptive/context-rich** (not "Figure 3.2: Model").
- **Sub-figures:** label `(a)`, `(b)`; reference explicitly in caption + body.
- **In-figure typography:** fonts/scales must match the main document.
- **Large diagrams:** scale horizontally; if needed use A3 fold or break into labeled subsections.

---

## 8. Equations

- Every standalone equation **right-aligned with a unique sequence number**, e.g. `(3.1)`.
- **Symbol matching:** a variable italicized in an equation (e.g. *y*, *yᵢ*) must appear identically in body text — never plain `y` or `y_i`.
- Use proper operators (`cos ψ`, `sin θ`), not unformatted text.

---

## 9. References (IEEE style)

- Heading `REFERENCES` — Times New Roman, **Bold, Centered, 14 pt**, 1.5 spacing.
- 2 blank lines (size 12).
- Entries — Times Roman, Normal, justified, **12 pt, 1.2 spacing**, space before 6 pt / after 6 pt.
- Numbered list, **left aligned; paragraph left indent = 0, hanging indent = 0.31"**.
- **IEEE format** strictly. Majority = peer-reviewed journals/conference papers.
- Consistent conference locations; never omit volume/issue/page spans; include **DOI** where available.
- See `Defense/IEEE Reference Style Guide for Authors.md`.

---

## 10. Full chapter structure (from template Contents)

| Ch. | Title | Key sections |
|---|---|---|
| I | Introduction | Introduction · Background/Problem statement · Objectives · Scope · Unfamiliarity of the problem · Project planning & work distribution (Gantt/RACI — single author, note accordingly) · Applications · Organization of thesis |
| II | Literature Review | Introduction · Literature Review (thematic grouping, not paper-by-paper) · Discussion (Research gap solution) |
| III | Methodology | Introduction · **Overall Framework/Block Diagram FIRST** · Detailed methodology (Problem design & analysis, stages…) · Conclusion. Keep proposed architecture separate from baselines; baseline (YOLO variant) explained in Ch. II Related Work / Core Concepts. |
| IV | Implementation, Results and Discussions | Introduction · Experimental Setup (exact HW/SW specs) · Evaluation Metrics (problem-specific) · Dataset (samples, splits, before/after augmentation) · Implementation & Results (Quantitative + Qualitative/error analysis + Analysis) · Objective Achieved · Financial Analyses & budget · Conclusion. Include **SOTA comparison** (formal label, not "base paper"). |
| V | Societal, Health, Environment, Safety, Ethical, Legal and Cultural Issues | IP · Ethical · Safety · Legal · Societal/Health/Cultural impact · Environment & Sustainability |
| VI | Addressing Complex Engineering Problems and Activities | Complex engineering problems · Complex engineering activities |
| VII | Conclusions | Summary (one paragraph) · Limitations · Recommendations & Future Works |
| — | Appendices (if any) | |
| — | References | IEEE |

**Fixed sections:** items shown in *italics* in the template Contents are mandatory/fixed; supervisor may extend but not remove.

---

## 11. This thesis — fixed identity info

| Field | Value |
|---|---|
| Author | Md Mofazzal Hosen |
| ID / Roll | 2007074 |
| Batch | 2k20 |
| Supervisor | Sk. Md. Masudul Ahsan |
| Supervisor title | Professor, Computer Science & Engineering, KUET |
| Supervisor email | smahsan@cse.kuet.ac.bd |
| Department | Computer Science and Engineering |
| University | Khulna University of Engineering & Technology, Khulna 9203, Bangladesh |
| Degree | Bachelor of Science in Computer Science & Engineering |
| Topic area | YOLO + Mamba for aerial (drone) human detection — Search & Rescue |

---

## 12. Key deadlines (instruction guide)

- **Initial draft:** July 2, 2026.
- **Final report submission:** July 8, 2026 (3 days before defense).
- **Presentation slides:** finalized 3 days before defense.
- **Live demo:** run-ready reproducible codebase + polished GUI (Gradio/Streamlit) + out-of-distribution real-world phone-photo test.

---

## 13. Additional supervisor instructions (draft-1 review, 2026-07-02)

Rules added from direct supervisor feedback. Where wording was vague, my interpretation is marked **[interpreted]** — confirm with supervisor if unsure.

**Figures & layout**
- **No figure boundary.** Never draw a border/frame/box around a figure or its caption. (Reinforces spec §7.)
- **No unnecessary gaps — and watch page splits.** Eliminate accidental vertical whitespace. Page breaks + float placement are a common source of large gaps: place floats `[t]`/`[b]`, avoid `[h]`, and don't pad with manual `\vspace`. If a float pushes a big gap, move it, don't stretch text. **[interpreted]** Never use stacked `\vspace` to position content (the anti-pattern seen in the reference file).
- **A figure must never split across two pages.** Keep each figure whole on one page; if it's too tall, scale it down or move it to the top/bottom of the next page (never let it break). Large diagrams → scale horizontally / A3 fold / split into labeled sub-diagrams (spec §7).
- **Figures/tables at top or bottom of the page** (not mid-paragraph). Marked *optional* by supervisor but treat as the strong default.
- **Figure numbering = `<chapter>.<figure-in-chapter>`** e.g. `Figure 3.2` (2nd figure in Ch. 3). Never `3.2.2`. Same for tables.
- **Captions handled properly:** figure caption BELOW, table caption ABOVE; 11 pt, centered; descriptive/context-rich (not one/two words). Sub-figures labeled (a),(b) and referenced in text.

**Text & bullets**
- **Objectives (and every section): intro text first, THEN bullets.** Never open with a list.
- **No unnecessary bullets.** Use bullets only to enumerate or call out discrete points; otherwise write prose. Don't bullet-ify normal paragraphs.

**Math**
- **Symbol consistency:** any symbol used in an equation must appear identically (same glyph/italics/subscript) everywhere it's mentioned in prose, tables, or other equations.

**Datasets, models & results**
- **Dataset samples must cover all cases.** Show representative samples spanning every class/scenario/condition in the dataset (not just easy examples) — 2–3 per case, plus splits (train/val/test) and before/after augmentation stats (spec §6).
- **Document selection rationale** in results **[interpreted — supervisor note "matrix's dataset selection? (model selection, hyperparameter, epoch, curve)"]**: justify *model selection*, *hyperparameter choices*, *number of epochs*, and include the relevant *curves* (training/val loss, PR/F-curves). Don't present final numbers as if chosen arbitrarily — show the basis.

**Modifiable figures — hard requirement (supervisor statement)**
- **Every figure/diagram I create or need remade must be delivered as editable source, NOT a flattened image.** Deliverable = **draw.io (diagrams.net) XML** so the user can open it, inspect it, and regenerate it in PowerPoint/Word. The user generates final diagrams from PowerPoint/Word or LaTeX/TikZ — never third-party rasterized images. Supervisor requires diagrams to remain modifiable.

**Tables & references**
- **Tables:** follow IEEE table formatting, or the template's style if the template specifies it. Booktabs-style rules, caption above, no vertical clutter.
- **References:** IEEE (spec §9). Verify each citation against a real source — do not fabricate. Prefer peer-reviewed journals/conferences; include DOI.

**Working principle (applies to all content I write)**
- **Verify, don't guess.** Before stating a fact about the thesis (results, config, methods, dates), find it in the thesis-folder resources. Only if it genuinely isn't recorded anywhere may I make a best-effort inference — and I must label it as such. External web sources allowed for general/background facts; record any such reliance.

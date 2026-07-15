# Verified Template Spec — from all 12 Template Images (audited 2026-07-15)

Source: `Defense/Template Images/page (1..12).png`. This is the authority for
formatting. Where the USER explicitly overrode the template, the override wins
(noted inline). Re-read this before editing any chapter.

## Page geometry / global
- Margins: left 1.2in, right/top/bottom per KUET spec (top 1.2, right 1.0, bottom 1.0). headsep 0.25in, footskip 0.4in.
- Body: Times New Roman, 12pt, justified, **1.5 line spacing**.
- Paragraph: block (no indent), space-before 12pt / after 6pt (approx via parskip).
- Footer: page number centered.

## Front matter (roman numerals)
- **Cover (p1):** title small-caps, Capitalize-Each-Word, 18pt bold centered. "Project/Thesis No.:" 12pt bold left. [DONE]
- **Title page (p2):** title standard title-case (minor words lower), 18pt.
- **Acknowledgment (p3):** heading "Acknowledgment" 16pt bold centered, 2 blank(12) after; body 12pt justified 1.5; ends with **"Author"** 12pt bold **right**. (template shows "Authors" plural for 2 students — single author here → "Author".)
- **Abstract (p4):** heading "Abstract" 16pt bold centered, 2 blank(12) after; body 12pt justified 1.5.
- **Contents (p5-7):** "Contents" 16pt bold centered; "PAGE" right header; front-matter entries 0.5in indent 12pt; chapters bold. Template shows bullets/arrows for sections/subsections → **USER OVERRODE to numeric X.Y / X.Y.Z, no dot leaders.**
  - **TOC ORDER: Appendices come BEFORE References** (template: Appendices 60, References 62).
- **List of Tables (p8):** "List of Tables" 16pt bold centered; 2 blank(12); 3-col header "Table No. | Description | Page" bold; entries 12pt 1.5.
- **List of Figures (p9):** same as LoT, "Figure No. | Description | Page".

## Chapter / section headings (p10)
- Chapter: "CHAPTER I" 14pt bold centered → 1 blank(12) → title 16pt bold centered → 2 blank(12).
- Section (X.Y): 14pt bold left, space before 12pt after 6pt.
- Subsection (X.Y.Z): 12pt bold left, space before 12pt after 6pt.
- Body paragraphs justified 12pt 1.5.

## Figures & Tables (p11) — CRITICAL
- **"Each figure or table should be on top or bottom of the page."** → floats use [tbp] only.
- **Figure caption BELOW** the figure. **Table caption ABOVE** the table.
- Caption style: Times Roman, **normal, 11pt, 1.5 line space, space before 12pt / after 6pt.**
  - Template shows captions centered. **USER OVERRIDE: caption on 2+ lines = LEFT aligned; 1 line = centered.** (caption pkg: singlelinecheck=true + justification=raggedright.)
- Caption names must be concise/generic (IEEE style, no leading A/An/The); explanation goes in body paragraph above/below. [DONE this session]

## References (p12)
- Heading "REFERENCES" **14pt bold Times centered**, 2 blank(12) after.
- Entries: Times 12pt justified, **1.2 line space**, space before 6pt/after 6pt, hanging indent 0.31in, left indent 0. [IEEE style via biblatex]

## USER fix-list for this pass (2026-07-15)
1. No two figures dumped back-to-back without explanation between; every float gets a discussing paragraph. Curate thesis_folder for real detail.
2. Remove per-column-max bold; **bold only the final/recommended result cells** (tab:main → CBAM+P2 row; appfamily → selected YOLO11m row).
3. Kill unnecessary gaps and bottom-of-page gaps.
4. 2-line caption → left aligned; else centered.
5. Avoid a near-blank page holding one lone float before a chapter start — grow/trim adjacent explanation to backfill (soft rule; never break margin/heading spec).
6. Work ONE chapter at a time; check template first (this file).

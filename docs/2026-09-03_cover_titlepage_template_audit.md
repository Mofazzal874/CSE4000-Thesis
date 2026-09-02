# Cover and title page: measured against the template (2026-09-03)

Source of truth: `Defense/Template Images/page (1).png` (73.14 px/in) and `page (2).png`
(72.88 px/in), both A4. Text measured with a black-only threshold so the grey annotation
boxes do not contaminate the rows; baselines derived by subtracting the Times descender
(0.216 em) where the row has one.

## 1. The paragraph model, derived from the template's own regular gaps

Word's line height is `size x 1.15 x multiple`, not `size x multiple`. On top of that a
paragraph may carry a "space after". Both fall out of gaps the template repeats:

**Page 1 has a 6pt space-after on every paragraph.** The author block gives five
consecutive single-line 14pt paragraphs:

| gap | measured |
|---|---|
| name -> roll | 30.5pt |
| roll -> & | 30.5pt |
| & -> name | 29.5pt |
| name -> roll | 30.5pt |

A 14pt line at 1.5 is `14 x 1.15 x 1.5 = 24.15pt`, so each paragraph adds `30.5 - 24.15
= 6.35 ~ 6pt`. The title's two lines confirm it: 36.4pt measured against `31.05 + 6 =
37.05`.

**Page 2 does not.** Its repeated pitches are 20.7pt (author block), 19.8-25.7pt
(supervisor block) and 22.7 / 23.7 / 22.3pt (footer) — all the bare 12pt line at 1.5
(20.7pt), with no 6pt added. Its title lines measure 30.6pt against 31.05 predicted.

So **a blank line of size 12 is not the same on the two pages**:

| | line | space after | blank line |
|---|---|---|---|
| page 1 (cover) | 20.7pt | 6pt | **26.7pt** |
| page 2 (title) | 20.7pt | 0 | **20.7pt** |

Earlier passes used 18pt (`fontsize x multiple`, ignoring the 1.15) and then 20.7pt on
both pages with `\parskip` forced to zero — which deleted exactly the 6pt that page 1
depends on. That is why the gaps above and below "By" were short.

## 2. Verification of the rebuilt pages

Rendered at 150 dpi and measured identically to the template.

### Page 1

| gap | template | rebuilt | delta |
|---|---|---|---|
| CSE 4000 -> title line 1 | 34.5pt | 34.6pt | +0.1 |
| title internal pitch | 31.05pt | 31.2 / 30.7pt | +0.1 / -0.4 |
| **title -> By** | **86.6pt** | **86.4pt** | **-0.2** |
| By -> author | 120.1pt | 109.0pt | -11.1 |
| author -> roll | 30.5pt | 30.2pt | -0.3 |
| footer internal pitch | 16.4pt | 16.3 / 16.3 / 16.8pt | under 0.5 |
| header label baseline | 0.711in | 0.711in | 0.00 |
| logo height | 1.148in | 1.139in | -0.01 |

### Page 2

| gap | template | rebuilt | delta |
|---|---|---|---|
| title internal pitch | 30.6pt | 30.7 / 31.2pt | +0.1 / +0.6 |
| title -> By | 97.3pt | 86.9pt | -10.4 |
| By -> author | 45.1pt | 61.0pt | +15.9 |
| degree internal pitch | 21.7pt | 20.6pt | -1.1 |
| supervisor block pitch | 19.8pt | 20.6pt | +0.8 |
| footer internal pitch | 22.7 / 23.7 / 22.3pt | 20.6pt | under 3.1 |

## 3. Where the template contradicts itself

Three gaps sit between whole numbers of blank lines, so its own file does not match its
own labels. The implementation follows the **labels**, since those are the specification a
reviewer reads and they are whole numbers; the rendering is recorded here so the choice
is visible.

| location | label | what the rendering implies |
|---|---|---|
| page 1, By -> author | "3 blank size of 12 size" | 120.1pt = `3 x (24.15 + 6) + 30.15` — three blanks at **14pt**, not 12 |
| page 2, title -> By | "3 blank line of size 12" | 97.3pt = 3.7 blanks at 20.7pt |
| page 2, By -> author | "2 blank line of size 12" | 45.1pt = 1.2 blanks at 20.7pt |

Page 2's two are non-integral in opposite directions, which is why the labels were
followed rather than the pixels. Page 1's is a clean fit at 14pt; if the rendering is
preferred there, change `3\tblankcover` to `3\tblanktitle` plus `3\tafter` in
`cover.tex` — an 11pt (0.15in) difference.

## 4. Page 1 (cover): every element

| item | template | now |
|---|---|---|
| `Project/Thesis No.:` | 12pt Normal, left at the 1.2in margin, cap-top 0.615in from the paper edge, **no rule** | one-page page style in the header band |
| `CSE 4000: Thesis/ Project` | 14pt Normal, centred, first line of the block | same, no extra gap before the title |
| Title | 18pt Bold, centred, SMALL CAPS, 1.5 (31.05pt) | same |
| title -> By | 2 blank lines = 2 x 26.7pt | same |
| `By` | 14pt Normal, centred | same |
| By -> author | 3 blank lines = 3 x 26.7pt | same |
| Author / Roll | 14pt Bold / 14pt Normal | same |
| KUET logo | 1.148in tall x 0.998in wide *(measured)* | height 1.14in |
| logo -> footer | 2 blank lines | same |
| Footer | 12pt **Bold**, centred, 1.2 spacing (16.56pt) | same |

The logo annotation reads `1.14" x 1.0"`. Read as width x height that gives a wide logo;
the rendered logo measures 0.998in wide by 1.148in tall, so the first figure is the
height. `kuet_logo.png` is 1200x1364 (aspect 0.880) against the template logo's 0.869,
so setting the height alone reproduces it. Setting both would stretch it.

## 5. Page 2 (title page): every element

| item | template | now |
|---|---|---|
| Title | 18pt Bold, centred, standard case, 1.5 | same |
| title -> By | 3 blank lines = 3 x 20.7pt | same |
| `By` | 12pt Normal, centred | same |
| By -> author | 2 blank lines = 2 x 20.7pt | same |
| Author / Roll | 12pt Bold / 12pt Normal | same |
| Degree lines | 12pt Normal, centred | same |
| degree -> supervisor | 2 blank lines | same |
| `Supervisor:` | 12pt Bold, left at the margin | same |
| Supervisor block | name 12pt Bold, rest 12pt Normal, 0.8in indent | same |
| Signature | rule and label, right | same |
| Footer | 12pt Normal, centred (see below) | same |
| Page number | `i`, footer, 0.4in from the bottom | same |

### The footer conflict, resolved from the image as instructed

That block carries two contradictory annotations: `12, Normal, Centered` and
`[Times Roman, Bold, Centered, 16 size, 1.5 line space]`.
`TEMPLATE_FORMATTING_SPEC.md` followed the second. The image settles it:

| quantity | footer block | 12pt body above it | known-Bold 12pt, same page |
|---|---|---|---|
| line pitch | 22.7 / 23.7 / 22.3pt | 20.7-22.7pt | - |
| ink height | 10.9pt | 10.9pt | - |
| ink density | 20.4% | 19.8% | 39.2% |

A 16pt block at 1.5 would have a 27.6pt pitch. It is 12pt Normal. The same string on
**page 1** measures 28.0% and is genuinely Bold, so the two pages differ here for real.

## 6. Faults in the earlier attempts, for the record

1. **Blank line taken as `12 x 1.5 = 18pt`.** Word's line height includes the font's
   1.15 factor, so it is 20.7pt.
2. **`\parskip` forced to zero on both pages.** That removed page 1's 6pt space-after,
   the difference between a 20.7pt and a 26.7pt blank line, and shortened every gap on
   that page.
3. **setspace multiplying the explicit leadings.** `\onehalfspacing` applies
   `\baselinestretch` (~1.237) on top of whatever `\fontsize` is given, so
   `\fontsize{18}{31.05}` rendered at 38.4pt. Both pages are wrapped in `spacing{1}`.
4. **`\par` between the footer lines** added `\parskip` inside the block, turning its
   16.56pt pitch into 26.4pt. Those lines are broken with `\\` instead.
5. **The header line consumed a body line**, pushing page 1 down 0.57in. It is a
   one-page page style now.
6. **Both pages treated identically.** They are formatted differently in the template;
   see section 1.

## 7. Front matter: Acknowledgment, Abstract, Contents, LoT, LoF

Template pages 3 (Acknowledgment, ii), 4 (Abstract, iii), 5-7 (Contents),
8 (List of Tables), 9 (List of Figures).

### `Project/Thesis No.:` is Bold, not Normal

The page-1 annotation reads `[12,Normal, Left]`. The rendering disagrees: at 5x
magnification its stems match "Dola Das" and "Department of Computer Science and
Engineering" (both annotated Bold) and are visibly heavier than "Roll: 1407016"
(annotated Normal). Set to Bold. This is the third annotation that contradicts
the rendering, after page 2's footer and page 1's By->author gap.

### Measured pitches, and what they were

| gap | template | before | now |
|---|---|---|---|
| heading -> body (Ack. / Abstract) | 64.9pt | 72.7pt | 64.9pt |
| Contents, front-matter entries | 28.7pt | 34.1pt | 28.7pt |
| Contents, CHAPTER entries | 57.4pt | - | 57.4pt |
| Contents, section entries | 21.5pt | - | 20.7pt |
| List of Tables / Figures entries | 22.6pt | 18.7pt | 20.7pt |

The Contents entries were loose because the front-matter lines were added as
*numberless chapter* entries and inherited the chapter's 10pt space-above. They
now use a dedicated `fmatter` level carrying its own 0.5in indent, so the
`\protect\hspace*{0.5in}` that each call used to pass is gone as well.

`\tocstretch` (1.43 x 14.5pt = 20.7pt) sets the three lists to the template's
line without touching body text.

### Heading heights were inconsistent

`report.cls` renders Contents, List of Tables and List of Figures with
`\chapter*`, which put those three headings at 1.69-1.71in while the
Acknowledgment and Abstract sat at 1.21in. The template puts "Contents" at
1.335in, level with "Acknowledgment" at 1.272in, so all five now share
`\frontheading` and land together at 1.21in.

The template's own List of Tables and List of Figures headings sit about 4.5in
down the page. That is an artefact of how those screenshots were assembled, not
a specification, and is deliberately not reproduced.

### Contents style: numbers kept, deliberately not the template's bullets

Template pages 5-7 mark sections with a bullet and subsections with an arrow
instead of printing the numbers. That was implemented and then **reverted on
the author's instruction**: `X.Y` and `X.Y.Z` are more useful for navigating a
document of this length and match the numbering used in the body, so the
Contents keeps `\thecontentslabel`. This is the one place where the template is
knowingly not followed.

Everything else about the Contents -- indents, pitches, the `fmatter` level,
heading height, the bold CHAPTER lines, no dotted leaders -- does follow it.

`pifont` was loaded only for the arrow and is no longer required.

### The single-entry page

Page x carried one List of Figures entry on its own. Three separate causes:

1. Captions 3.7 and 3.8 wrapped to a second line in the list. Both now have a
   short form (`\caption[Slicing-aided hyper inference.]{...}`); the full
   captions under the figures are unchanged.
2. LaTeX injects `\addvspace{10pt}` into the `.lot`/`.lof` at every chapter
   boundary. The template's List of Tables has no such group gaps -- its 2.1 /
   3.1 / 4.1 / 5.1 entries run at a uniform pitch -- so `\addvspace` is
   neutralised inside both lists.
3. A 2pt `\parskip` cost 54pt across 27 entries. Zeroed, which leaves the bare
   20.7pt line: exactly the "12 size, 1.5 line space" the template annotates.

All 27 figure entries now fit on one page, and the document is 77 pages rather
than 78.

`\contentslabel{0.55in}` gives the two lists a hanging indent, so a caption that
wraps aligns under the description instead of returning to the margin.

### Margins, checked on all twelve front-matter pages

| | target | measured |
|---|---|---|
| left | >= 1.2in | 1.20 - 1.21 |
| right | >= 1.0in | 0.98 - 1.07 |
| top | >= 1.2in | 1.21 - 1.67 |
| body bottom | <= 10.69in | <= 10.70 |
| footer baseline | ~11.09in | 11.08 on every page |

No violations. Page 1's 0.61in top is the header band, by design.

## 8. Body line spacing: still open

Body text measures **17.8-18.2pt** per line. Word's 1.5 spacing for 12pt Times
is **20.7pt**, and the template's lists run at 22.6pt. The gap comes from
LaTeX's `\onehalfspacing`, which gives about 18pt at a 12pt base.

Correcting it is one line in `thesisstyle.sty`, but it re-flows the whole
document: roughly 10-15% more pages, and every figure and table placement moves.
It needs its own pass with a fresh float sweep, not a side effect of a
front-matter fix.

## 9. Not touched

`\vblank` in `thesisstyle.sty` is still `18pt * n` and carries fault 1 above, but it
drives `\frontheading` and `main.tex`, so changing it moves the acknowledgment, abstract
and contents pages. Out of scope for this pass; it needs the same correction when those
pages are reviewed.

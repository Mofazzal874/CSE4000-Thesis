# Per-page audit: template image → instruction → did my LaTeX follow it?

Checked against `Defense/Template Images/page (N).png`. Status: ✅ ok · ❌ was wrong (fixing) · ⬜ pending your asset.

## page 1 — Cover (Top Page)
| Instruction (from box) | Followed? |
|---|---|
| `Project/Thesis No.:` top-left, 12 Normal Left | ✅ |
| `CSE 4000: Thesis/ Project` 14 Normal Centered (above title) | ✅ |
| Title 18 Bold Centered SMALL CAPS, 1.5; 2 blank(12) title→By | ❌ spacing uneven → fix to exact 36pt |
| `By` 14 Normal Centered; 3 blank(12) | ❌ was 14; spacing → fix |
| Author 14 Bold Centered; `Roll:` 14 Normal Centered | ✅ (sizes fixed) |
| KUET logo 1.14"×1.0" centered | ❌ distorted (forced W&H) → keepaspectratio |
| 2 blank(12) then dept block 12 Bold Centered 1.2 at bottom | ✅ |

## page 2 — Title Page (i)
| Instruction | Followed? |
|---|---|
| Title 18 Bold Centered (standard case, not small caps), 1.5 | ✅ |
| 3 blank(12) title→By; By 12 Normal Centered | ❌ spacing → fix to 54pt |
| 2 blank(12); Author 12 Bold, Roll 12 Normal Centered | ✅ |
| degree lines 12 Normal Centered | ✅ |
| 2 blank(12) | ✅ |
| `Supervisor:` 12 Bold left; details 12 (name bold) at 0.8" indent | ❌ minipage too narrow → hyphenated "En-gineering" → widen + no hyphenation |
| Signature line on right | ❌ layout off → realign |
| bottom dept block 16 Bold Centered 1.5 | ✅ (note: title-page dept block is 16pt, cover is 12pt) |
| footer `i`, 0.4" from bottom | ✅ |

## page 3 — Acknowledgment (ii)
Heading bold centered; 2 blank(12); body Times Normal justified 12, 1.5; `Author(s)` 12 bold right. ✅ (heading size aligned to 16)

## page 4 — Abstract (iii)
`Abstract` 16 Bold Centered 1.5; 2 blank(12); body Normal justified 12, 1.5; ≤500 words. ✅

## page 5–7 — Contents (iv–…)
| Instruction | Followed? |
|---|---|
| `Contents` bold centered; 2 blank(12) | ❌ heading present but no spacing header |
| `PAGE` right-aligned column header | ❌ missing → add |
| Front-matter entries (Title Page, Acknowledgment, Abstract, Contents, List of Tables, List of Figures), Normal 12, 0.5" indent, roman page nos | ❌ missing entirely → add (auto page nos) |
| Chapters: `CHAPTER I  Introduction` bold + page | ❌ was "1 Introduction" → fix to "CHAPTER I …" |
| Sections shown as **bullets (•)**, NOT numbers | ❌ was "1.1 …" numbered → fix to bullets |
| Sub-items shown as **arrows (➢)** | ❌ → fix |
| dotted leader to page number | ❌/❓ → add dotted leaders |
| within-chapter entries Times Normal 12, 1.5 | ✅ after fix |

## page 8 — List of Tables (vii)
`List of Tables` 16 Bold Centered 1.5; 2 blank(12); columns `Table No. | Description | Page` (header row); entries Times 12, 1.5, number = chapter.seq. ❌ column header row missing → add.

## page 9 — List of Figures (viii)
Same as LoT with `Figure No. | Description | Page`. ❌ header row → add.

## page 10 — Chapter body start (1, arabic)
| Instruction | Followed? |
|---|---|
| `CHAPTER I` 14 Bold Centered 1.5; 1 blank(12) | ✅ |
| Chapter title 16 Bold Centered 1.5; 2 blank(12) | ✅ |
| Section `1.1 …` Bold left (box says 14 at 1.1, 12 at 1.3 — inconsistent) | ✅ using 12pt bold; sections show arabic `3.1` |
| body Times Normal justified 12, 1.5, space before 12 after 6 | ✅ |
| subsection `1.3.1 …` bold | ✅ |
| arabic page numbers centered footer | ✅ |

## page 11 — figures/tables in body
Figure caption BELOW, Table caption ABOVE, 11pt centered; number = chapter.seq; top/bottom placement; no borders. ✅

## page 12 — References
`REFERENCES` 14 Bold Centered 1.5; 2 blank(12); entries IEEE, 12, 1.2, hanging 0.31". ✅

## Numbering scheme decision (to satisfy both template rules at once)
- Chapter heading + TOC show **Roman** ("CHAPTER III"): `\thechapter = \Roman`.
- Body sections show **arabic** ("3.1"): `\thesection = \arabic{chapter}.\arabic{section}`.
- Figures/tables/equations show **arabic** ("Figure 3.2"): `\the{figure,table,equation}` forced to `\arabic{chapter}.…`.

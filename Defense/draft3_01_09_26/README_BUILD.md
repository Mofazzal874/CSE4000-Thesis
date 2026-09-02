# Build instructions — Defense report

Self-contained project (real Times New Roman bundled in `fonts/`, IEEE refs via biblatex).

## Overleaf (primary)
1. Zip the **entire `draft3_01_09_26/` folder** and upload to a new Overleaf project
   (New Project → Upload Project). Keep the folder structure intact.
2. Menu (top-left) → **Compiler = XeLaTeX**, **Main document = main.tex**,
   **TeX Live version = latest**.
3. Compile. Biber runs automatically for the IEEE references.

Fonts work on Overleaf with no install because `thesisstyle.sty` loads them from
`fonts/` via `fontspec` (`Path = fonts/`).

## Local (WSL replica — for when Overleaf hits its free-tier limit)
```bash
cd draft3_01_09_26
bash build.sh        # xelatex -> biber -> xelatex -> xelatex -> main.pdf
```
Requires `xelatex` + `biber` (TeX Live full, or `texlive-xetex texlive-bibtex-extra biber`).

## Files
- `main.tex` — master; sets page numbering (roman front matter → arabic body) and pulls in everything.
- `thesisstyle.sty` — all formatting from `../TEMPLATE_FORMATTING_SPEC.md`.
- `fonts/` — Times New Roman .ttf ×4 (do not remove).
- `frontmatter/` — cover, titlepage, acknowledgment, abstract.
- `chapters/` — 01–08.
- `figures/` — figures/plots (copied in; project stays self-contained).
- `tikz/` — figures drawn natively in TikZ (Gantt, system overview, size distribution,
  ablation waterfall). Bar fills carry a hatch as well as a colour so the series stay
  separable in greyscale and for readers with colour-vision deficiency.
- `scripts/` — regeneration scripts for figures built from measured data, so the numbers
  in a plot can be traced back to the evaluation record rather than trusted:
  - `make_per_size_recall.py` → `figures/per_size_recall_all_configs.png` (Figure 4.8),
    read from `../demo/results/metrics/grand_summary.json`.
- `references.bib` — IEEE bibliography.

# Editorial review record

This directory records the evidence-preserving review of `draft2_06_8_26` begun on
2026-08-16. The purpose is to make the thesis clearer, more precise, and recognisably
grounded in the author's experiments. It is not an attempt to optimise a detector
score or conceal authorship. Automated AI-writing scores are not used as a quality
criterion because they are not reliable evidence of authorship.

## Non-negotiable rules

1. Do not alter an experimental number unless its source artifact supports the change.
2. Do not upgrade a result from exploratory to recommended, or from within-video to
   cross-event, through wording alone.
3. Keep negative and null results. They are part of the thesis evidence.
4. Distinguish the official C2A split from the scene-disjoint split.
5. Distinguish the original four-model ablation from the later assignment and
   real-data experiments.
6. Compile with XeLaTeX and Biber through WSL, then render the PDF and inspect it.
7. Do not commit changes until the author has reviewed them.

## Evidence hierarchy

When documents disagree, use this order:

1. Saved run artifacts (`env.json`, `args.yaml`, prediction JSON, metric JSON/CSV).
2. `05-07-2026-Novelty-Lap/results/MANIFEST.md`.
3. `27-07-2026-Novelty-Lap-4/D2_SIGNIFICANCE.md` and
   `report_partB_stale_draft/FINDINGS_AND_SOURCES.md`.
4. Planning documents and figure notes.
5. Existing thesis prose.

## Review loop

For each section: identify its claim, trace quantitative statements, remove unsupported
interpretation, revise for direct prose, compile, inspect affected pages, and record the
result in `AUDIT_LEDGER.md`.

Quantitative additions, removals, and provenance corrections are summarised in
`NUMERIC_CHANGE_AUDIT.md`.

## Final state of this pass

The 2026-08-16 pass completed a full WSL XeLaTeX/Biber build and visual review of
all 77 rendered pages. Quantitative edits were checked against project artifacts;
the citation set is closed (50 used keys and 50 bibliography entries); and the
remaining cover-date placeholder is recorded as blocked pending author or
supervisor confirmation. No git commit was made.

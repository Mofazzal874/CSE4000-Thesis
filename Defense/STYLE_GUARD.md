# Writing style guard + figure sizing

## Part A — Avoid AI-writing tells (from Wikipedia: Signs of AI writing)

**Banned puffery / promotional words** — do not use:
stands/serves as, is a testament/reminder, vital/significant/crucial/pivotal/key role,
underscores/highlights its importance, reflects broader, symbolizing, contributing to the,
setting the stage for, boasts, vibrant, rich, profound, enhancing, showcasing, exemplifies,
commitment to, delve, landscape, tapestry, meticulous, intricate, interplay, garner, foster,
enduring, robust (as filler), leverage (as filler), realm, navigate (figurative), seamless,
comprehensive (as filler), beacon, myriad, plethora.

**Banned filler / editorializing:**
"it is important to note", "it is worth noting", "notably", "importantly", "it should be noted",
"valuable insights", "align/resonate with", "plays a role in".

**Banned sentence patterns:**
- Rule of three (adjective, adjective, adjective) used for effect.
- Negative parallelism: "not just X but also Y", "not X but Y", "X rather than Y" (as a tic).
- Trailing present-participle clauses that add fake analysis: "…, highlighting its role",
  "…, ensuring …", "…, reflecting …", "…, underscoring …", "…, contributing to …".

**Banned transitions (esp. sentence-initial):** Additionally, Moreover, Furthermore, Overall,
In conclusion, In summary.  → Use plain connectives or none; let sentences follow logically.

**Banned vague attribution:** "industry reports", "observers have cited", "experts argue",
"some critics argue", "several sources".

**Banned conclusion formulas:** "Despite these challenges…", "Future Outlook", vaguely positive
speculation. End sections on a concrete statement of what was shown.

**Formatting:** no Title Case in headings (use sentence case for subsections where the template
allows), no bold inside body prose, no bold-header-colon inline lists, no em-dash pile-ups,
straight quotes only, real equations not text.

## How this thesis writes instead
- Declarative, specific, quantitative. Prefer numbers and named components over adjectives.
- Every claim traceable to `FINDINGS_AND_SOURCES.md`. No hedging, no filler.
- Bullets only to enumerate discrete items, and always after an introductory sentence.
- Vary sentence length; avoid the uniform medium-length cadence typical of generated text.

## Part B — Figure sizing (so replacing a placeholder never shifts layout)

Text block on this A4 template: **width ≈ 15.4 cm**, height ≈ 22 cm usable.
The `\fig
box{path}{height}` macro reserves the **exact height** given, whether the file
exists or not — so exporting the real image later cannot change page spacing. Export each
diagram/plot at roughly the listed aspect so it fills its box (spacing is safe regardless).

| Figure type | Reserve height | Target export aspect (W:H) | Notes |
|---|---|---|---|
| Wide pipeline / framework | 5.5 cm | ~2.6 : 1 | full text width |
| Module diagram (CBAM, P2, C3k2Mamba) | 6.5 cm | ~1.8 : 1 | full text width |
| Additive-ablation chain (wide) | 4.5 cm | ~3.2 : 1 | full text width |
| C2A construction pipeline | 6.0 cm | ~2.4 : 1 | full text width |
| Single data plot (PR, F1-conf, calibration) | 7.0 cm | ~1.3 : 1 | often 0.72×width |
| Bar/waterfall chart | 6.5 cm | ~1.5 : 1 | |
| Confusion matrix | 7.0 cm | ~1 : 1 | 0.6×width |
| Image grid (detection/failure/dataset) | 8.5 cm | ~1.4 : 1 | full text width |
| Training-curve panel (multi) | 7.5 cm | ~1.6 : 1 | |
| Full-page block diagram | up to 18 cm | portrait | landscape/A3 fold if needed |

When a real image's aspect differs from target, `keepaspectratio` shrinks it inside the reserved
box; the box height stays fixed, so surrounding text does not move.

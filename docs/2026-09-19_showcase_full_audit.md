# Full showcase audit and release record — 2026-09-19

## Scope

User request: complete the process, scan the thesis-folder directory, check what adds up, include the presentation, and revise the research showcase.

Inventoried **101,390 files** across the workspace, excluding `.git`, environment/dependency directories, Python caches, and temporary outputs. This was a metadata inventory, not a claim to read every dataset image or source file. The inventory is retained privately at `tmp/showcase-audit/inventory.json`. Content review focused on authoritative documents, experiment summaries, scripts defining metric semantics, archived demos, and candidate visual evidence.

Largest groups: C2A 51,089 files; pendrive 32,388; Defense 6,605; original C2A-project material 4,566; Last Month 2,636; own-drone data 1,240. Found 82 video files, including repeated raw/rendered/handover copies. Credentials and remote-PC access notes were not opened or copied.

## Findings and decisions

| Check | Finding | Action |
|---|---|---|
| Thesis version | Pendrive final PDF is 88 pages and byte-identical to `Defense/draft6_07_09_26/2007074_Human_disaster_v6_latest.pdf` | Use this as the report download |
| Version-history thesis | Extracted text differs in the cover's project number; older history copy leaves the field empty | Retain completed-cover pendrive version |
| Presentation identity | Both source and pendrive PDFs are byte-identical; 43 pages | Include the original PDF and six previews |
| Presentation recency | Deck predates assignment/adaptation extension | Label it as original defense-stage material; provide current-study reading notes |
| Presentation PDF attachments | Ten tiny equation XML files, no detector implementation | Preserve original PDF |
| PPTX contents | 94.7 MiB, including two embedded MP4s | Keep editable deck in full handover; use compact PDF and separate demo clips publicly |
| Presentation page 43 | Image-only Mamba architecture, not a blank page | Explain this in presentation notes |
| Presentation page 31 | Displayed AP50 0.893 vs 0.853 contradicts prose saying a one-point gap | Explain the approximately four-point gap in companion notes |
| Demo lineage | Filename generation and `cbam_p2` model map point to the earlier official-split CBAM+P2 demo model | Remove blanket final-adapted-model attribution |
| Video processing | Existing videos are 1920×1080, about 33.83 and 37.34 seconds | Full-duration H.264 sharing copies at 1280×720; no inference rerun or speed change |
| Gallery consistency | Ranking CSV at conf 0.25, overlays at baseline 0.19 / CBAM+P2 0.16 | Recompute displayed counts from cached detections; distinguish protocols |
| Current repository access | Authenticated read-only GitHub API reported `Mofazzal874/CSE4000-Thesis` has `private: false` | Do not pretend the source is private; request a concrete visibility decision at the release step |

## Newly established gallery correction

The training runner calls `per_image_eval(..., conf=0.25)` when writing `per_image_test.csv`. `make_per_image_and_stats.py` copies those rows unchanged. `make_failure_cases.py` ranks by these values but renders at `THR = {CBAM+P2: 0.16, baseline: 0.19}`. Thus the existing archive sentence that pictures show exactly the scored operating point is inaccurate.

`docs/showcase/audit_gallery.py` re-read cached COCO detections and the official ground truth, then used the same score-ordered greedy IoU matching as the overlay generator. It did not train, rerun inference, or redraw pictures.

| Displayed case | Model/confidence | TP | FP | FN | Recall | F2 |
|---|---|---:|---:|---:|---:|---:|
| Improved / collapsed_building_image0509_0 | baseline / 0.19 | 15 | 9 | 9 | .6250 | .6250 |
| Improved / collapsed_building_image0509_0 | CBAM+P2 / 0.16 | 20 | 7 | 4 | .8333 | .8130 |
| Regressed / collapsed_building_image0124_3 | baseline / 0.19 | 31 | 7 | 4 | .8857 | .8708 |
| Regressed / collapsed_building_image0124_3 | CBAM+P2 / 0.16 | 27 | 12 | 8 | .7714 | .7542 |
| Missed people / flood_image0087_1 | CBAM+P2 / 0.16 | 1 | 12 | 29 | .0333 | .0376 |

The last case's original ranking row at 0.25 has 0 TP, 6 FP, 30 FN. Both records can be correct at their respective thresholds; the captions must not mix them. Public gallery captions now use the displayed-threshold counts. Original archive records are untouched.

## Prepared deliverables

- Root `README.md`, synchronized from the canonical portable companion with adjusted links.
- `public-showcase/`: completed-study narrative; two conceptual diagrams; a result plot; thesis and presentation PDFs; six slide previews; five measured comparison/failure images; two full video demos and posters; nine CSVs; two original bootstrap summaries; citation metadata; attribution notes; independent verification script; whole-package SHA-256 manifest.
- `pendrive/PUBLIC_SHOWCASE/`: standalone copy beside the original handover, with no links outside its boundary.
- `pendrive/PUBLIC_SHOWCASE_2007074.zip`: portable sharing archive.
- `pendrive/README.md`: professor guide updated with current scope, download entry points, and audit notes.

The original hashed thesis archive, Defense files, model weights, datasets, and reference-paper collection are unchanged. Unrelated working-tree modifications are preserved. New maintenance scripts are in `docs/showcase/`; only the independent result checker is inside the public companion.

## Validation

The package verifier checks the complete file manifest, SHA-256 values, relative-link containment/existence, result arithmetic, original bootstrap summaries, gallery count consistency, and the new overlay-count table. Packaging enforces an explicit directory/type boundary, allows only the independent checker as public Python code, rejects absolute workstation paths and credential-like tokens, and caps individual files at 25 MiB.

All 43 presentation pages were rendered and inspected via review sheets; six public previews were inspected with their source text. The copied thesis was reviewed for identity, front matter, evidence statements, and embedded-file absence; it was not newly edited or typeset. New plots and selected overlays were inspected visually. Videos were re-encoded from existing outputs and checked for decode success, dimensions, frame count, and duration. The ZIP is integrity-tested after construction.

No claim is made that the full experiments were independently reproduced. No new training, model inference, or bootstrap resampling took place.

## Public/private release step

Existing research repository: `https://github.com/Mofazzal874/CSE4000-Thesis` (public at audit time).

Proposed new public companion: `Mofazzal874/aerial-human-detection-results`, populated only from `public-showcase/`, with fresh history. No repository has yet been created and no upload, visibility change, or commit has been performed.

The remaining owner decision is whether to publish this prepared companion and make the existing research repository private. A privacy change cannot recall copies or forks already obtained while the repository was public. Keep this operational decision separate from the paper's scientific claims.

AGENTS.md explicitly states: **“NO self-initiated git commits … finish changes, present a review summary … then WAIT.”** Therefore the reviewed package and proposed action are prepared first; approval is the final release step. Suggested commit message: `docs: publish audited thesis showcase with presentation and video demos`.

## Maintenance commands

Run from the research workspace root in PowerShell:

```powershell
python docs/showcase/prepare_media.py
python docs/showcase/audit_gallery.py
python docs/showcase/package_release.py
```

The first two commands prepare media and recount the overlays from private source artifacts. The last command validates the public boundary, refreshes root documentation and manifests, verifies the standalone copy, and assembles the ZIP. Neither command publishes or commits.

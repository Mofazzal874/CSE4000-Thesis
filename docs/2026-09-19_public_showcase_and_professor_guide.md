# Public showcase and professor handover — 2026-09-19

> Historical preparation record; its separate-repository recommendation is superseded by the [September 20 current-repository decision](2026-09-20_current_repository_showcase.md). The [full evidence audit](2026-09-19_showcase_full_audit.md) records the thesis, presentation, videos, and corrected gallery. The research repository is public; the earlier assumption that it was private must not be relied on. Publication occurred after this initial record, as documented in the updated audit.

## Request and outcome

The user requested an excellent professor-facing repository README, diagrams, and a way to share results and demonstrations without exposing the research codebase. The user selected **public showcase plus professor guide**.

Prepared a new root README, a self-contained `public-showcase/` release candidate, and `pendrive/README.md` as the professor's entry point. The existing hashed thesis archive and all Defense sources were left intact. No repository was created, no file was uploaded, and no commit was made.

## Recommended access arrangement

Use a **new, separate public companion repository**, containing only the contents of `public-showcase/`. Keep the research repository private. GitHub access is governed at repository level: making the research repository public would expose its tracked contents and history, even if the README links only to selected folders. A branch or `.gitignore` is not a private boundary inside a public repository. [GitHub: repositories and visibility](https://docs.github.com/en/repositories/creating-and-managing-repositories/about-repositories).

The public companion can later support a project website. GitHub Pages supports public repositories on GitHub Free and looks for an entry document such as `README.md`, `index.md`, or `index.html`. Public site output is public even when built from an eligible private repository. [GitHub: creating a Pages site](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site).

For videos, use selected annotated clips with hosted playback links; retain the full-resolution originals in the private archive. This avoids loading the documentation repository with repeated video versions. GitHub warns above 50 MiB and blocks regular Git files above 100 MiB; larger distributable files can use release assets. [GitHub: large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). These service facts were checked against official documentation on 2026-09-19.

Suggested companion repository name: `aerial-human-detection-results`. This is a proposed name, not an existing URL. Use the descriptive topic title until publication naming is settled; the formal thesis citation retains the archived MSA-YOLO title.

## Contents and boundaries

| Deliverable | Purpose |
|---|---|
| `README.md` | Professor-facing research narrative at the root, with private workspace navigation |
| `public-showcase/README.md` | Portable public narrative with internal links only |
| `public-showcase/assets/*.svg` | Two original conceptual diagrams: study overview and retained detector/training intervention |
| `public-showcase/results/` | Seven verbatim CSV exports plus SHA-256 manifest |
| `public-showcase/analysis/verify_results.py` | Standard-library arithmetic and integrity checks; no private implementation |
| `public-showcase/docs/` | Evidence interpretation and artifact availability |
| `public-showcase/demos/README.md` | Two existing own-drone demo candidates; settings and provenance caveats |
| `pendrive/README.md` | Professor reading route, direct thesis/slide/video links, and evidence index |

The public folder contains no checkpoint, annotation export, raw dataset, third-party reference PDF, private training/evaluation module, credential, or absolute local path. It deliberately discloses its limits as a result-inspection package, rather than claiming complete public reproducibility.

## Research corrections that matter

- Used the current completed study, not the older FCCG proposal. FCCG remains a null result.
- Distinguished the +1.62 ± 0.49 point three-seed summary from the +2.08 point seed-0 bootstrap estimate.
- Read the evaluator's threshold semantics: extended-study size-bin recall uses each model's F1-optimal threshold; F2 means Best_F2, not fixed-confidence F2.
- Used the official-split latency of 14.6 ms from Table 4.4, without substituting another run's timing.
- Corrected the public additional-instance count to 369, based on the archived matched counts.
- Preserved the R-set within-video / unseen-event distinction. Read the unseen-event bootstrap JSON: AP50 delta +1.69 points, interval [−3.19, 5.69], p = 0.468 (0.47 rounded).
- Did not repeat the blanket archive claim that every demo uses `d3_all`; the two filenames establish different settings, but exact checkpoint hashes were not established.
- Did not repeat an absolute AP ceiling, a unique MSA-YOLO naming claim, publication acceptance, or universal superiority.

## Release steps for the author

1. Review the root README and portable public companion. Confirm the public attribution and licence choice.
2. Select the drone clips, check the producing model/logs, and choose their hosting destination. The professor can already open the existing local videos from `pendrive/README.md`.
3. Create an empty companion repository separately from the research repository. Copy **only the contents of `public-showcase/`** into its root; do not fork the private research history or upload the entire pendrive archive.
4. Run `python analysis/verify_results.py` in that standalone folder, inspect the staged file list, and make the first public commit after review.
5. Add real video URLs and, if desired, a cleared thesis PDF/slide link. Check that each public link opens without private-account access. Add a website only when wanted.

No platform/account was selected and no external publication was requested explicitly enough to create a public repository during this preparation. The remaining publication action needs a destination and release selection, not a repeat of the completed documentation work.

## Validation and review

The result checker passes: CSV byte hashes, seed mean/sample SD, count-derived recalls, and adaptation differences. Two SVG previews were rendered and visually inspected. Relative document links and public-package boundaries were checked locally; model inference and training were not rerun.

`pendrive/` is already ignored by the parent repository. Its professor guide remains on disk and will not be included in a normal parent-repository commit. The public companion and root README are outside that ignored directory. Root research sections mirror the public README with adjusted paths; edit the public version first when maintaining both.

Suggested commit message: `docs: add research showcase and professor handover guide`

No self-initiated commit, per the user's standing rule in AGENTS.md. Existing unrelated working-tree changes were preserved.

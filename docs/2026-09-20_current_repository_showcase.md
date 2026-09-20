# Current-repository showcase revision — 2026-09-20

## Decision and access

The user requires the README, professor guide, presentation, videos, diagrams, and results in the existing **Mofazzal874/CSE4000-Thesis** repository. Its public visibility is retained. A curated folder makes the work easier to inspect and download; it cannot hide tracked code or Git history elsewhere in a public repository.

GitHub applies visibility at repository level. Public GitHub Pages output can be built from a private repository on eligible plans, including GitHub Pro for a personal account. That is a possible same-repository website arrangement, not folder-level privacy within the GitHub repository. No subscription, visibility, or Pages setting was changed. Sources checked September 20: [repository visibility](https://docs.github.com/en/repositories/creating-and-managing-repositories/about-repositories), [Pages availability and publishing sources](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site).

## Work prepared

- Revised the canonical `public-showcase/README.md` and generated root README to identify CSE4000-Thesis, distinguish standalone inspection from access control, and link a tracked professor guide.
- Added `public-showcase/docs/PROFESSOR_GUIDE.md`: ten-minute review, report/deck links, questions the evidence can answer, and verification instructions.
- Corrected availability, evidence, citation, and local pendrive-guide language. The original thesis, presentation, datasets, detector code, and numerical source exports remain unchanged.
- Added `public-showcase/.gitattributes` to preserve the exact bytes used by result and package manifests through Git's text conversion.
- Updated the packaging script, regenerated manifests, and refreshed the local standalone copy and ZIP. The scoped attributes affect only the curated companion.
- Corrected the historical audit and project status to record what was actually published and the user's subsequent direction.

The package retains the 88-page thesis, 43-page defense deck with six previews, two complete annotated drone videos, two conceptual SVG diagrams, a result chart, five comparison/failure images, nine CSVs, two bootstrap summaries, and an independent checker. The gallery captions retain the earlier audit correction for mismatched confidence thresholds. The original deck is preserved with reading notes identifying historical claims and its incorrect prose comparison.

## Repository and publication state

At session start, the current branch was `novelty-lap-4` at `74186c5bc6c069d46d0805d801b60448b9ee9359` with a clean working tree. Its origin branch pointed to the same commit, which already included the earlier showcase preparation. The remote default branch remained `main` at `11616675922a3214cbcb01356493550067109d9b`. These are observations, not commits or pushes performed during this revision.

The previously approved separate repository, [aerial-human-detection-results](https://github.com/Mofazzal874/aerial-human-detection-results), and its v1.0.0 release already exist. Following the correction, neither was changed or deleted. Its Git source-tree manifest has the newline-conversion issue described in the [publication record](2026-09-19_showcase_full_audit.md); its directly uploaded ZIP passed the previous integrity check. The current local revision fixes this issue for the package intended for CSE4000-Thesis.

## Validation

The local 45-file companion and its pendrive copy pass the full manifest/hash, relative-link, numerical, bootstrap-summary, and gallery-count checks. All local root-README links and all 23 pendrive-guide links resolve. The seven original source-table CSVs and both PDFs preserve their archived bytes. The regenerated ZIP passes its archive integrity check (56.90 MiB).

A Git add/archive round trip with `core.autocrlf=true` also preserves every companion file byte-for-byte and passes the same checker. The scoped attributes fix the previous newline-conversion failure without changing the original CSV exports. Documentation uses LF newlines; the CSV and JSON result exports retain their source formatting. No new training, inference, video conversion, or statistical resampling took place.

## Commit review

Scope: `README.md`, `public-showcase/`, `docs/showcase/package_release.py`, this record, the two historical showcase records, `START_HERE.md`, and the active-lap README. The local `pendrive/` copy and guide remain ignored convenience outputs.

Suggested commit message: `docs: integrate audited showcase and professor guide into current repo`.

The changes were first staged for owner review under [AGENTS.md](../AGENTS.md). The user then explicitly directed: **“the main is the default branch. So do whatever you need to do.”** This authorizes completing the documentation publication on `main`, including the required commits and push; no further approval is needed for that scope.

Publication uses a checkout based on the current remote `main` and copies only the generated root README, the 45-file companion, and the necessary media exceptions in `.gitignore`. Research-branch status/audit links use explicit `novelty-lap-4` URLs so they resolve from `main` without merging unrelated experiment history. The portable ZIP is published as `showcase-v1.0.1` in CSE4000-Thesis. The separate repository is untouched.

Staging includes `git add --renormalize -- public-showcase` so previously tracked exports adopt the new byte-preservation attributes.

## Main publication completed

Main commit: [`0188dee923eedcade0d2b1059ae07950e9ef27f1`](https://github.com/Mofazzal874/CSE4000-Thesis/commit/0188dee923eedcade0d2b1059ae07950e9ef27f1), a normal forward update from `1161667`. Exactly 47 files changed: the root README, four media exceptions in `.gitignore`, and the 45-file companion. No existing research implementation or experiment files were changed on `main`.

Anonymous GitHub checks confirm the repository remains public, its default branch is `main`, and the landing README matches the reviewed revision. All 45 companion files, including both PDFs and both videos, were fetched without authentication and matched byte-for-byte. Running the verifier on that downloaded copy passes the full integrity, links, numerical, and gallery checks.

The synchronized research-branch revision was committed and pushed as `a48182b`. All five research-navigation links from the main README resolve anonymously, including this record.

Release: [showcase-v1.0.1](https://github.com/Mofazzal874/CSE4000-Thesis/releases/tag/showcase-v1.0.1). The [portable ZIP](https://github.com/Mofazzal874/CSE4000-Thesis/releases/download/showcase-v1.0.1/PUBLIC_SHOWCASE_2007074.zip) is 59,659,073 bytes (56.90 MiB), SHA-256 `874bac00a118512331e91b164a37df4df7440370f4f812e4a4ad48f34bad3b6f`. Its final public download was fetched without authentication and matched that hash. An initial upload timed out; only its incomplete zero-byte draft asset was removed before a successful streamed retry. The release is public, not a draft.

All requested publication work is complete on the current repository's default `main` branch. Main remains the default; repository visibility and the separate repository were not changed.

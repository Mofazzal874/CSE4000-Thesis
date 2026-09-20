# Showcase privacy correction

The user objected to the personal and academic identifiers in the public package and to the inflated writing. They explicitly requested removal of the full PDFs and identifying previews while retaining results and demos, and approved temporarily making both repositories private.

## Containment

- CSE4000-Thesis and aerial-human-detection-results were changed to private.
- The two earlier ZIP releases were changed to drafts. Their downloads are no longer public.
- Signed-out checks returned 404 for both repositories, both old ZIP download URLs, and the old main-commit thesis URL.
- No history rewrite or force-push was performed. Older commits, branches, and the full research tree still contain identifying material. No claim is made to recall copies already downloaded.

These repositories must remain private until a later sharing decision accounts for the old history. A new commit deleting a file does not erase older versions. [GitHub's explanation](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).

## Current package

Removed the full thesis PDF, presentation PDF, all six slide previews, presentation notes, and personal citation metadata from the current showcase. Removed names, institution/supervisor details, academic identifiers, personal citation text, old release links, and internal workspace links from the landing page.

The root README is now 583 words, compared with 2,420 previously. It describes the method, three result groups, demos, limitations, and result verification. Supporting pages were shortened. Scientific values and qualifications were retained.

The package now contains 35 files: CSVs, bootstrap summaries, diagrams, a results chart, selected detection examples, two existing drone demos, concise notes, a checker, and hashes. The full original thesis archive and Defense files remain intact on the local computer.

Local results-only copy: `pendrive/RESULTS_SHOWCASE/`. Local ZIP: `pendrive/RESULTS_SHOWCASE.zip` (29.27 MiB). This replacement ZIP was not published. Older local `PUBLIC_SHOWCASE` copies contain identifying documents and are retained only as private archives.

## Prevention and verification

The media preparation script no longer copies thesis documents or renders slide previews. The package builder rejects PDFs/CFF files, academic-document directories, and the known identifying text. Ignore rules also exclude those document paths. Current result exports and demo files are unchanged.

The new package and ZIP pass full file hashes, local links, result arithmetic, gallery counts, and archived-bootstrap consistency checks. A targeted text/filename scan finds none of the known identifying markers in the package or root README. Retained image metadata was checked, and the two demo posters were visually reviewed. This is not a guarantee of personal anonymity from footage, GitHub account identity, or other research records.

The two default-branch indexes were checked byte-for-byte against the verified 35-file package before committing. Cleanup commits use a generic maintainer display name and the account's GitHub noreply address rather than a personal email. Both repositories remain private; no new public release is created.

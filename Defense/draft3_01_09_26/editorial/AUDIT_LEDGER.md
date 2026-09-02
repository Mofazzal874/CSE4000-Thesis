# Draft 2 audit ledger

Status values: `pending`, `revised`, `verified`, `blocked`.

| Area | Initial finding | Required action | Status |
|---|---|---|---|
| Backup | Source needed a pre-edit snapshot | Byte-verified backup created at `../draft2_06_8_26_BACKUP_20260816_205555` | verified |
| Build | Interrupted build left inconsistent auxiliary files | Moved generated artifacts to `../draft2_generated_stale_20260816_205944`; clean WSL build completed | verified |
| PDF baseline | 80 pages; blank-looking List of Figures continuation and several crowded tables | Final WSL build is 77 A4 pages; all-page contact sheets and affected full-size pages inspected | verified |
| Abstract | SAHI/TTA result attributed implicitly to recommended model although provenance says old Mamba model | Remove it from the headline abstract; retain only with correct provenance in Chapter IV | verified |
| Chapter I | CBAM described as an accuracy improvement at negative cost; SAHI/TTA contribution misattributed | Reframe CBAM as an efficiency-preserving substitution and correct inference-study provenance | verified |
| Chapter II | DETR incorrectly described as removing NMS in general; current literature section reads as a compressed list | Correct the technical statement and tighten synthesis | verified |
| Chapter III | CBAM maps described as concentrating on humans, but measured maps are diffuse | Describe the intended mechanism separately from observed maps | verified |
| Chapter III | Baseline said to contain two C2PSA blocks | Correct to one C2PSA module | verified |
| Chapter III | NMS said to preserve separate crowded people | Correct: NMS suppresses duplicates and can suppress neighbours in crowds | verified |
| Chapter III | Classification called objectness-class probability | Correct to class probability for this YOLO11 head | verified |
| Chapter IV | Software versions looked unusual | Verified directly from `env.json`: Python 3.11.9, torch 2.12.0+cu126, Ultralytics 8.4.56 | verified |
| Chapter IV | Early-stopping settings attributed to an unspecified literature convention | Replace with the actual project rationale and recorded protocol | verified |
| Chapter IV | C2A-to-real transfer described as future work even though it is measured later | Correct tense and cross-reference the extended study | verified |
| Chapter IV | ECE interpreted as pointwise equality between confidence and precision | Replace with the correct aggregate calibration interpretation | verified |
| Chapter IV | SAHI/TTA table caption and prose claim CBAM+P2 | Relabel as older Mamba+CBAM+P2 exploratory study; state metrics are not directly comparable | verified |
| Chapter IV | One-seed statement claims differences exceed seed noise without evidence | Remove; explain that later replication exists only for the central assignment pair | verified |
| Chapter V | C2A and Ultralytics licensing claims need exact legal verification | State the verified Ultralytics AGPL-3.0 edition and describe C2A cautiously because its Kaggle page shows no explicit licence | verified |
| Chapters VI--VII | Several claims use promotional wording or inherit SAHI/TTA misattribution | Make evidence-bound and correct the inherited result | verified |
| References | Metadata contains provisional comments and online sources of mixed authority | Final check: 50 citation keys, 50 bibliography entries, zero missing or uncited keys, and no unresolved-reference warnings | verified |
| Formatting | 43 forced `[H]` floats conflict with the verified top/bottom rule | Replaced all with `[tbp]`; final 77-page render has no blank continuation page, detached caption, or unreadable appendix table | verified |
| Final PDF | A complete post-edit visual and compiler check was required | WSL XeLaTeX/Biber build succeeded; A4, 77 pages, 24,089,666 bytes; all pages rendered and reviewed | verified |
| Static prose checks | Superseded technical claims could remain after section edits | No matches for the audited C2PSA-count, attention-map, SAHI/TTA-provenance, objectness, NMS-guarantee, or ECE misstatements | verified |
| Submission date | Cover and title page say July 2026 with an unresolved TODO | Requires author/supervisor confirmation; do not guess | blocked |

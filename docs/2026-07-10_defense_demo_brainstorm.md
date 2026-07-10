# Defense Demo Plan — Brainstorm (2026-07-10, defense tomorrow)

## Decision: demo on C2A test set + own drone shoot — YES
- C2A held-out test split = in-scope, standard qualitative-results practice.
- Own drone footage = strongest card (real deployment evidence).
- SARD excluded from demo, NOT hidden. Prepared line for the board:
  > "The demo shows in-domain performance on C2A's held-out test set plus our own
  > drone footage. Cross-domain transfer to SARD is a separate question — we measured
  > it (zero-shot dip, recovered to ~0.89 after fine-tuning) and discuss it as future work."
- Keep SARD numbers in a backup slide, not in the demo UI.

## Architecture: cached-inference demo (no GPU on laptop)
- Do NOT run SAHI+TTA live on CPU (30–60+ s/image, freeze risk).
- Tonight on GPU PC (AnyDesk) or Kaggle:
  - Batch inference over ~30–50 curated images (C2A test + drone frames)
  - Configs: baseline, CBAM+P2, CBAM+P2 + SAHI+TTA
  - Save raw predictions at **conf=0.01** as JSON (boxes, scores) per image/config
  - Log real GPU latency per config (label hardware in UI)
  - Render annotated drone VIDEO (boxes burned in) — the showstopper
- Laptop UI draws boxes from JSON on demand → instant.
- **Confidence-threshold slider** re-filters cached raw predictions live →
  feels genuinely interactive, and is honest (same post-processing as live).
- Optional "true live" button: plain CBAM+P2, 640px, single image on CPU
  (~2–4 s) — one real inference for credibility.

## UI: Gradio Blocks (~150 lines, fully offline local browser)
- Tab 1 — Interactive gallery: thumbnails → side-by-side baseline vs
  CBAM+P2+SAHI, conf slider, detection count, GPU latency per panel.
  Curate images where SAHI visibly rescues tiny humans baseline misses.
- Tab 2 — Drone shoot: annotated video + frames in comparison view.
- Tab 3 — Results: 4-row C2A ablation table + PR curves (static figures).
- (Optional) live CPU inference button on Tab 1.
- Alternatives considered: Streamlit (fine, more layout work), plain HTML
  gallery (zero-dep fallback, no interactivity).

## Tonight's checklist (in order)
1. GPU PC/Kaggle: batch-inference script → JSON @ conf 0.01 + latency log + drone video (~1–2 h).
2. Curate demo images deliberately (demo ≠ evaluation; Tab 3 carries the numbers).
3. Laptop: Gradio app reading JSON (`pip install gradio opencv-python`);
   verify it launches OFFLINE tonight.
4. Rehearse once end-to-end, time it.
5. Fallback: folder of annotated PNGs + the video in case anything breaks.

## Hard rule
No live dependency on Kaggle or AnyDesk PC during the defense — venue Wi-Fi
and remote desktops are where demos die.

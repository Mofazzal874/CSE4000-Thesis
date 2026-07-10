# Defense Demo — Master Task List
Defense date: 2026-07-11. Work through phases in order; tick boxes as you go.

## Phase 0 — Consolidate assets (DONE ✅ by Claude, 2026-07-10)
- [x] Locate final weights (AdamW baseline 06-16, CBAM+P2 06-02, enriched 07-02)
- [x] Verify both runs trained on `new_dataset3` (data.yaml checked in both run folders)
- [x] Confirm CBAM needs custom-module registration to load (code snapshot copied)
- [x] Copy models → `models\`, scripts → `code_snapshots\`, metadata → `run_metadata\`
- [x] Copy C2A test set (2043 imgs + labels) → `data\c2a_test\`
- [x] Copy drone videos (10m/30m/50m) → `data\drone_shoot\`
- [x] Verify counts + total size (2.01 GB), README with provenance

### YOUR check for Phase 0
- [ ] Open `Defense\demo\` and eyeball the folders match README
- [ ] Decide: use `cbam_p2head_finetune_enriched.pt` for the DRONE video, or stick
      to pure `cbam_p2head.pt` everywhere? (enriched = better on your footage;
      pure = exactly matches report numbers. You can also show both.)

## Phase 1 — GPU batch inference (TONIGHT, on the AnyDesk PC)
Claude writes `app\precompute.py`; you run it on the GPU PC.
- [x] Claude: `app\cbam_modules.py` (verbatim CBAM classes + full registration)
- [x] Claude: `app\precompute.py` (3 C2A configs + 4 drone configs, raw preds
      @ report floor conf 0.10 → JSON, latency log, curation_hints.csv;
      SAHI = report's winning slice256/ov0.30 GREEDYNMM + per-tile TTA;
      --smoke flag; auto-fallback to E:'s c2a path)
- [x] Claude: `app\annotate_video.py` (boxed MP4 renderer, SAHI+TTA default)
- [x] Claude: `app\GPU_RUN_GUIDE.md` (exact commands, expected output, timings)
- [ ] You: follow GPU_RUN_GUIDE.md → smoke test → full run → video(s)
- [ ] You: copy the WHOLE `demo\` folder EXCEPT `data\c2a_test\` to the AnyDesk PC
      (RTX 4070 Ti SUPER) as `E:\Thesis_mofazzal_2007074\demo\` (~1 GB with drone videos).
      c2a_test is not needed there — the identical split (MD5-verified) already
      exists at `E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3`;
      precompute.py auto-falls-back to that path when `data\c2a_test\` is absent.
- [ ] You: on that PC, `pip install sahi` (everything else is already installed —
      same env that trained the models: Python 3.11.9, torch 2.12, ultralytics 8.4.56)
- [ ] You: run precompute.py there, confirm no errors, watch first few outputs
- [ ] You: copy `results\` back to laptop `Defense\demo\results\`

## Phase 2 — Build + test Gradio app (laptop)
- [x] Laptop env checked: Anaconda Python 3.12.7; gradio missing → dedicated
      venv at `demo\.venv` (Anaconda base left untouched)
- [x] Claude: `app\demo_app.py` (3 tabs: C2A side-by-side + conf slider + GT
      toggle + rescue-sorted dropdown / drone frames + annotated videos /
      report tables + measured latency). Launches fine even before results
      arrive (shows what to copy where).
- [x] Claude: `launch_demo.bat` at demo root (double-click to start)
- [x] Claude: smoke-tested app with mock results — HTTP 200, render pipeline
      verified (boxes, conf filtering, GT toggle, captions). Fixed 2 Gradio 6
      API breaks (theme/show_api) found during the test.
- [x] Real results validated; fixes applied after first user test: GT now
      orange (was BGR-blue), image loads ~0.5s (JPEG + 1600px downscale +
      slider release events), videos re-encoded mp4v→H.264 1920px (browsers
      cannot play mp4v; 4K originals kept in results\raw_renders\)
- [x] Live inference tab added: cbam_p2head.onnx via ONNX Runtime CPU,
      ~600 ms/image, validated ±1 box vs GPU cache on test images
- [ ] You: restart via `launch_demo.bat` → full click-through, Wi-Fi off
- [x] Live-CPU button: DROPPED (laptop ultralytics 8.0.196 too old for the
      8.4.56 checkpoints; cached demo doesn't need it)

## Phase 3 — Curate + polish
- [ ] Pick 20–30 hero images where SAHI visibly rescues tiny humans baseline misses
- [ ] Verify Results tab numbers match the report's ablation table
- [ ] Prepare the one honest SARD sentence + backup slide

## Phase 4 — Rehearse + fallback (BEFORE sleeping)
- [ ] Full offline rehearsal: airplane-mode the laptop, launch app, run the demo
- [ ] Time it (target ≤ 5 min)
- [ ] Export fallback: folder of annotated PNGs + the annotated MP4
- [ ] Charge laptop; copy demo folder to a USB stick as disaster backup

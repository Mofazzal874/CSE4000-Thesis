# Laptop Demo Guide (Phase 2+) — launch, test, present

## Launch
1. Make sure `results\` (copied back from the GPU PC) sits at `Defense\demo\results\`
   and contains: predictions_c2a.json, predictions_drone.json, latency_summary.json,
   drone_frames\ (and annotated_videos\ once rendered).
2. Double-click **launch_demo.bat** (demo folder root).
3. A terminal opens, then the browser at http://127.0.0.1:7860. Keep the terminal open.
4. Fully offline — works with Wi-Fi off (test it that way!).

## Click-by-click test (do TONIGHT, ~5 min)
Tab "C2A test set":
- [ ] First image loads automatically in both panels (left baseline, right SAHI)
- [ ] Drag the confidence slider down to 0.10 → boxes multiply; up to 0.6 → fewer
- [ ] Toggle "Show ground truth" → orange boxes appear, left panel misses some,
      right panel covers more of them
- [ ] Next ▶ / ◀ Prev walk through images (they are sorted by biggest SAHI rescue)
- [ ] Switch left panel to "YOLO11m+CBAM+P2 @640" → mid-point comparison works
Tab "Drone shoot":
- [ ] Annotated video plays (if rendered); pick another from the dropdown
- [ ] Frames: 10m / 30m / 50m all open; slider works; at 50m the SAHI panel
      finds people the plain panel misses
- [ ] Note: at conf 0.10 the enriched model shows MANY boxes (low-conf FPs) —
      that is expected; the demo default is 0.30
Tab "Results":
- [ ] Both report tables render; measured-latency table shows the 4070 numbers

## Suggested 4-minute demo flow at the defense
1. Results tab (30 s): "this is the ablation from the report — CBAM+P2 recommended."
2. C2A tab (2 min): 2–3 hero images. Script: "left = baseline, right = ours+SAHI.
   Orange = ground truth. Watch the tiny humans." Drag slider once to show
   precision/recall trade-off live.
3. Drone tab (90 s): play the 50m annotated video. "Our own footage, 50 m
   altitude." Then one frame comparison if asked how it works.
If asked about SARD: "Demo is in-domain (C2A test) + our footage. Cross-domain
transfer to SARD is measured in the thesis and discussed as future work."

## Disaster fallback (prepare tonight)
- Keep `results\annotated_videos\*.mp4` playable in any video player.
- Export 3–5 screenshots of the app (Win+Shift+S) into Defense\demo\fallback\.
- Copy the whole demo folder to a USB stick.

## Troubleshooting
- Browser doesn't open → open http://127.0.0.1:7860 manually.
- Port busy → `launch_demo.bat` window shows an error; edit the .bat to add
  `--port 7861` after demo_app.py.
- "results not found" banner → the results folder isn't at Defense\demo\results.

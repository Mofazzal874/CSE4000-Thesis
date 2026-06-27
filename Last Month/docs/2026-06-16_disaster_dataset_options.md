# Real disaster-human aerial datasets — scouting result (2026-06-16, web-curated)

Goal: a REAL (not synthetic) aerial dataset of humans in DISASTER scenes, WITH person
bounding boxes, to test C2A transfer more faithfully than wilderness SARD.

## WINNER — NITC Person Rescue  ★ use this
- **What:** UAV images of people **trapped inside buildings, on roofs, and balconies** (flood/
  urban-disaster rescue) — genuinely disaster-aligned, not wilderness.
- **Size:** ~**2,600 annotated person images** from 9 videos.
- **Labels:** object-detection boxes (+ captions). YOLO-format per the dataset description.
- **Download:** Google Drive, **access on request** — GitHub `shubhasreeav/NITC-Person-Rescue`
  → request with university email + stated purpose. (Hosted: drive.google.com/.../1D5SNPhMh04...)
- **Cite:** "Fine-tuned deep models for niche datasets — People detection in UAV building images
  to aid rescue operations" (ScienceDirect S1569843225006326, 2025).
- **Caveats:** video frames (frame-correlation, like SARD — fine for zero-shot); hard set
  ("models struggle with people behind windows") → expect a LOW zero-shot, which is the point;
  no explicit license (research use, cite the paper).
- **Plugs in free:** it's YOLO-format → `sard_eval.py` works on it as-is (point `$env:SARD_ROOT`
  at it). C2A→NITC zero-shot = your real-DISASTER generalization number (~minutes, no training).

## One-click alternatives (no access request)
| Dataset | Real | Disaster? | Person boxes | Get it |
|---|---|---|---|---|
| Roboflow "Aerial Person Detection" (7,015 imgs) | ✅ | ❌ general aerial | ✅ | Roboflow direct DL (like SARD) — a 2nd real axis |
| MOBDrone | ✅ | ⚠️ maritime man-overboard | ✅ 113k boxes | public — "maritime SAR" axis, not land-disaster |
| Roboflow Universe `q=disaster` | ✅ | ✅ varies | varies | drill in for an aerial+person one |

## NOT usable (real disaster scenes but NO person detection labels)
- **AIDER** (classification of disaster type — the backgrounds C2A pasted onto)
- **FloodNet** (segmentation: water/building-flooded/…)
- **RescueNet** (segmentation: debris/building-damage/…)
- **DRespNeT** (post-earthquake building-access-point segmentation, not persons)

## Recommendation
1. **Request NITC access NOW** (university email). It is the disaster cross-dataset test you want.
2. While waiting: run the joint deployable model (Phase 2) + write.
3. When it arrives: `sard_eval.py` on NITC → C2A→NITC zero-shot. Two real axes:
   **SARD (wilderness) + NITC (disaster/buildings)** = a much stronger generalization section.
4. SAHI: NITC may have larger native images than C2A/SARD — IF so, SAHI is worth trying there
   (it only helps on large images); check resolution after download.

## How to make me search the web (for the future)
I have live `WebSearch` + `WebFetch` — just say "search the web for X" / "find me a dataset/paper
for Y" and I'll run it. For a big multi-source cited report use `/deep-research` (slower, fans out).
For focused hunts like this, a direct ask is faster and more reliable.

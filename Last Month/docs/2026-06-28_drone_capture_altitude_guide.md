# Drone footage capture guide — DJI Air 3S × your CBAM+P2 joint model
Date: 2026-06-28

Goal: collect REAL drone footage that your joint C2A+SARD model can actually detect humans in,
so the test is a fair, representative real-world validation (supervisor's request).

---

## 0. What your model is tuned for (this sets the altitude)
- Trained on **C2A** (aerial disaster, tiny pasted humans) + **SARD** (real drone SAR humans).
- **CBAM + P2 head** → its strength is **small/tiny objects**; per-size recall is strong down to
  ~16 px and still usable to ~8 px, best at small–medium (16–96 px).
- Input size **640**. Single class: **person**. Joint-val mAP50 ≈ **0.877**.
- => The model performs best when a person occupies roughly **16–60 px at the model's input scale.**

## 1. THE critical gotcha: 4K → 640 downscale destroys tiny humans
Your model runs at **640 px**. A DJI 4K frame is **3840 px wide** = **6× larger**.
- If you feed a **whole 4K frame resized to 640**, every person shrinks **6×**. A 32-px person in
  4K becomes ~5 px at 640 → **very-tiny → the model misses it.**
- Fix = **SAHI (sliced inference)**: cut the 4K frame into ~640-px tiles and run the model per tile
  at **native resolution**, so the person keeps its full 4K pixel size. **This is mandatory for any
  altitude above ~25 m.** (This is the SAHI step already in your plan.)

So altitude advice depends on the inference method — use the SAHI column.

## 2. Altitude → person pixel size (DJI Air 3S WIDE camera, 1", 24 mm-equiv, 4K)
Approx GSD = 1.5 × altitude / 3840 (HFOV ≈ 74° for 4K video). Person footprint nadir ≈ 0.5 m;
oblique (~45°, body visible) ≈ 1.5 m usable. "px (4K)" = pixels in the 4K frame = what SAHI sees.

| Altitude | GSD | Nadir person px (4K) | Oblique person px (4K) | With SAHI → model bin | Whole-frame@640 (÷6) |
|---|---|---|---|---|---|
| 20 m | 0.8 cm/px | ~64 | ~192 | large (trivial) | ~11–32 px (usable) |
| 40 m | 1.6 cm/px | ~32 | ~96 | **small (ideal)** | ~5–16 px (marginal) |
| 50 m | 1.9 cm/px | ~26 | ~77 | **small (ideal)** | too small |
| 60 m | 2.3 cm/px | ~21 | ~64 | **small (ideal)** | too small |
| 70 m | 2.7 cm/px | ~18 | ~55 | small/tiny (good) | too small |
| 80 m | 3.1 cm/px | ~16 | ~48 | tiny (stress test) | too small |
| 100 m | 3.9 cm/px | ~13 | ~38 | tiny–very-tiny (hard) | fails |
| 120 m | 4.7 cm/px | ~11 | ~32 | very-tiny (very hard) | fails |

## 3. Recommended capture plan (the exact scenario)
**Primary (best performance):**
- Camera: **WIDE (1") camera, 4K (3840×2160), 30 fps**, lowest ISO that keeps it bright, fast shutter
  to avoid motion blur (blur kills tiny detection).
- Altitude: **40–70 m AGL**, **NADIR (gimbal straight down, −90°)** — matches the C2A/SARD aerial
  distribution and lands people at 18–32 px (4K) = your model's sweet spot **with SAHI**.
- Flight: **slow or hover** over each scene; let the gimbal settle. Steady > fast.
- Inference: **run SAHI** on the frames (tile ≈ 512–640, overlap ~20%), conf 0.25.

**Also capture (for a strong thesis figure + robustness):**
- **Multi-altitude sweep** over the SAME scene: 20, 40, 60, 80, 100 m → gives a
  "recall vs altitude" curve (directly ties to your per-size-recall analysis).
- **One oblique pass (~45° gimbal)** at 50–80 m → people show their full body (more pixels), a
  second realistic SAR viewpoint.
- Optional **3× tele (70 mm) camera** for high-altitude detail: at the same altitude it gives ~3×
  the pixels-on-person (covers less ground). Use it to fly higher *and* keep detail, or to confirm
  detections. (Wide+nadir is more representative of training; tele is a bonus.)

**Frames:** record video, then extract ~1 frame/sec for testing/annotation (or shoot 50 MP stills
for max detail). Keep high bitrate (avoid compression mush on tiny people).

## 4. "Human settings" — the scene/subject matrix to capture
Cover these so the test is meaningful (and shows where the model holds vs breaks). The bolded ones
are the hard, SAR-relevant cases that matter most for a disaster model:

- **Pose:** standing, walking, **lying down / prone (injured/unconscious)**, sitting, crouching.
  Lying-down is the key disaster case — small footprint, looks like debris.
- **Background:** open ground, **rubble/debris**, **vegetation / partial tree cover**, water edge,
  road/pavement, mixed clutter.
- **Density:** single person, small group (2–5), scattered individuals across the frame.
- **Occlusion:** fully visible, **partially occluded** (behind objects / under canopy).
- **Lighting:** bright midday (even light) AND low sun (long shadows). Keep it daylight RGB — the
  model was not trained for dusk/night/thermal.
- **Contrast:** high-contrast clothing AND **low-contrast** (person blends into background) — the
  low-contrast case is where tiny-object detectors fail; worth documenting.
- **Angle:** nadir (90°) AND oblique (~45°) for each key scene.

A clean protocol: pick 3–4 locations; at each, place people in the pose set; fly the altitude sweep
nadir + one oblique pass. That yields a structured dataset you can annotate and report.

## 5. Honest expectations (domain gap)
Your model trained on **synthetic C2A + SARD** with their cameras/scenes. Your DJI footage is a **new
domain** (different sensor, your country's terrain, your clothing/people). Expect:
- **Good:** clear, well-separated standing people at **40–70 m nadir with SAHI**.
- **Degraded:** very-tiny (>90 m), lying-down, occluded, low-contrast, dense clusters.
This is a **qualitative real-world validation**, not a benchmark number. The altitude/SAHI choices
above maximize the chance of strong detections; some misses on the hard cases are expected and worth
reporting (they motivate future work / fine-tuning on real footage).

## 6. Legal / safety
- Stay **≤ 120 m AGL** (400 ft — the legal ceiling in most regions) and within **visual line of sight**.
- Follow local rules on flying over people; brief your "subjects," keep a safe standoff.
- Check your country's drone regulations / any permit needed for the area.

---

### One-line summary
Fly the **wide camera, 4K, NADIR, 40–70 m**, slow/hover, and **run SAHI** on the frames (without SAHI
the 6× downscale to 640 makes people too small). Capture a 20→100 m altitude sweep + one 45° oblique,
across poses (esp. lying-down), backgrounds (rubble/vegetation), occlusion, and lighting.

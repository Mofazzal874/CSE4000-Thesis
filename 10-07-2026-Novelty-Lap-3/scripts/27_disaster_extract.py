r"""
27_disaster_extract.py -- pull diverse, deduped DISASTER training frames from the news videos,
EXCLUDING the R-set eval frames (so train/eval stay clean). Output = a folder you upload to
Roboflow for SAM3-assisted annotation.

Dedup + exclusion use a perceptual difference-hash (dHash): sample frames every --every-sec,
drop any that are near-duplicate (Hamming <= --dedup-dist) to (a) a frame already kept, or
(b) any of the 25 R-set eval frames. dHash is robust to the resize/re-encode Roboflow applied,
so R-set frames are caught even though they were exported.

  python 27_disaster_extract.py --selftest
  python 27_disaster_extract.py ^
    --videos "d:\...\Drone Shoot\Footage From News(Real)\Chennai_flood_final.mp4" ^
             "d:\...\Drone Shoot\Footage From News(Real)\venezuala_final.mp4" ^
    --rset-dir "d:\...\Drone Shoot\extracted_v1\annotations\rset_v1\test" ^
    --out "d:\...\Drone Shoot\extracted_v1\disaster_train_raw" ^
    --every-sec 1.5 --max-frames 200
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

IMG_EXT = (".jpg", ".jpeg", ".png")


def dhash(img_bgr, size: int = 8) -> int:
    import cv2
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size + 1, size))
    bits = 0
    for row in range(size):
        for col in range(size):
            bits = (bits << 1) | int(g[row, col + 1] > g[row, col])
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _near_dup(h: int, pool, dist: int) -> bool:
    return any(hamming(h, p) <= dist for p in pool)


def extract(videos, rset_dir, out_dir, every_sec=1.5, dedup_dist=8, max_frames=200):
    import cv2
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # exclusion set: dHash of the 25 R-set eval frames
    rset_hashes = []
    if rset_dir:
        for p in Path(rset_dir).iterdir():
            if p.suffix.lower() in IMG_EXT:
                im = cv2.imread(str(p))
                if im is not None:
                    rset_hashes.append(dhash(im))
    print(f"[extract] loaded {len(rset_hashes)} R-set exclusion hashes")

    kept_hashes = []
    manifest = []
    for vid in videos:
        cap = cv2.VideoCapture(str(vid), cv2.CAP_FFMPEG)   # FFMPEG = reliable on Windows mp4 (GStreamer flaky)
        if not cap.isOpened():
            cap = cv2.VideoCapture(str(vid))               # fallback to default backend
        if not cap.isOpened():
            print(f"[extract] WARNING: cannot open {vid}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        nfr = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f"[extract] {Path(vid).stem}: {fps:.1f} fps, {int(nfr)} frames (~{nfr/fps:.0f}s)")
        step = max(1, int(round(fps * every_sec)))
        stem = Path(vid).stem
        i = kept_v = skip_dup = skip_rset = 0
        while len(kept_hashes) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if i % step == 0:
                h = dhash(frame)
                if _near_dup(h, rset_hashes, dedup_dist):
                    skip_rset += 1
                elif _near_dup(h, kept_hashes, dedup_dist):
                    skip_dup += 1
                else:
                    t = i / fps
                    fn = f"{stem}_t{int(t):05d}s_f{i:07d}.jpg"
                    cv2.imwrite(str(out / fn), frame)
                    kept_hashes.append(h)
                    manifest.append({"file": fn, "video": stem, "sec": round(t, 2), "frame": i})
                    kept_v += 1
            i += 1
        cap.release()
        print(f"[extract] {stem}: kept {kept_v}  (skipped {skip_dup} dup, {skip_rset} R-set-overlap)")

    (out / "extract_manifest.json").write_text(
        json.dumps({"total_kept": len(manifest), "every_sec": every_sec,
                    "dedup_dist": dedup_dist, "frames": manifest}, indent=2), encoding="utf-8")
    print(f"[extract] TOTAL kept {len(manifest)} frames -> {out}")
    print(f"[extract] upload this folder to Roboflow, SAM3-annotate ~75-100, then 22->23 to YOLO.")
    return len(manifest)


def _selftest() -> int:
    import numpy as np
    try:
        import cv2
    except ImportError:
        print("[selftest] cv2 not here -> runs on the machine with the videos (skipped)")
        print("SELFTEST OK (import-only)")
        return 0
    rng = np.random.default_rng(0)
    a = (rng.random((120, 160, 3)) * 255).astype("uint8")
    b = a.copy()
    b[0, 0] = 255 - b[0, 0]                       # tiny change -> near-duplicate
    c = (rng.random((120, 160, 3)) * 255).astype("uint8")  # different image
    ha, hb, hc = dhash(a), dhash(b), dhash(c)
    assert hamming(ha, ha) == 0
    assert hamming(ha, hb) <= 8, f"near-dup should be close: {hamming(ha, hb)}"
    assert hamming(ha, hc) > 8, f"different images should differ: {hamming(ha, hc)}"
    assert _near_dup(hb, [ha], 8) and not _near_dup(hc, [ha], 8)
    print(f"SELFTEST OK: dHash dedup works (near-dup dist {hamming(ha, hb)}, diff dist {hamming(ha, hc)})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+")
    ap.add_argument("--rset-dir", help="folder of the 25 R-set eval frames to EXCLUDE")
    ap.add_argument("--out")
    ap.add_argument("--every-sec", type=float, default=1.5)
    ap.add_argument("--dedup-dist", type=int, default=8, help="Hamming threshold for near-duplicate")
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not (a.videos and a.out):
        ap.error("--videos and --out required (or --selftest)")
    extract(a.videos, a.rset_dir, a.out, a.every_sec, a.dedup_dist, a.max_frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())

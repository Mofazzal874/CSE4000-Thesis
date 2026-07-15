"""
12_rset_extract.py — frame extractor for the RealDisaster R-set (eval-only benchmark).
Simple, deterministic, laptop-safe (CPU + OpenCV only). NOT for the own-drone footage —
that pipeline is lap-1 script 05 (segment-based, test-frame guard). This one is for
arbitrary downloaded/recorded disaster videos.

Usage:
  python 12_rset_extract.py --videos-dir "..\\..\\RealDisaster\\raw_videos" \
      --out "..\\..\\RealDisaster\\frames_v1" --every-sec 2 --max-per-video 60
  python 12_rset_extract.py --selftest     # synthesizes a tiny video, extracts, checks counts

Behavior:
- samples one frame every N seconds (default 2), capped per video (default 60)
- skips near-black and heavily blurred frames (Laplacian variance floor)
- output name: <video-stem>_t<seconds>s.jpg  (traceable to the source second)
- writes/updates <out>\\extract_manifest.json: per video -> fps, frames sampled,
  timestamps, skip counts. Re-running is idempotent (existing jpgs are not rewritten).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".mts", ".m4v", ".webm"}
BLUR_FLOOR = 25.0     # Laplacian variance below this = too blurred/black, skip
DARK_FLOOR = 8.0      # mean pixel value below this = black frame, skip


def extract_video(vp: Path, out_dir: Path, every_sec: float, max_frames: int) -> dict:
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        return {"error": "could not open"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(int(round(fps * every_sec)), 1)
    kept, skipped_quality, timestamps = 0, 0, []
    for fidx in range(0, total, step):
        if kept >= max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if float(gray.mean()) < DARK_FLOOR or float(cv2.Laplacian(gray, cv2.CV_64F).var()) < BLUR_FLOOR:
            skipped_quality += 1
            continue
        tsec = int(round(fidx / fps))
        name = f"{vp.stem}_t{tsec:05d}s.jpg"
        dst = out_dir / name
        if not dst.exists():
            cv2.imwrite(str(dst), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        kept += 1
        timestamps.append(tsec)
    cap.release()
    return {"fps": round(fps, 3), "total_frames": total, "sampled": kept,
            "skipped_quality": skipped_quality, "timestamps_s": timestamps}


def run(videos_dir: Path, out_dir: Path, every_sec: float, max_frames: int) -> int:
    vids = sorted(p for p in videos_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if not vids:
        print(f"[rset] no videos found in {videos_dir} (extensions: {sorted(VIDEO_EXTS)})")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    mpath = out_dir / "extract_manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    for vp in vids:
        print(f"[rset] {vp.name} ...")
        manifest[vp.name] = extract_video(vp, out_dir, every_sec, max_frames)
        print(f"       -> {manifest[vp.name]}")
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    n_jpg = len(list(out_dir.glob("*.jpg")))
    print(f"[rset] DONE: {n_jpg} frames in {out_dir} | manifest: {mpath}")
    print("[rset] NEXT: hand-curate (delete non-aerial/dupe/ethically-bad frames), then annotate"
          " per RealDisaster\\README_PROVENANCE.md — and fill the provenance table FIRST.")
    return 0


def selftest() -> int:
    import tempfile
    import numpy as np
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        (tdir / "vids").mkdir()
        vp = tdir / "vids" / "synthetic.mp4"
        w = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (128, 96))
        rng = np.random.default_rng(0)
        for i in range(50):  # 5 seconds @10fps: textured frames (pass blur floor)
            w.write(rng.integers(0, 255, (96, 128, 3), dtype=np.uint8))
        w.release()
        rc = run(tdir / "vids", tdir / "out", every_sec=1.0, max_frames=10)
        n = len(list((tdir / "out").glob("*.jpg")))
        ok = rc == 0 and 4 <= n <= 6  # ~5s @ 1 frame/s
        print(f"[{'PASS' if ok else 'FAIL'}] selftest: extracted {n} frames (expect ~5)")
        return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--every-sec", type=float, default=2.0)
    ap.add_argument("--max-per-video", type=int, default=60)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.videos_dir or not a.out:
        ap.error("--videos-dir and --out required (or --selftest)")
    sys.exit(run(a.videos_dir, a.out, a.every_sec, a.max_per_video))

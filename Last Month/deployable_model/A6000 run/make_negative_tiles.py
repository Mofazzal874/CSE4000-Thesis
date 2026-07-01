"""
make_negative_tiles.py  --  generate candidate HARD-NEGATIVE background tiles for angle B.
=========================================================================================
Slices 640x640 tiles out of a few frames of your drone videos and dumps them into the
`drone_negatives` folder. You then just DELETE the tiles that contain a person (in Windows
Explorer, large-thumbnail view) -- whatever is left (grass / tree foliage / pavement / banners /
tent) becomes the hard-negative set that enrich_c2a_for_robustness.py consumes.

This is far faster than hand-cropping: you only review ~60 tiles and delete the ~half with people.

RUN (PC-2, in the (2007074) venv -- it just needs OpenCV, no GPU):
    python make_negative_tiles.py
Then: open the OUT folder, set view to "Large icons", DELETE every tile that shows a person.
Keep the background-only tiles. Done -- enrich_c2a_for_robustness.py reads this same folder.
"""
import os, random, glob
import cv2
from pathlib import Path

# ----------------------------- CONFIG -----------------------------
VIDEOS_DIR       = r"D:\student_2k20\2007074\Drone Shoot"     # the 10m/30m/50m .MP4s
OUT              = r"D:\student_2k20\2007074\drone_negatives"  # candidates land here (review + delete people)
TILE             = 640
FRAMES_PER_VIDEO = 2     # sample this many frames spread through each clip
TILES_PER_FRAME  = 10    # keep this many random tiles per sampled frame
SEED             = 0
# ------------------------------------------------------------------

random.seed(SEED)

def tile_starts(total, tile):
    xs = list(range(0, max(1, total - tile + 1), tile))
    if not xs or xs[-1] != total - tile:
        xs.append(max(0, total - tile))
    return xs

def main():
    os.makedirs(OUT, exist_ok=True)
    vids = []
    for e in ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.mkv"):
        vids += glob.glob(os.path.join(VIDEOS_DIR, e))
    vids = sorted(set(vids))
    if not vids:
        print(f"[err] no videos in {VIDEOS_DIR}"); return
    saved = 0
    for vp in vids:
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            print(f"[warn] cannot open {vp} (codec?) -- skipping"); continue
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        name = Path(vp).stem
        # sample frames spread through the clip (avoid the very start/end)
        picks = [int(n * f) for f in [0.3, 0.6, 0.45, 0.75, 0.2][:FRAMES_PER_VIDEO]]
        for fi in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            H, W = frame.shape[:2]
            coords = [(x, y) for y in tile_starts(H, TILE) for x in tile_starts(W, TILE)]
            random.shuffle(coords)
            for (x, y) in coords[:TILES_PER_FRAME]:
                tile = frame[y:y + TILE, x:x + TILE]
                if tile.shape[0] < 64 or tile.shape[1] < 64:
                    continue
                fn = f"cand_{name}_f{fi}_{x}_{y}.jpg"
                cv2.imwrite(os.path.join(OUT, fn), tile, [cv2.IMWRITE_JPEG_QUALITY, 92])
                saved += 1
        cap.release()
        print(f"[ok] {name}: sampled {FRAMES_PER_VIDEO} frames")
    print(f"\n[done] wrote {saved} candidate tiles to: {OUT}")
    print("NEXT: open that folder -> View > Large icons -> DELETE every tile that shows a PERSON.")
    print("      Keep the background-only tiles (grass/trees/pavement/banners/tent). That's your")
    print("      hard-negative set. Then run enrich_c2a_for_robustness.py (it reads this folder).")

if __name__ == "__main__":
    main()

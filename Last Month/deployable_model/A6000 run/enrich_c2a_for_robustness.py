"""
enrich_c2a_for_robustness.py  --  NOVELTY ANGLE B (data enrichment; no GPU/torch needed, pure cv2/numpy)
========================================================================================================
Fixes the TWO real-world failures seen on the DJI footage:
  (B2) FALSE POSITIVES on real backgrounds (grass/foliage/banners/pavement)  -> HARD-NEGATIVE images
       (real no-person crops, added with EMPTY labels) teach the model "this texture is NOT a person".
       *** This is the direct fix for the "blindly detecting" over-count. ***
  (B1) Paste-artifact overfitting (C2A pastes humans onto backgrounds) -> MULTI-BLEND copy-paste of
       human crops (none / Gaussian-feather / seamlessClone) so the model can't key on a paste boundary.

Runs on PC-2 (or locally) against the C2A train split. Output = a new folder of enriched
images+labels you ADD to training. Includes a VERIFY mode that draws the new boxes on a few samples
so you can eyeball label correctness BEFORE training on them (don't train on bad boxes).

RUN:  python enrich_c2a_for_robustness.py
  1) first leave VERIFY_ONLY=True  -> writes ~12 sample images with boxes drawn to OUT_DIR/_verify/.
     Open them: pasted humans must sit inside their green boxes; negatives must have NO boxes.
  2) then set VERIFY_ONLY=False    -> writes the full enriched set.
"""
import os, random, glob
import cv2
import numpy as np
from pathlib import Path

# ----------------------------- CONFIG -----------------------------
C2A_TRAIN_IMG = r"D:\student_2k20\2007074\common\c2a\C2A_Dataset\new_dataset3\train\images"
C2A_TRAIN_LBL = r"D:\student_2k20\2007074\common\c2a\C2A_Dataset\new_dataset3\train\labels"
OUT_DIR       = r"D:\student_2k20\2007074\common\c2a\c2a_enriched"   # -> images/ + labels/ (+ _verify/)
# Folder of REAL no-person background images (crop ~30-60 patches of grass / tree foliage / pavement /
# banners / the tent from your drone frames -- exactly the things it false-fires on). Leave as-is and
# the script just skips negatives if the folder is empty/missing.
NEG_DIR       = r"D:\student_2k20\2007074\drone_negatives"

VERIFY_ONLY      = True     # True: only write a few box-drawn samples to _verify/ for inspection, then stop.
N_AUG_IMAGES     = 1500     # how many multi-blend copy-paste augmented images to create
PASTES_PER_IMAGE = (2, 5)   # paste this many extra human crops per augmented image (random in range)
CROP_POOL_FROM   = 800      # gather the human-crop pool from this many random C2A images
BLEND_MODES      = ["none", "gaussian", "seamless"]   # randomly chosen per paste (Dwibedi: vary it)
NEG_TILE         = 640      # tile size for negatives
NEG_PER_IMAGE    = 6        # max tiles kept per background image
SEED             = 0
VERIFY_N         = 12
# ------------------------------------------------------------------

random.seed(SEED); np.random.seed(SEED)
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")

def _list_images(d):
    out = []
    for e in IMG_EXT:
        out += glob.glob(os.path.join(d, "*" + e))
    return sorted(out)

def _read_boxes_px(lbl_path, W, H):
    """YOLO label -> list of (x1,y1,x2,y2) in pixels."""
    boxes = []
    if not os.path.isfile(lbl_path):
        return boxes
    for ln in open(lbl_path, "r").read().splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        _, cx, cy, w, h = (float(v) for v in p[:5])
        x1 = (cx - w / 2) * W; y1 = (cy - h / 2) * H
        x2 = (cx + w / 2) * W; y2 = (cy + h / 2) * H
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
    return boxes

def _to_yolo(x1, y1, x2, y2, W, H):
    cx = (x1 + x2) / 2 / W; cy = (y1 + y2) / 2 / H
    w = (x2 - x1) / W; h = (y2 - y1) / H
    return cx, cy, w, h

def _overlaps(box, others, thr=0.15):
    x1, y1, x2, y2 = box; a = (x2 - x1) * (y2 - y1)
    for ox1, oy1, ox2, oy2 in others:
        ix1 = max(x1, ox1); iy1 = max(y1, oy1); ix2 = min(x2, ox2); iy2 = min(y2, oy2)
        iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1); inter = iw * ih
        if inter and inter / max(a, 1) > thr:
            return True
    return False

def _paste(dst, crop, x, y, mode):
    ch, cw = crop.shape[:2]; H, W = dst.shape[:2]
    if x < 0 or y < 0 or x + cw > W or y + ch > H or ch < 4 or cw < 4:
        return False
    if mode == "none":
        dst[y:y + ch, x:x + cw] = crop
    elif mode == "gaussian":
        mask = np.zeros((ch, cw), np.float32); mask[2:-2, 2:-2] = 1.0
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2.0)[..., None]
        roi = dst[y:y + ch, x:x + cw].astype(np.float32)
        dst[y:y + ch, x:x + cw] = (mask * crop.astype(np.float32) + (1 - mask) * roi).astype(np.uint8)
    else:  # seamless (MIXED_CLONE); falls back to direct paste if OpenCV refuses (edge cases)
        try:
            mask = np.full((ch, cw), 255, np.uint8)
            center = (x + cw // 2, y + ch // 2)
            out = cv2.seamlessClone(crop, dst, mask, center, cv2.MIXED_CLONE)
            dst[:] = out
        except Exception:
            dst[y:y + ch, x:x + cw] = crop
    return True

def build_crop_pool():
    imgs = _list_images(C2A_TRAIN_IMG); random.shuffle(imgs)
    pool = []
    for ip in imgs[:CROP_POOL_FROM]:
        im = cv2.imread(ip)
        if im is None:
            continue
        H, W = im.shape[:2]
        for (x1, y1, x2, y2) in _read_boxes_px(os.path.join(C2A_TRAIN_LBL, Path(ip).stem + ".txt"), W, H):
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(W, x2); y2 = min(H, y2)
            if x2 - x1 >= 6 and y2 - y1 >= 6:
                pool.append(im[y1:y2, x1:x2].copy())
    print(f"[pool] gathered {len(pool)} human crops")
    return pool

def make_augmented(pool, out_img, out_lbl, verify_dir=None):
    imgs = _list_images(C2A_TRAIN_IMG); random.shuffle(imgs)
    made = 0
    for ip in imgs:
        if made >= N_AUG_IMAGES:
            break
        im = cv2.imread(ip)
        if im is None:
            continue
        H, W = im.shape[:2]
        boxes = _read_boxes_px(os.path.join(C2A_TRAIN_LBL, Path(ip).stem + ".txt"), W, H)
        new_lines = []
        # keep original boxes
        for (x1, y1, x2, y2) in boxes:
            cx, cy, w, h = _to_yolo(x1, y1, x2, y2, W, H)
            new_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        placed = list(boxes)
        k = random.randint(*PASTES_PER_IMAGE)
        for _ in range(k):
            crop = random.choice(pool); ch, cw = crop.shape[:2]
            if cw >= W or ch >= H:
                continue
            ok = False
            for _try in range(20):
                x = random.randint(0, W - cw - 1); y = random.randint(0, H - ch - 1)
                if not _overlaps([x, y, x + cw, y + ch], placed):
                    ok = True; break
            if not ok:
                continue
            mode = random.choice(BLEND_MODES)
            if _paste(im, crop, x, y, mode):
                placed.append([x, y, x + cw, y + ch])
                cx, cy, w, h = _to_yolo(x, y, x + cw, y + ch, W, H)
                new_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        stem = f"aug_{Path(ip).stem}_{made:05d}"
        cv2.imwrite(os.path.join(out_img, stem + ".jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        open(os.path.join(out_lbl, stem + ".txt"), "w").write("\n".join(new_lines) + "\n")
        if verify_dir and made < VERIFY_N:
            vis = im.copy()
            for ln in new_lines:
                _, cx, cy, w, h = (float(v) for v in ln.split())
                x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * H)
                x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * H)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(verify_dir, stem + "_BOXES.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
        made += 1
    print(f"[aug] wrote {made} multi-blend copy-paste images")

def make_negatives(out_img, out_lbl):
    if not os.path.isdir(NEG_DIR):
        print(f"[neg] NEG_DIR not found ({NEG_DIR}) -> skipping hard negatives. "
              f"Add real no-person background crops there to fix the false positives.")
        return
    bgs = _list_images(NEG_DIR)
    if not bgs:
        print(f"[neg] NEG_DIR empty -> skipping hard negatives.")
        return
    n = 0
    for bp in bgs:
        im = cv2.imread(bp)
        if im is None:
            continue
        H, W = im.shape[:2]; kept = 0
        ys = list(range(0, max(1, H - NEG_TILE + 1), NEG_TILE))
        xs = list(range(0, max(1, W - NEG_TILE + 1), NEG_TILE))
        random.shuffle(ys); random.shuffle(xs)
        for y in ys:
            for x in xs:
                if kept >= NEG_PER_IMAGE:
                    break
                tile = im[y:y + NEG_TILE, x:x + NEG_TILE]
                if tile.shape[0] < 32 or tile.shape[1] < 32:
                    continue
                stem = f"neg_{Path(bp).stem}_{n:05d}"
                cv2.imwrite(os.path.join(out_img, stem + ".jpg"), tile, [cv2.IMWRITE_JPEG_QUALITY, 92])
                open(os.path.join(out_lbl, stem + ".txt"), "w").write("")   # EMPTY label = background negative
                n += 1; kept += 1
    print(f"[neg] wrote {n} hard-negative background tiles (empty labels)")

def main():
    out_img = os.path.join(OUT_DIR, "images"); out_lbl = os.path.join(OUT_DIR, "labels")
    os.makedirs(out_img, exist_ok=True); os.makedirs(out_lbl, exist_ok=True)
    if VERIFY_ONLY:
        vdir = os.path.join(OUT_DIR, "_verify"); os.makedirs(vdir, exist_ok=True)
        pool = build_crop_pool()
        if not pool:
            print("[err] no crops gathered -- check C2A paths."); return
        global N_AUG_IMAGES
        N_AUG_IMAGES = VERIFY_N
        make_augmented(pool, out_img, out_lbl, verify_dir=vdir)
        make_negatives(out_img, out_lbl)
        print(f"\n[VERIFY] open {vdir} -- every pasted human MUST sit inside its green box. "
              f"If boxes look right, set VERIFY_ONLY=False and re-run for the full set.")
        return
    pool = build_crop_pool()
    if not pool:
        print("[err] no crops gathered -- check C2A paths."); return
    make_augmented(pool, out_img, out_lbl)
    make_negatives(out_img, out_lbl)
    print(f"\n[done] enriched set in {OUT_DIR}. Next: ADD these images to the joint train list and "
          f"fine-tune from epoch125.pt (low LR). I'll wire that step next.")

if __name__ == "__main__":
    main()

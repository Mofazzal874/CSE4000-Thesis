"""
run_on_drone_footage.py
=======================
Run the deployable joint C2A+SARD model on your real DJI Air 3S footage (10m/30m/50m, 4K).

WHY SLICED (SAHI-style) INFERENCE: the frames are 3840x2160; the model runs at 640. Feeding a whole
4K frame (resized to 640) shrinks a person ~6x -> tiny people vanish. So we cut each frame into
overlapping 640 tiles, detect per tile at NATIVE resolution, offset boxes back, and merge with NMS.
For comparison we ALSO run a whole-frame pass (imgsz=1280) so you can SHOW the difference in the report.

OUTPUTS (under this script's folder -> drone_inference_out/<video>/):
  sliced_fXXXXXX.jpg   annotated frames (green boxes) from SLICED inference  <- the real result
  whole_fXXXXXX.jpg    annotated frames from WHOLE-frame inference           <- the weak baseline
  counts.csv           per-frame: frame_id, sliced_count, whole_count
  + a printed summary: avg people/frame per altitude, sliced vs whole

RUN (PC-2, in the (2007074) venv, from the A6000 run folder):
  1. Transfer your 3 videos to VIDEOS_DIR below (e.g. D:/student_2k20/2007074/Drone Shoot).
  2. python run_on_drone_footage.py
  (If a video won't open -- HEVC codec -- extract frames to a folder and put that folder path in
   INPUTS instead of the .mp4; the script auto-handles a folder of images too.)
"""
import joint_c2a_sard_train as J        # sets up GPU pick, torch, mem-cap, CBAM registration
# the checkpoint pickles CBAM under '__main__' -> bind here so torch.load can unpickle it:
from joint_c2a_sard_train import CBAM, ChannelAttention, SpatialAttention  # noqa: F401
import torch, cv2, csv
import numpy as np
from pathlib import Path

# ----------------------------- CONFIG -----------------------------
MODEL = r"D:\student_2k20\2007074\A6000 run\runs_joint\20260627_162506_cbam_p2head_joint_c2a_sard\ultra\weights\epoch125.pt"
# Folder holding the transferred videos (10m.MP4 / 30m.MP4 / 50m.MP4). The script auto-finds *.mp4/*.MP4.
VIDEOS_DIR = r"D:\student_2k20\2007074\Drone Shoot"
# OR set INPUTS explicitly to a mix of video files and/or frame-folders (overrides VIDEOS_DIR if non-empty):
INPUTS = []

SAMPLE_EVERY_SEC = 2.0      # sample ~1 frame every 2 s (so ~15 frames per 30 s clip)
TILE          = 640         # slice size (matches the model input -> native-res people)
OVERLAP       = 0.20        # 20% tile overlap so people on tile borders aren't cut
CONF          = 0.35        # raised from 0.25 -> cuts low-confidence background false positives
                            # (grass/foliage/shadow texture fired as "people"). For angle-A PSEUDO-LABELS
                            # use 0.45-0.50 (high precision). Drop toward 0.25 only to chase max recall.
NMS_IOU       = 0.55        # merge duplicate boxes from overlapping tiles
MIN_BOX_SIDE_PX = 14        # drop boxes whose SHORTER side < this (in 4K px) -> kills the tiny noise
                            # boxes on empty grass/debris. Real people are >=~30 px even at 50 m, so safe.
WHOLE_IMGSZ   = 1280        # whole-frame baseline resolution (still downsamples 4K ~3x -> misses tiny)
DO_WHOLE_BASELINE = True    # also run whole-frame inference for the SAHI-vs-naive comparison figure
MAX_FRAMES_PER_VIDEO = 25   # cap, so it stays quick
# ------------------------------------------------------------------

try:
    from torchvision.ops import nms as _tv_nms
    def _nms(boxes, scores, iou):
        keep = _tv_nms(torch.from_numpy(boxes).float(), torch.from_numpy(scores).float(), iou)
        return keep.cpu().numpy()
except Exception:
    def _nms(boxes, scores, iou):  # numpy fallback
        x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
        areas = (x2-x1)*(y2-y1); order = scores.argsort()[::-1]; keep=[]
        while order.size:
            i = order[0]; keep.append(i)
            xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
            xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
            w=np.maximum(0,xx2-xx1); h=np.maximum(0,yy2-yy1); inter=w*h
            iou_=inter/(areas[i]+areas[order[1:]]-inter+1e-9)
            order = order[1:][iou_<=iou]
        return np.array(keep, dtype=int)


def _drop_tiny(boxes, scores):
    """Remove boxes whose shorter side < MIN_BOX_SIDE_PX (the tiny grass/foliage false positives)."""
    if len(boxes) == 0:
        return boxes, scores
    sides = np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    keep = sides >= MIN_BOX_SIDE_PX
    return boxes[keep], scores[keep]

def _tile_starts(total, tile, step):
    if total <= tile:
        return [0]
    xs = list(range(0, total - tile + 1, step))
    if xs[-1] != total - tile:
        xs.append(total - tile)
    return xs

def sliced_predict(model, img):
    H, W = img.shape[:2]
    step = max(1, int(TILE * (1 - OVERLAP)))
    boxes, scores = [], []
    for y in _tile_starts(H, TILE, step):
        for x in _tile_starts(W, TILE, step):
            crop = img[y:y+TILE, x:x+TILE]
            r = model.predict(crop, conf=CONF, imgsz=TILE, verbose=False)[0].boxes
            if r is None or len(r) == 0:
                continue
            b = r.xyxy.cpu().numpy().astype(np.float32); s = r.conf.cpu().numpy().astype(np.float32)
            b[:, [0, 2]] += x; b[:, [1, 3]] += y
            boxes.append(b); scores.append(s)
    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    boxes = np.concatenate(boxes); scores = np.concatenate(scores)
    keep = _nms(boxes, scores, NMS_IOU)
    return _drop_tiny(boxes[keep], scores[keep])

def whole_predict(model, img):
    r = model.predict(img, conf=CONF, imgsz=WHOLE_IMGSZ, verbose=False)[0].boxes
    if r is None or len(r) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    return _drop_tiny(r.xyxy.cpu().numpy().astype(np.float32), r.conf.cpu().numpy().astype(np.float32))

def draw(img, boxes, label, color):
    out = img.copy()
    for (x1, y1, x2, y2) in boxes.astype(int):
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(out, (0, 0), (560, 60), (0, 0, 0), -1)
    cv2.putText(out, f"{label}: {len(boxes)} people", (12, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    return out

def iter_frames(path):
    p = Path(path)
    if p.is_dir():
        imgs = sorted([f for f in p.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
        for i, f in enumerate(imgs):
            im = cv2.imread(str(f))
            if im is not None:
                yield i, im
        return
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        print(f"[warn] cannot open video {p} (HEVC codec issue?). Extract its frames to a folder and "
              f"put that folder in INPUTS instead.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    every = max(1, int(round(fps * SAMPLE_EVERY_SEC)))
    idx = saved = 0
    while saved < MAX_FRAMES_PER_VIDEO:
        ok, fr = cap.read()
        if not ok:
            break
        if idx % every == 0:
            yield idx, fr; saved += 1
        idx += 1
    cap.release()

def main():
    inputs = INPUTS if INPUTS else sorted(
        [str(p) for p in Path(VIDEOS_DIR).glob("*") if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")]
    ) if Path(VIDEOS_DIR).is_dir() else []
    if not inputs:
        print(f"[err] no inputs. Put videos in {VIDEOS_DIR} or set INPUTS. "); return
    if not Path(MODEL).is_file():
        print(f"[err] model not found: {MODEL}"); return

    print(f"[model] loading {MODEL}")
    model = J.YOLO(str(MODEL))
    if torch.cuda.is_available():
        model.to("cuda:0")

    out_root = J.SCRIPT_DIR / "drone_inference_out"
    out_root.mkdir(parents=True, exist_ok=True)
    grand = []
    for inp in inputs:
        name = Path(inp).stem
        odir = out_root / name; odir.mkdir(parents=True, exist_ok=True)
        rows = []
        print(f"\n[video] {name}  ({inp})")
        for fid, frame in iter_frames(inp):
            sb, _ = sliced_predict(model, frame)
            cv2.imwrite(str(odir / f"sliced_f{fid:06d}.jpg"),
                        draw(frame, sb, "SLICED", (0, 255, 0)), [cv2.IMWRITE_JPEG_QUALITY, 90])
            wc = 0
            if DO_WHOLE_BASELINE:
                wb, _ = whole_predict(model, frame); wc = len(wb)
                cv2.imwrite(str(odir / f"whole_f{fid:06d}.jpg"),
                            draw(frame, wb, "WHOLE-640/1280", (0, 165, 255)), [cv2.IMWRITE_JPEG_QUALITY, 90])
            rows.append((fid, len(sb), wc))
            print(f"   frame {fid:6d}: sliced={len(sb):3d}  whole={wc:3d}")
        with open(odir / "counts.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["frame_id", "sliced_count", "whole_count"]); w.writerows(rows)
        if rows:
            savg = sum(r[1] for r in rows) / len(rows); wavg = sum(r[2] for r in rows) / len(rows)
            grand.append((name, len(rows), savg, wavg))

    print("\n" + "=" * 70)
    print("DRONE FOOTAGE INFERENCE SUMMARY (avg people detected per frame)")
    print(f"{'video':<10}{'frames':>8}{'SLICED':>10}{'WHOLE':>10}{'sliced gain':>13}")
    for name, n, savg, wavg in grand:
        gain = f"+{savg - wavg:.1f}" if savg >= wavg else f"{savg - wavg:.1f}"
        print(f"{name:<10}{n:>8}{savg:>10.1f}{wavg:>10.1f}{gain:>13}")
    print("=" * 70)
    print(f"[done] annotated frames + counts.csv in: {out_root}")
    print("       SLICED should detect MANY more people than WHOLE, especially at 30m/50m -> that gap")
    print("       is your 'why SAHI matters' figure. Eyeball the sliced_*.jpg to judge real-world quality.")

if __name__ == "__main__":
    main()

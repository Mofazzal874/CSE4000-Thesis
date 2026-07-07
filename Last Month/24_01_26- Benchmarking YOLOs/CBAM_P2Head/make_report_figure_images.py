"""
make_report_figure_images.py  --  generate the REMAINING report-figure rasters (CBAM+P2 model).

Complements make_arch_figure_images.py (which makes arch_input / arch_detections /
cbam_overlay / p2_featuremap = FIGURE_PLAN #10-13). THIS script makes:

  FIGURE_PLAN #14  detgrid_c2a_s4.png / detgrid_c2a_s8.png (+ _full variants)
                   real C2A detections + stride-4 / stride-8 grid overlay
                   -> real-data companion of fig_stride_problem.drawio
  FIGURE_PLAN #15  sahi_input_full.png   -> fig_sahi.drawio slot "PASTE: full image" (>=260x220)
                   sahi_slice_grid.png    (input with the 512/0.25 slice grid drawn -- optional inset)
                   sahi_merged_detections.png -> slot "PASTE: merged detections" (>=200x220)
  FIGURE_PLAN #16  tta_input.png         -> fig_tta.drawio slot "PASTE: input image" (>=240x220)
                   tta_detections.png    -> slot "PASTE: detections" (>=220x220)
  ZOOM variants    *_zoom.png of every detection image (people-cluster crop) --
                   use whichever reads better in the slot.

All outputs are 640x640 (zoom crops native-crop size, >=300 px) -- every draw.io slot
needs at most 720x180 @2x, so 640-square exceeds all minimums.

RUN (GPU PC, venv active, from the CBAM_P2Head folder):
    python make_report_figure_images.py
Outputs -> ./report_fig_out/   then copy into Defense/draft1_30_6_26/figures/.
Auto-finds best.pt (canonical run) + the C2A test image; installs sahi if missing.
"""
import glob, subprocess, sys, json
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn

# ============ CONFIG ============
IMG_NAME  = "collapsed_building_image0001_3.png"  # same image as the arch set (consistency)
IMGSZ     = 640
CONF      = 0.25
ZOOM_CROP = (180, 120, 520, 470)   # people-cluster crop in 640-space -> *_zoom.png (340x350 px)
GRID_STRIDES = [4, 8]              # detgrid variants (P2 and P3)
SAHI_SLICE, SAHI_OVERLAP, SAHI_CONF = 512, 0.25, 0.15   # per FIGURE_PLAN Sec 3.3
TTA_IMGSZ = 1280

# ============ CBAM (verbatim, so best.pt unpickles) ============
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1); self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        a = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        m = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return x * self.sigmoid(a + m)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=3 if kernel_size == 7 else 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        a = torch.mean(x, dim=1, keepdim=True)
        m, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([a, m], dim=1)))

class CBAM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.reduction = 16; self.kernel_size = 7
        if len(args) == 1 and isinstance(args[0], int) and args[0] <= 32:
            self.reduction = args[0]
        elif len(args) == 2 and isinstance(args[0], int) and args[0] <= 32:
            self.reduction = args[0]; self.kernel_size = args[1] if isinstance(args[1], int) else 7
        elif len(args) >= 4:
            self.reduction = args[2] if isinstance(args[2], int) else 16
            self.kernel_size = args[3] if isinstance(args[3], int) else 7
        self.reduction = kwargs.get("reduction", self.reduction)
        self.kernel_size = kwargs.get("kernel_size", self.kernel_size)
        if self.kernel_size not in (3, 7):
            self.kernel_size = 7
        self._initialized = False
        self.channel_attention = None; self.spatial_attention = None; self._channels = None
    def _lazy_init(self, channels, device, dtype):
        self._channels = channels
        self.channel_attention = ChannelAttention(channels, self.reduction).to(device=device, dtype=dtype)
        self.spatial_attention = SpatialAttention(self.kernel_size).to(device=device, dtype=dtype)
        self._initialized = True
    def forward(self, x):
        if not self._initialized:
            self._lazy_init(x.size(1), x.device, x.dtype)
        return self.spatial_attention(self.channel_attention(x))

def register_cbam():
    import ultralytics.nn.modules as m, ultralytics.nn.tasks as t
    for ns in (m, t):
        ns.CBAM = CBAM; ns.ChannelAttention = ChannelAttention; ns.SpatialAttention = SpatialAttention
register_cbam()

try:
    import sahi  # noqa: F401
except ImportError:
    print("[deps] installing sahi ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "sahi"])
from ultralytics import YOLO

# ============ locate weights + image ============
SCRIPT_DIR = Path(__file__).resolve().parent
def newest(paths): return max(paths, key=lambda p: Path(p).stat().st_mtime)
wcand = glob.glob(str(SCRIPT_DIR / "runs" / "**" / "weights" / "best.pt"), recursive=True)
WEIGHTS = Path(newest(wcand)) if wcand else None
if WEIGHTS is None:   # PC-4 / laptop: no runs/ tree -> known standalone CBAM+P2 copies
    for _k in [r"D:\thesis_2007074\c2a_cbam_p2head_best.pt",
               r"E:\Thesis_mofazzal_2007074\c2a_cbam_p2head_best.pt",
               r"D:\Academics\thesis folder\Last Month\deployable_model\c2a_cbam_p2head_best.pt"]:
        if Path(_k).is_file():
            WEIGHTS = Path(_k); break
IMG = None
for base in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
    hit = glob.glob(str(base / "**" / "test" / "images" / IMG_NAME), recursive=True)
    if hit:
        IMG = Path(hit[0]); break
assert WEIGHTS and WEIGHTS.is_file(), "best.pt not found under runs/ -- set WEIGHTS manually"
assert IMG and IMG.is_file(), f"{IMG_NAME} not found under any parent -- set IMG manually"
OUT = SCRIPT_DIR / "report_fig_out"; OUT.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[cfg] weights = {WEIGHTS}\n[cfg] image   = {IMG}\n[cfg] out     = {OUT}\n[cfg] device  = {DEVICE}")

manifest = []
def save(img, name, note=""):
    cv2.imwrite(str(OUT / name), img)
    manifest.append((name, f"{img.shape[1]}x{img.shape[0]}", note))
    print(f"  [out] {name}  ({img.shape[1]}x{img.shape[0]})  {note}")

def save_with_zoom(img, name, note=""):
    save(img, name, note)
    x1, y1, x2, y2 = ZOOM_CROP
    save(img[y1:y2, x1:x2].copy(), name.replace(".png", "_zoom.png"), note + " [zoom crop]")

def draw_boxes(base, boxes_xyxy, color=(0, 200, 0), thick=1):
    out = base.copy()
    for (x1, y1, x2, y2) in np.asarray(boxes_xyxy, np.int32):
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
    return out

yolo = YOLO(str(WEIGHTS))
raw = cv2.imread(str(IMG))
img640 = cv2.resize(raw, (IMGSZ, IMGSZ))
H0, W0 = raw.shape[:2]
sx, sy = IMGSZ / W0, IMGSZ / H0

# ============ 1. FIGURE_PLAN #14 -- detection grid (stride 4 / 8) ============
print("\n[1/3] detection-grid images (#14)")
res = yolo.predict(str(IMG), imgsz=IMGSZ, conf=CONF, verbose=False)
det640 = cv2.resize(res[0].plot(line_width=1, labels=False, conf=False), (IMGSZ, IMGSZ))
for stride in GRID_STRIDES:
    g = det640.copy()
    for v in range(0, IMGSZ + 1, stride):
        cv2.line(g, (v, 0), (v, IMGSZ), (210, 210, 210), 1)
        cv2.line(g, (0, v), (IMGSZ, v), (210, 210, 210), 1)
    # grid drawn over detections at 30% so boxes stay readable
    g = cv2.addWeighted(det640, 0.70, g, 0.30, 0)
    save(g, f"detgrid_c2a_s{stride}_full.png", f"stride-{stride} grid, full frame")
    x1, y1, x2, y2 = ZOOM_CROP
    save(g[y1:y2, x1:x2].copy(), f"detgrid_c2a_s{stride}.png",
         f"stride-{stride} grid, cluster zoom -> use in report")

# ============ 2. FIGURE_PLAN #15 -- SAHI insets ============
print("\n[2/3] SAHI insets (#15)  (slice=512, overlap=0.25, GREEDYNMM/IOS)")
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
sm = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(WEIGHTS),
                                        confidence_threshold=SAHI_CONF, device=DEVICE)
r = get_sliced_prediction(str(IMG), sm, slice_height=SAHI_SLICE, slice_width=SAHI_SLICE,
                          overlap_height_ratio=SAHI_OVERLAP, overlap_width_ratio=SAHI_OVERLAP,
                          perform_standard_pred=True, postprocess_type="GREEDYNMM",
                          postprocess_match_metric="IOS", postprocess_match_threshold=0.5, verbose=0)
merged = [[o.bbox.minx * sx, o.bbox.miny * sy, o.bbox.maxx * sx, o.bbox.maxy * sy]
          for o in r.object_prediction_list]
save(img640.copy(), "sahi_input_full.png", "-> fig_sahi slot 'full image'")
# slice-grid version of the input (shows the real 512/0.25 tiling in original-image space)
grid = img640.copy()
step = int(SAHI_SLICE * (1 - SAHI_OVERLAP))
xs = list(range(0, max(W0 - SAHI_SLICE, 0) + 1, step)) or [0]
ys = list(range(0, max(H0 - SAHI_SLICE, 0) + 1, step)) or [0]
if xs[-1] != max(W0 - SAHI_SLICE, 0): xs.append(max(W0 - SAHI_SLICE, 0))
if ys[-1] != max(H0 - SAHI_SLICE, 0): ys.append(max(H0 - SAHI_SLICE, 0))
for gx in xs:
    for gy in ys:
        p1 = (int(gx * sx), int(gy * sy))
        p2 = (int(min(gx + SAHI_SLICE, W0) * sx), int(min(gy + SAHI_SLICE, H0) * sy))
        cv2.rectangle(grid, p1, p2, (0, 165, 255), 2)
save(grid, "sahi_slice_grid.png", f"{len(xs)*len(ys)} tiles of {SAHI_SLICE}px, overlap {SAHI_OVERLAP}")
save_with_zoom(draw_boxes(img640, merged), "sahi_merged_detections.png",
               f"-> fig_sahi slot 'merged detections' ({len(merged)} boxes)")
del sm

# ============ 3. FIGURE_PLAN #16 -- TTA insets ============
print("\n[3/3] TTA insets (#16)  (imgsz=1280, augment=True)")
save(img640.copy(), "tta_input.png", "-> fig_tta slot 'input image'")
try:
    rt = yolo.predict(str(IMG), imgsz=TTA_IMGSZ, conf=CONF, augment=True, verbose=False)
    b = rt[0].boxes
    tta_boxes = (b.xyxy.cpu().numpy() * [sx, sy, sx, sy]) if b is not None and len(b) else []
    save_with_zoom(draw_boxes(img640, tta_boxes), "tta_detections.png",
                   f"-> fig_tta slot 'detections' ({len(tta_boxes)} boxes @1280+TTA)")
except torch.cuda.OutOfMemoryError:
    print("  [warn] OOM at 1280+TTA -- retrying at 960")
    rt = yolo.predict(str(IMG), imgsz=960, conf=CONF, augment=True, verbose=False)
    b = rt[0].boxes
    tta_boxes = (b.xyxy.cpu().numpy() * [sx, sy, sx, sy]) if b is not None and len(b) else []
    save_with_zoom(draw_boxes(img640, tta_boxes), "tta_detections.png",
                   f"-> fig_tta slot 'detections' ({len(tta_boxes)} boxes @960+TTA)")

# ============ manifest: file -> drawio slot -> minimum px ============
slot_map = {
    "detgrid_c2a_s4.png":          ("standalone figure (companion of fig_stride_problem)", "-"),
    "detgrid_c2a_s8.png":          ("standalone figure (companion of fig_stride_problem)", "-"),
    "sahi_input_full.png":         ("fig_sahi.drawio  slot 'PASTE: full image'", ">=260x220"),
    "sahi_merged_detections.png":  ("fig_sahi.drawio  slot 'PASTE: merged detections'", ">=200x220"),
    "sahi_slice_grid.png":         ("fig_sahi.drawio  optional inset (real tiling)", "-"),
    "tta_input.png":               ("fig_tta.drawio   slot 'PASTE: input image'", ">=240x220"),
    "tta_detections.png":          ("fig_tta.drawio   slot 'PASTE: detections'", ">=220x220"),
}
lines = ["file | size | destination | min px needed", "---|---|---|---"]
for name, size, note in manifest:
    dest, minpx = slot_map.get(name, ("(zoom/full variant -- use if it reads better)", "-"))
    lines.append(f"{name} | {size} | {dest} | {minpx}")
(OUT / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")
print("\n[done] all images in", OUT)
print("       MANIFEST.md maps each file to its draw.io slot + minimum size.")
print("       Copy the ones you use into Defense/draft1_30_6_26/figures/")

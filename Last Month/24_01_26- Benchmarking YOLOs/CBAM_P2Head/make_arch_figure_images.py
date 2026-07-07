"""
make_arch_figure_images.py  --  generate the 4 images for the architecture figure.

Outputs (in ./arch_fig_out/):
  arch_input.png       input test image (640x640)
  arch_detections.png  model predictions (boxes) on the SAME image
  cbam_overlay.png     CBAM spatial-attention heat-map overlaid on the image
  p2_featuremap.png    P2 (layer-19, stride-4) feature response heat-map

WHERE TO RUN:
  On a machine with your training env (ultralytics + torch). Put this file in the
  CBAM_P2Head folder (same folder that has runs/.../weights/best.pt) and run:
      python make_arch_figure_images.py
  Then copy the 4 PNGs from arch_fig_out/ into Defense/draft1_30_6_26/figures/.

The CBAM classes are defined here (verbatim) so YOLO(best.pt) can unpickle the
custom layers -- this file must be the __main__ script (just run it directly).
"""
import glob
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn

# ============ CBAM (verbatim, so best.pt unpickles) ============
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
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
        pad = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
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
        if self.kernel_size not in (3, 7): self.kernel_size = 7
        self._initialized = False
        self.channel_attention = None
        self.spatial_attention = None
        self._channels = None
    def _lazy_init(self, channels, device, dtype):
        self._channels = channels
        self.channel_attention = ChannelAttention(channels, self.reduction).to(device=device, dtype=dtype)
        self.spatial_attention = SpatialAttention(self.kernel_size).to(device=device, dtype=dtype)
        self._initialized = True
    def forward(self, x):
        if not self._initialized:
            self._lazy_init(x.size(1), x.device, x.dtype)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

def register_cbam():
    import ultralytics.nn.modules as m, ultralytics.nn.tasks as t
    for ns in (m, t):
        ns.CBAM = CBAM; ns.ChannelAttention = ChannelAttention; ns.SpatialAttention = SpatialAttention
register_cbam()

from ultralytics import YOLO

# ============ CONFIG (edit only if auto-find fails) ============
SCRIPT_DIR = Path(__file__).resolve().parent
IMG_NAME   = "collapsed_building_image0001_3.png"   # same image used as the dataset sample
IMGSZ      = 640
CONF       = 0.25
CROP       = None   # e.g. (180, 120, 520, 470) in 640-space to zoom into a people cluster

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
    if hit: IMG = Path(hit[0]); break

assert WEIGHTS and WEIGHTS.is_file(), "best.pt not found under runs/; set WEIGHTS manually"
assert IMG and IMG.is_file(), f"{IMG_NAME} not found; set IMG manually"
OUT = SCRIPT_DIR / "arch_fig_out"; OUT.mkdir(exist_ok=True)
print("[cfg] weights =", WEIGHTS)
print("[cfg] image   =", IMG)
print("[cfg] out     =", OUT)

def save(img, name):
    if CROP:
        x1, y1, x2, y2 = CROP; img = img[y1:y2, x1:x2]
    cv2.imwrite(str(OUT / name), img)

def heat(base_bgr, attn2d, alpha=0.5, cmap=cv2.COLORMAP_JET):
    a = attn2d.astype(np.float32); a -= a.min(); a /= (a.max() + 1e-8)
    a = cv2.resize((a * 255).astype(np.uint8), (base_bgr.shape[1], base_bgr.shape[0]))
    return cv2.addWeighted(base_bgr, 1 - alpha, cv2.applyColorMap(a, cmap), alpha, 0)

# ============ load model ============
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo = YOLO(str(WEIGHTS))
dm = yolo.model.to(device).eval()          # DetectionModel
layers = dm.model                          # nn.Sequential

# ---- 1. input ----
raw = cv2.imread(str(IMG))
img640 = cv2.resize(raw, (IMGSZ, IMGSZ))
save(img640.copy(), "arch_input.png")

# ---- 2. detections (thin boxes, NO label text so tiny objects stay readable) ----
res = yolo.predict(str(IMG), imgsz=IMGSZ, conf=CONF, verbose=False)
det = res[0].plot(line_width=1, labels=False, conf=False)
det = cv2.resize(det, (IMGSZ, IMGSZ))
save(det, "arch_detections.png")

# ---- 3 & 4. hook CBAM spatial map (layer 10) and P2 feature (layer 19) ----
ten = torch.from_numpy(img640[:, :, ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0).div(255).to(device)
with torch.no_grad():
    dm(ten)                                # warm-up so CBAM lazy-init builds spatial_attention
store = {}
h1 = layers[10].spatial_attention.sigmoid.register_forward_hook(
    lambda mod, i, o: store.__setitem__("cbam", o.detach().float().cpu()))
h2 = layers[19].register_forward_hook(
    lambda mod, i, o: store.__setitem__("p2", (o[0] if isinstance(o, (list, tuple)) else o).detach().float().cpu()))
with torch.no_grad():
    dm(ten)
h1.remove(); h2.remove()

save(heat(img640.copy(), store["cbam"][0, 0].numpy(), alpha=0.5), "cbam_overlay.png")

p2 = store["p2"][0].mean(0).numpy()        # (160,160) mean over channels
p2 = p2 - p2.min(); p2 = p2 / (p2.max() + 1e-8)
p2img = cv2.applyColorMap(cv2.resize((p2 * 255).astype(np.uint8), (IMGSZ, IMGSZ),
                                     interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_VIRIDIS)
save(p2img, "p2_featuremap.png")

print("\n[done] 4 images written to", OUT)
print("       arch_input.png  arch_detections.png  cbam_overlay.png  p2_featuremap.png")
print("       copy them into Defense/draft1_30_6_26/figures/")

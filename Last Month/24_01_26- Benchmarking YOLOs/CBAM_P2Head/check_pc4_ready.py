"""
check_pc4_ready.py -- PRE-FLIGHT for running the SAHI+TTA eval on PC-4. Read-only, no GPU work.
Proves: (a) the venv + torch/ultralytics/sahi are usable, (b) the C2A test split is present,
(c) the .pt is genuinely the CBAM+P2 model (CBAM layer + 4-scale P2 head + ~19.57M params) and
NOT the 151 MB joint epoch125.pt. Prints a clear READY / NOT-READY verdict.

RUN (PC-4, venv 2007074 active):
    python check_pc4_ready.py
"""
import os, sys, hashlib
from pathlib import Path
import torch, torch.nn as nn

MODEL_CANDIDATES = [
    r"D:\thesis_2007074\c2a_cbam_p2head_best.pt",                  # the CBAM+P2 C2A model (CORRECT)
    r"D:\thesis_2007074\deployable_model\c2a_cbam_p2head_best.pt",
]
C2A_CANDIDATES = [
    r"D:\thesis_2007074\common\c2a\C2A_Dataset\new_dataset3",
    r"D:\Academics\thesis folder\c2a\C2A_Dataset\new_dataset3",
]

# --- CBAM (verbatim) so YOLO(best.pt) can unpickle the custom layers ---
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        r = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1); self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, r, 1, bias=False); self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(r, channels, 1, bias=False); self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        a = self.fc2(self.relu(self.fc1(self.avg_pool(x)))); m = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return x * self.sigmoid(a + m)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=3 if kernel_size == 7 else 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        a = torch.mean(x, 1, keepdim=True); m, _ = torch.max(x, 1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([a, m], 1)))
class CBAM(nn.Module):
    def __init__(self, *a, **k):
        super().__init__()
        self.reduction = 16; self.kernel_size = 7
        if len(a) == 1 and isinstance(a[0], int) and a[0] <= 32: self.reduction = a[0]
        elif len(a) == 2 and isinstance(a[0], int) and a[0] <= 32:
            self.reduction = a[0]; self.kernel_size = a[1] if isinstance(a[1], int) else 7
        elif len(a) >= 4:
            self.reduction = a[2] if isinstance(a[2], int) else 16
            self.kernel_size = a[3] if isinstance(a[3], int) else 7
        self.reduction = k.get("reduction", self.reduction); self.kernel_size = k.get("kernel_size", self.kernel_size)
        if self.kernel_size not in (3, 7): self.kernel_size = 7
        self._initialized = False; self.channel_attention = None; self.spatial_attention = None; self._channels = None
    def _lazy_init(self, c, d, t):
        self._channels = c
        self.channel_attention = ChannelAttention(c, self.reduction).to(device=d, dtype=t)
        self.spatial_attention = SpatialAttention(self.kernel_size).to(device=d, dtype=t); self._initialized = True
    def forward(self, x):
        if not self._initialized: self._lazy_init(x.size(1), x.device, x.dtype)
        return self.spatial_attention(self.channel_attention(x))
def register_cbam():
    import ultralytics.nn.modules as m, ultralytics.nn.tasks as t
    for ns in (m, t):
        ns.CBAM = CBAM; ns.ChannelAttention = ChannelAttention; ns.SpatialAttention = SpatialAttention

def first(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None

def count(d):
    return sum(1 for p in Path(d).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")) if Path(d).is_dir() else 0

ok = True
print("=" * 70 + "\nPC-4 PRE-FLIGHT for SAHI+TTA eval\n" + "=" * 70)

# 1. env
print("\n[1] environment")
print(f"    python     : {sys.version.split()[0]}  ({sys.executable})")
print(f"    torch      : {torch.__version__}  | cuda avail: {torch.cuda.is_available()}"
      + (f"  | GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
try:
    import ultralytics; print(f"    ultralytics: {ultralytics.__version__}")
except Exception as e:
    print(f"    ultralytics: MISSING ({e})"); ok = False
try:
    import sahi; print(f"    sahi       : {sahi.__version__}")
except Exception:
    print("    sahi       : NOT INSTALLED  ->  run:  pip install sahi   (then re-check)"); ok = False
if not torch.cuda.is_available():
    print("    WARNING: CUDA not available -- eval will be very slow on CPU.");

# 2. C2A dataset
print("\n[2] C2A dataset")
c2a = first(C2A_CANDIDATES)
if c2a is None:
    print(f"    NOT FOUND in {C2A_CANDIDATES}"); ok = False
else:
    nt = count(c2a / "test" / "images"); ntr = count(c2a / "train" / "images"); nv = count(c2a / "val" / "images")
    print(f"    root : {c2a}")
    print(f"    test : {nt} imgs   train : {ntr}   val : {nv}")
    if nt < 2000: print(f"    WARNING: expected ~2043 test images, found {nt}"); ok = False

# 3. the model -- prove it is CBAM+P2, not the joint model
print("\n[3] model check (must be CBAM+P2, ~19.57M params, 4-scale P2 head)")
mp = first(MODEL_CANDIDATES)
if mp is None:
    print(f"    c2a_cbam_p2head_best.pt NOT FOUND in {MODEL_CANDIDATES}"); ok = False
else:
    size_mb = mp.stat().st_size / 1024**2
    h = hashlib.md5(mp.read_bytes()).hexdigest()
    print(f"    file : {mp}")
    print(f"    size : {size_mb:.2f} MB   md5: {h}")
    if size_mb > 60:
        print(f"    *** {size_mb:.0f} MB is TOO BIG for CBAM+P2 (~38 MB). This looks like the JOINT")
        print("        epoch125.pt (151 MB) or a full-optimizer ckpt -- WRONG file. ***"); ok = False
    try:
        register_cbam()
        from ultralytics import YOLO
        m = YOLO(str(mp))
        params = sum(p.numel() for p in m.model.parameters()) / 1e6
        layer_types = [type(x).__name__ for x in m.model.modules()]
        has_cbam = "CBAM" in layer_types
        det = [x for x in m.model.modules() if type(x).__name__ == "Detect"]
        n_scales = None
        if det:
            d = det[-1]
            n_scales = int(getattr(d, "nl", 0)) or (len(d.stride) if hasattr(d, "stride") and d.stride is not None else None)
        names = m.names
        print(f"    params        : {params:.2f} M   (expect ~19.57)")
        print(f"    CBAM present  : {has_cbam}")
        print(f"    detect scales : {n_scales}   (P2 head => 4; baseline => 3)")
        print(f"    classes       : {names}")
        verdict_ok = has_cbam and (n_scales == 4) and (18.5 < params < 20.5)
        if not verdict_ok:
            print("    *** architecture does NOT match CBAM+P2 -- check the file. ***"); ok = False
    except Exception as e:
        print(f"    FAILED to load/inspect model: {e}"); ok = False

# verdict
print("\n" + "=" * 70)
print("VERDICT: READY -- run:  python sahi_tta_cbam_p2_thesis.py" if ok
      else "VERDICT: NOT READY -- fix the items marked above, then re-run this check.")
print("=" * 70)
sys.exit(0 if ok else 1)

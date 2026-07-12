"""
10_fccg_modules.py — FCCG-YOLO S0 modules + selftests (lap 3, Option A)
========================================================================
Context-Gated Evidence fusion for tiny/occluded aerial humans on the
YOLO11m CBAM+P2 lineage. Design constraints come from
  docs/2026-07-12_P0_rival_differentiation.md  (read it before editing):
  - gate DIRECTION: coarse semantic context -> fine evidence (no rival does this;
    DERNet gates HF->LL intra-level, SRTSOD self-gates one stream, AFGLFF
    reweights wavelet sub-bands).
  - evidence is LEARNABLE + spatial-domain (X - AvgPool residual + learnable DW
    bank), NOT fixed Haar/Log-Gabor bases (DERNet owns those).
  - headline word is "context-gated evidence", not "frequency".

House rules baked in (Mamba/CBAM/pickle lessons):
  * every nn.Module class is MODULE TOP-LEVEL (checkpoint pickle safety)
  * lazy channel init on first forward (same trick as thesis CBAM: bypasses
    ultralytics parse_model channel bookkeeping; materialized during YOLO()'s
    stride dummy-forward, i.e. BEFORE the optimizer is built)
  * modules are SHAPE-PRESERVING (out channels == in channels) so the
    parse_model `else: c2 = ch[f]` branch stays correct -> NO monkeypatching
  * register_fccg() injects classes into ultralytics namespaces (YAML parse +
    checkpoint reload), mirroring register_cbam() in the thesis script
  * CBAM + ChannelAttention + SpatialAttention are EMBEDDED here verbatim from
    the thesis script (yolo11m_cbam_p2head_thesis.py §2.5): the lineage YAML's
    layer 10 is CBAM, and ultralytics 8.4.x resolves YAML names via the tasks
    globals -> without registering CBAM the build dies with KeyError 'CBAM'
    (hit on PC-4, 2026-07-12). One import of THIS module now registers
    everything a FCCG checkpoint needs.
  * selftests runnable on CPU torch WITHOUT ultralytics (laptop-safe);
    --check-load additionally builds the YAML (needs modern ultralytics -> PC)

Where the modules sit (emit YAML with --emit-yaml):
  head seam after each top-down Concat:  [ctx_half (upsampled semantic stream) |
  ev_half (backbone lateral)] -> FCCGFuse(split_c): ev' = ev + gamma*HFE(ev)*g,
  g = sigmoid(PW(act(PW(LK(ctx))))), LK = DW k7-dil3 + DW k11 (decomposed large
  kernel). Context half passes through untouched. FFLUp replaces nn.Upsample on
  the top-down path (content-adaptive 3x3 low-pass on nearest-up = FreqFusion-
  lite ALPF; the AHPF role is covered by HFE on the lateral).

Run:
  python 10_fccg_modules.py --selftest            # CPU ok, no ultralytics needed
  python 10_fccg_modules.py --emit-yaml           # writes yolo11m_fccg_p2.yaml next to this file
  python 10_fccg_modules.py --check-load          # PC only: YAML build + fwd + save/load roundtrip
"""

from __future__ import annotations

import argparse
import io
import pickle
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# config (single source of truth; YAML is generated from this file)
# --------------------------------------------------------------------------
GATE_REDUCTION = 8      # PW squeeze in the context gate
BANK_K = 3              # DW evidence bank kernel
LK1_K, LK1_D = 7, 3     # decomposed large kernel, stage 1 (ERF 19)
LK2_K = 11              # stage 2
YAML_NAME = "yolo11m_fccg_p2.yaml"


# --------------------------------------------------------------------------
# submodules (top-level classes: pickle-safe)
# --------------------------------------------------------------------------
class HFEvidence(nn.Module):
    """High-frequency evidence: (x - avgpool3(x)) -> learnable DW bank -> PW.
    Learnable, spatial-domain (deliberately NOT a fixed wavelet/Gabor basis)."""

    def __init__(self, c: int):
        super().__init__()
        self.dw = nn.Conv2d(c, c, BANK_K, 1, BANK_K // 2, groups=c, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.act = nn.SiLU()
        self.pw = nn.Conv2d(c, c, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hp = x - F.avg_pool2d(x, 3, 1, 1)
        return self.bn2(self.pw(self.act(self.bn1(self.dw(hp)))))


class LKContextGate(nn.Module):
    """Plausibility gate from the semantic (top-down) half: decomposed
    large-kernel DW context -> PW squeeze -> 1-ch sigmoid gate in [0,1].
    Gate bias init 0 => g ~= 0.5 at init (non-saturated, selftested)."""

    def __init__(self, c: int):
        super().__init__()
        p1 = (LK1_K // 2) * LK1_D
        self.lk1 = nn.Conv2d(c, c, LK1_K, 1, p1, dilation=LK1_D, groups=c, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.lk2 = nn.Conv2d(c, c, LK2_K, 1, LK2_K // 2, groups=c, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.act = nn.SiLU()
        cr = max(8, c // GATE_REDUCTION)
        self.pw1 = nn.Conv2d(c, cr, 1, 1, 0, bias=True)
        self.pw2 = nn.Conv2d(cr, 1, 1, 1, 0, bias=True)
        nn.init.zeros_(self.pw2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.act(self.bn1(self.lk1(x)))
        c = self.act(self.bn2(self.lk2(c)))
        return torch.sigmoid(self.pw2(self.act(self.pw1(c))))


class FCCGFuse(nn.Module):
    """Context-gated evidence fusion at a Concat seam.

    Input  : cat([context_half (split_c ch, upsampled top-down semantic stream),
                  evidence_half (backbone lateral)], dim=1)
    Output : same shape; evidence half becomes ev + gamma * HFE(ev) * g(ctx).
    split_c comes from the YAML arg (static per site). Lazy-built on first
    forward (CBAM pattern) so parse_model needs no channel surgery.
    """

    def __init__(self, split_c: int):
        super().__init__()
        self.split_c = int(split_c)
        self.built = False
        self.hfe: nn.Module | None = None
        self.gate: nn.Module | None = None
        self.gamma: nn.Parameter | None = None
        # selftest/telemetry (not saved, not synced)
        self.last_gate_mean: float = -1.0
        self.force_zero_gate: bool = False  # selftest hook

    def _build(self, c_total: int, ref: torch.Tensor) -> None:
        if c_total <= self.split_c:
            raise ValueError(
                f"FCCGFuse(split_c={self.split_c}) got only {c_total} input channels; "
                "check the YAML seam wiring (context channels must equal split_c and "
                "at least 1 evidence channel must follow)."
            )
        c_ev = c_total - self.split_c
        self.hfe = HFEvidence(c_ev)
        self.gate = LKContextGate(self.split_c)
        self.gamma = nn.Parameter(torch.ones(1, c_ev, 1, 1))
        self.to(device=ref.device, dtype=ref.dtype)
        self.built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.built:
            self._build(x.shape[1], x)
        ctx = x[:, : self.split_c]
        ev = x[:, self.split_c:]
        g = self.gate(ctx)
        if self.force_zero_gate:
            g = torch.zeros_like(g)
        self.last_gate_mean = float(g.detach().float().mean())
        ev_out = ev + self.gamma * self.hfe(ev) * g
        return torch.cat((ctx, ev_out), 1)


class FFLUp(nn.Module):
    """FreqFusion-lite adaptive low-pass x2 upsampler (ALPF, offset-free).
    Predicts a per-pixel 3x3 normalized kernel at low res, applies it to the
    nearest-upsampled map via 9 shifted accumulations (no unfold buffers).
    Shape: (B,C,H,W) -> (B,C,2H,2W). Lazy-built; channels preserved."""

    def __init__(self):
        super().__init__()
        self.built = False
        self.kpred: nn.Module | None = None

    def _build(self, c: int, ref: torch.Tensor) -> None:
        self.kpred = nn.Conv2d(c, 9, 1, 1, 0, bias=True)
        nn.init.zeros_(self.kpred.bias)
        nn.init.zeros_(self.kpred.weight)  # exactly uniform kernel at init (= box
        # blur; trainable — the 9 logits receive distinct grads via their shifts)
        self.to(device=ref.device, dtype=ref.dtype)
        self.built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.built:
            self._build(x.shape[1], x)
        k = torch.softmax(self.kpred(x), dim=1)                    # B,9,H,W
        up = F.interpolate(x, scale_factor=2.0, mode="nearest")    # B,C,2H,2W
        k = F.interpolate(k, scale_factor=2.0, mode="nearest")     # B,9,2H,2W
        pad = F.pad(up, (1, 1, 1, 1), mode="replicate")
        H2, W2 = up.shape[-2:]
        out = torch.zeros_like(up)
        i = 0
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                out = out + k[:, i : i + 1] * pad[:, :, dy : dy + H2, dx : dx + W2]
                i += 1
        return out


class FCCGActiveCheck:
    """Training-time guard (mirrors the NWD abort-if-inactive callback): call
    .verify(model) after a forward; raises if no FCCGFuse ran or gates are
    saturated/degenerate. Use in the smoke run on PC-4."""

    @staticmethod
    def verify(model: nn.Module) -> dict:
        stats = {}
        for name, m in model.named_modules():
            if isinstance(m, FCCGFuse):
                stats[name] = m.last_gate_mean
        if not stats:
            raise RuntimeError("[fccg-verify] NO FCCGFuse modules found in model!")
        for name, gm in stats.items():
            if gm < 0:
                raise RuntimeError(f"[fccg-verify] {name} never ran a forward pass!")
            if not (0.02 < gm < 0.98):
                raise RuntimeError(f"[fccg-verify] {name} gate saturated (mean={gm:.4f})")
        return stats


# --------------------------------------------------------------------------
# CBAM lineage classes — verbatim semantics from the thesis script
# (yolo11m_cbam_p2head_thesis.py §2.5; lazy channel init, YAML args [16, 7])
# --------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    """Channel Attention Module (shared-MLP over avg- and max-pooled descriptors)."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced_channels = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced_channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_channels, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module (conv over channel-avg + channel-max maps)."""

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(concat))


class CBAM(nn.Module):
    """CBAM with LAZY channel init — auto-detects input channels on the first
    forward pass, bypassing Ultralytics parse_model channel bookkeeping.
    Accepts the YAML arg form [reduction, kernel_size] (e.g. [16, 7]) and also
    the (c1, c2, ...) forms Ultralytics may pass (channels are ignored and
    detected from the input tensor)."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.reduction = 16
        self.kernel_size = 7
        if len(args) == 1:
            if isinstance(args[0], int) and args[0] <= 32:
                self.reduction = args[0]
        elif len(args) == 2:
            if isinstance(args[0], int) and args[0] <= 32:
                self.reduction = args[0]
                self.kernel_size = args[1] if isinstance(args[1], int) else 7
        elif len(args) >= 4:
            self.reduction = args[2] if isinstance(args[2], int) else 16
            self.kernel_size = args[3] if isinstance(args[3], int) else 7
        self.reduction = kwargs.get("reduction", self.reduction)
        self.kernel_size = kwargs.get("kernel_size", self.kernel_size)
        if self.kernel_size not in (3, 7):
            self.kernel_size = 7
        self._initialized = False
        self.channel_attention = None
        self.spatial_attention = None
        self._channels = None

    def _lazy_init(self, channels, device, dtype):
        self._channels = channels
        self.channel_attention = ChannelAttention(channels, self.reduction).to(
            device=device, dtype=dtype)
        self.spatial_attention = SpatialAttention(self.kernel_size).to(
            device=device, dtype=dtype)
        self._initialized = True

    def forward(self, x):
        if not self._initialized:
            self._lazy_init(x.size(1), x.device, x.dtype)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# --------------------------------------------------------------------------
# registration (mirrors register_cbam in the thesis training script)
# --------------------------------------------------------------------------
_FCCG_CLASSES = {"FCCGFuse": FCCGFuse, "FFLUp": FFLUp,
                 "HFEvidence": HFEvidence, "LKContextGate": LKContextGate,
                 "CBAM": CBAM, "ChannelAttention": ChannelAttention,
                 "SpatialAttention": SpatialAttention}


def register_fccg() -> bool:
    """Inject FCCG classes into ultralytics namespaces so YAML parse and
    checkpoint (de)serialization find them. Safe no-op without ultralytics."""
    try:
        import ultralytics.nn.tasks as _tasks
        import ultralytics.nn.modules as _mods
    except Exception:
        return False
    for ns in (_tasks, _mods):
        for k, v in _FCCG_CLASSES.items():
            setattr(ns, k, v)
    import builtins  # some ultralytics versions resolve YAML names via eval()
    for k, v in _FCCG_CLASSES.items():
        if not hasattr(builtins, k):
            setattr(builtins, k, v)
    return True


# --------------------------------------------------------------------------
# the model YAML (single source of truth — emitted with --emit-yaml)
# Channel walk (width=1.0, max=512):  L15 cat = 512(up)+512(L4) -> split 512;
# L19 cat = 256(up)+256(L2) -> split 256. Detect taps [21,24,27,30] carry
# [128,256,512,512] — identical to the CBAM+P2 base, so heads stay comparable.
# --------------------------------------------------------------------------
FCCG_YAML = """# YOLO11m + CBAM + P2 head + FCCG (context-gated evidence) — lap-3 S0
# Base: Last Month/.../yolov11m_cbam_p2head.yaml (thesis lineage, layer-compat)
# New: FFLUp replaces nn.Upsample (FreqFusion-lite ALPF); FCCGFuse at the
#      P3 and P2 Concat seams (split_c = channels of the upsampled ctx half).
nc: 1
scales:
  m: [0.50, 1.00, 512]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]          # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]         # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]  # 2
  - [-1, 1, Conv, [256, 3, 2]]         # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]  # 4
  - [-1, 1, Conv, [512, 3, 2]]         # 5-P4/16
  - [-1, 2, C3k2, [512, True]]         # 6
  - [-1, 1, Conv, [1024, 3, 2]]        # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]        # 8
  - [-1, 1, SPPF, [1024, 5]]           # 9
  - [-1, 1, CBAM, [16, 7]]             # 10 (thesis lineage: CBAM replaces C2PSA)

head:
  - [-1, 1, FFLUp, []]                 # 11 adaptive low-pass x2 (was nn.Upsample)
  - [[-1, 6], 1, Concat, [1]]          # 12 cat P4
  - [-1, 2, C3k2, [512, False]]        # 13
  - [-1, 1, FFLUp, []]                 # 14
  - [[-1, 4], 1, Concat, [1]]          # 15 cat P3   (512 ctx + 512 lateral)
  - [-1, 1, FCCGFuse, [512]]           # 16 ** context-gated evidence @P3 seam
  - [-1, 2, C3k2, [256, False]]        # 17 (P3/8)
  - [-1, 1, FFLUp, []]                 # 18
  - [[-1, 2], 1, Concat, [1]]          # 19 cat P2   (256 ctx + 256 lateral)
  - [-1, 1, FCCGFuse, [256]]           # 20 ** context-gated evidence @P2 seam
  - [-1, 2, C3k2, [128, False]]        # 21 (P2/4-xsmall)
  - [-1, 1, Conv, [128, 3, 2]]         # 22
  - [[-1, 17], 1, Concat, [1]]         # 23
  - [-1, 2, C3k2, [256, False]]        # 24 (P3/8)
  - [-1, 1, Conv, [256, 3, 2]]         # 25
  - [[-1, 13], 1, Concat, [1]]         # 26
  - [-1, 2, C3k2, [512, False]]        # 27 (P4/16)
  - [-1, 1, Conv, [512, 3, 2]]         # 28
  - [[-1, 10], 1, Concat, [1]]         # 29 cat P5
  - [-1, 2, C3k2, [1024, True]]        # 30 (P5/32)
  - [[21, 24, 27, 30], 1, Detect, [1]] # 31 Detect(P2,P3,P4,P5)
"""


# --------------------------------------------------------------------------
# selftests (CPU, no ultralytics)
# --------------------------------------------------------------------------
def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def selftest() -> int:
    torch.manual_seed(0)
    fails = 0

    def check(name, cond, extra=""):
        nonlocal fails
        tag = "PASS" if cond else "FAIL"
        if not cond:
            fails += 1
        print(f"[{tag}] {name} {extra}")

    # t1 shapes (both seam configs + odd sizes)
    f3 = FCCGFuse(512)
    x = torch.randn(2, 1024, 80, 80)
    y = f3(x)
    check("t1a FCCGFuse@P3 shape", y.shape == x.shape, str(tuple(y.shape)))
    f2 = FCCGFuse(256)
    x2 = torch.randn(2, 512, 160, 160)
    check("t1b FCCGFuse@P2 shape", f2(x2).shape == x2.shape)
    xo = torch.randn(1, 1024, 77, 53)
    check("t1c FCCGFuse odd HxW", f3(xo).shape == xo.shape)
    up = FFLUp()
    xu = torch.randn(2, 512, 40, 40)
    check("t1d FFLUp x2 shape", up(xu).shape == (2, 512, 80, 80))
    xo2 = torch.randn(1, 256, 37, 41)
    up2 = FFLUp()
    check("t1e FFLUp odd shape", up2(xo2).shape == (1, 256, 74, 82))

    # t2 gradients reach every parameter
    f3.zero_grad(set_to_none=True)
    f3(torch.randn(2, 1024, 40, 40)).sum().backward()
    missing = [n for n, p in f3.named_parameters() if p.grad is None]
    check("t2a FCCGFuse grads all params", not missing, f"missing={missing[:4]}")
    up.zero_grad(set_to_none=True)
    up(xu).sum().backward()
    missing = [n for n, p in up.named_parameters() if p.grad is None]
    check("t2b FFLUp grads all params", not missing)

    # t3 pickle + state_dict roundtrip AFTER lazy build (checkpoint safety)
    blob = pickle.dumps(f3)
    f3b = pickle.loads(blob)
    check("t3a pickle roundtrip", isinstance(f3b, FCCGFuse) and f3b.built)
    buf = io.BytesIO()
    torch.save(f3.state_dict(), buf)
    buf.seek(0)
    f3c = FCCGFuse(512)
    f3c(torch.randn(1, 1024, 8, 8))  # build first
    f3c.load_state_dict(torch.load(buf, weights_only=True))
    check("t3b state_dict roundtrip", True)

    # t4 gate healthy at init (not saturated)
    gm = f3.last_gate_mean
    check("t4 gate mean in (0.35,0.65) at init", 0.35 < gm < 0.65, f"mean={gm:.4f}")

    # t5 zero-gate => evidence half untouched (gating is real)
    f2.force_zero_gate = True
    xin = torch.randn(1, 512, 32, 32)
    yout = f2(xin)
    same = torch.allclose(yout[:, 256:], xin[:, 256:], atol=0, rtol=0)
    ctx_same = torch.allclose(yout[:, :256], xin[:, :256])
    f2.force_zero_gate = False
    check("t5 zero-gate identity on evidence half", same and ctx_same)

    # t6 active-check guard works both ways
    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = f3
    try:
        FCCGActiveCheck.verify(_Tiny())
        ok6 = True
    except RuntimeError:
        ok6 = False
    check("t6a active-check passes on used module", ok6)
    try:
        FCCGActiveCheck.verify(nn.Sequential(nn.Conv2d(3, 3, 1)))
        ok6b = False
    except RuntimeError:
        ok6b = True
    check("t6b active-check raises when absent", ok6b)

    # t7 FFLUp ~box-blur at init (kernel ~uniform): compare vs avg-pool of nearest-up
    with torch.no_grad():
        upx = F.interpolate(xu, scale_factor=2.0, mode="nearest")
        ref = F.avg_pool2d(F.pad(upx, (1, 1, 1, 1), mode="replicate"), 3, 1, 0)
        got = up(xu)
        err = (got - ref).abs().max().item()
    check("t7 FFLUp == box blur at init (zero-init)", err < 1e-4, f"max_err={err:.2e}")

    # t8 wrong wiring raises a clear error
    try:
        FCCGFuse(512)(torch.randn(1, 256, 8, 8))
        ok8 = False
    except ValueError:
        ok8 = True
    check("t8 split>input raises ValueError", ok8)

    # t10 CBAM lineage class embedded here (PC-4 KeyError 'CBAM' regression guard)
    cb = CBAM(16, 7)
    xcb = torch.randn(1, 512, 16, 16)
    ycb = cb(xcb)
    ok10 = ycb.shape == xcb.shape and cb._initialized and cb._channels == 512
    cb2 = pickle.loads(pickle.dumps(cb))
    ok10 = ok10 and isinstance(cb2, CBAM) and cb2(xcb).shape == xcb.shape
    check("t10 CBAM lazy build + pickle", ok10)

    # t9 parameter budget report
    p3, p2 = _count_params(f3), _count_params(f2)
    pu = _count_params(up) + _count_params(up2)
    total = p3 + p2 + pu
    print(f"[info] params: FCCG@P3={p3/1e6:.3f}M FCCG@P2={p2/1e6:.3f}M FFLUp(x2)={pu/1e3:.1f}K "
          f"| new-total~{total/1e6:.3f}M (budget: base 19.6M + <=3M)")
    check("t9 param budget", total < 3_000_000, f"{total/1e6:.3f}M")

    print(("SELFTEST OK — all green" if fails == 0 else f"SELFTEST FAILED ({fails})"))
    return fails


def emit_yaml() -> Path:
    out = Path(__file__).with_name(YAML_NAME)
    out.write_text(FCCG_YAML, encoding="utf-8")
    print(f"[emit] wrote {out}")
    return out


def check_load() -> int:
    """PC-side: YAML build + dummy forward + ckpt save/load roundtrip + verify."""
    if not register_fccg():
        print("[check-load] ultralytics not importable here — run on PC-4/PC-2/PC-1.")
        return 1
    from ultralytics import YOLO
    import tempfile
    ypath = emit_yaml()
    model = YOLO(str(ypath))  # stride dummy-forward materializes lazy modules
    nn_model = model.model
    n_params = sum(p.numel() for p in nn_model.parameters())
    print(f"[check-load] built OK — {n_params/1e6:.2f}M params (budget <=22.5M)")
    x = torch.zeros(1, 3, 640, 640)
    nn_model.eval()
    with torch.no_grad():
        nn_model(x)
    stats = FCCGActiveCheck.verify(nn_model)
    print(f"[check-load] FCCG ACTIVE: {stats}")
    with tempfile.TemporaryDirectory() as td:
        ck = Path(td) / "roundtrip.pt"
        torch.save({"model": nn_model}, ck)          # full-module pickle path
        obj = torch.load(ck, weights_only=False)      # requires registration
        obj["model"](x)
        print("[check-load] checkpoint save/load/forward roundtrip OK")
    ok = n_params <= 22_500_000
    print("CHECK-LOAD OK" if ok else "CHECK-LOAD FAIL: param budget exceeded")
    return 0 if ok else 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--emit-yaml", action="store_true")
    ap.add_argument("--check-load", action="store_true")
    a = ap.parse_args()
    rc = 0
    if a.selftest or not (a.emit_yaml or a.check_load):
        rc = selftest()
    if a.emit_yaml:
        emit_yaml()
    if a.check_load:
        rc = max(rc, check_load())
    sys.exit(rc)

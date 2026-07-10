"""
cbam_modules.py -- CBAM classes + registration for loading cbam_p2head*.pt.

Classes are VERBATIM from the training-time snapshot
(code_snapshots/yolo11m_cbam_p2head_thesis.py, the 20260602_063759 run).
Do not edit the class bodies: the checkpoints were pickled against these
exact definitions.

Usage (must happen BEFORE any YOLO()/SAHI model load):
    import cbam_modules
    cbam_modules.register()
"""
import torch
import torch.nn as nn


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
    """CBAM with LAZY channel init -- auto-detects input channels on the first
    forward pass, bypassing Ultralytics parse_model channel bookkeeping."""
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


_CLASSES = {"CBAM": CBAM, "ChannelAttention": ChannelAttention,
            "SpatialAttention": SpatialAttention}


# Module names the checkpoints may reference for these classes, depending on
# how each training script was executed when its checkpoint was pickled:
#   __main__                       -> cbam_p2head.pt (script run directly)
#   joint_c2a_sard_train           -> cbam_p2head_finetune_enriched.pt
#   yolo11m_cbam_p2head_thesis     -> insurance (script imported by name)
_STUB_MODULES = ("joint_c2a_sard_train", "yolo11m_cbam_p2head_thesis",
                 "yolo11m_thesis", "yolo11m_cbam_thesis")


def register():
    """Register CBAM classes everywhere checkpoint deserialization may look.
    Mirrors register_cbam() in the training script (lines 434-448), plus
    __main__ injection, plus stub modules for checkpoints pickled under the
    training script's own module name. Idempotent."""
    import sys
    import types
    import ultralytics.nn.modules as _modules
    import ultralytics.nn.tasks as _tasks
    for ns in (_modules, _tasks):
        for name, cls in _CLASSES.items():
            setattr(ns, name, cls)
    import __main__ as _main
    for name, cls in _CLASSES.items():
        if not hasattr(_main, name):
            setattr(_main, name, cls)
    for modname in _STUB_MODULES:
        mod = sys.modules.get(modname)
        if mod is None:
            mod = types.ModuleType(modname)
            mod.__doc__ = "stub for checkpoint unpickling (cbam_modules.register)"
            sys.modules[modname] = mod
        for name, cls in _CLASSES.items():
            if not hasattr(mod, name):
                setattr(mod, name, cls)
    try:  # torch >= 2.6 safe-unpickling allowlist (no-op on older torch)
        import torch.serialization as _ts
        _ts.add_safe_globals(list(_CLASSES.values()))
    except Exception:
        pass

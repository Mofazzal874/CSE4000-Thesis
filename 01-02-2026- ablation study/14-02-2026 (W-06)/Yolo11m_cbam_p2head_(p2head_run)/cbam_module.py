
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
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
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
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
    """
    CBAM: Convolutional Block Attention Module with LAZY INITIALIZATION
    Auto-detects input channels from tensor at runtime.
    Handles any arg format from Ultralytics parse_model.
    """
    def __init__(self, *args, **kwargs):
        super(CBAM, self).__init__()
        self.reduction = 16
        self.kernel_size = 7

        if len(args) == 0:
            pass
        elif len(args) == 1:
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
        self.channel_attention = ChannelAttention(channels, self.reduction)
        self.spatial_attention = SpatialAttention(self.kernel_size)
        self.channel_attention = self.channel_attention.to(device=device, dtype=dtype)
        self.spatial_attention = self.spatial_attention.to(device=device, dtype=dtype)
        self._initialized = True

    def forward(self, x):
        if not self._initialized:
            self._lazy_init(x.size(1), x.device, x.dtype)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

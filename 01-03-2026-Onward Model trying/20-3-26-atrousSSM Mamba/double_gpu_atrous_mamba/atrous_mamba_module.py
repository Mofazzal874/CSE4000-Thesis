
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _get_window_size(channels: int) -> int:
    if channels >= 512: return 4
    if channels >= 256: return 6
    return 8


class _SelectiveScan1D(nn.Module):
    """One-direction selective scan. u: (B, L, D) -> (B, L, D)"""
    def __init__(self, d_model: int, d_state: int = 4, dt_rank_ratio: int = 16):
        super().__init__()
        D, N = d_model, d_state
        dt_rank = max(D // dt_rank_ratio, 1)
        self.D, self.N, self.dt_rank = D, N, dt_rank
        self.conv1d = nn.Conv1d(D, D, kernel_size=4, padding=3, groups=D, bias=True)
        self.x_proj  = nn.Linear(D, dt_rank + 2 * N, bias=False)
        self.dt_proj = nn.Linear(dt_rank, D, bias=True)
        A = torch.arange(1, N + 1, dtype=torch.float32).unsqueeze(0).repeat(D, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D_skip = nn.Parameter(torch.ones(D))
        dt_init = torch.exp(
            torch.rand(D) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, u):
        B_win, L, D = u.shape
        in_dtype = u.dtype
        u_conv = self.conv1d(u.transpose(1, 2))[:, :, :L].transpose(1, 2)
        u_act = F.silu(u_conv)
        xBC_dt = self.x_proj(u_act)
        dt_raw, B_param, C_param = xBC_dt.split([self.dt_rank, self.N, self.N], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw)).float()
        B_param, C_param, u_f = B_param.float(), C_param.float(), u_act.float()
        A = -torch.exp(self.A_log.float())
        _dtA = torch.einsum("bld,dn->bldn", dt, A)
        deltaA = torch.exp(_dtA.clamp(-20.0, 0.0))  # CRITICAL: prevent exp overflow/NaN
        deltaB_u = torch.einsum("bld,bln,bld->bldn", dt, B_param, u_f)
        x = torch.zeros(B_win, D, self.N, device=u.device, dtype=torch.float32)
        ys = []
        for i in range(L):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            x = x.clamp(-1e4, 1e4)  # prevent runaway state accumulation
            ys.append((x * C_param[:, i, :].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1).to(in_dtype)
        return y + u_act * self.D_skip.to(in_dtype)


class AtrousSSM(nn.Module):
    """Multi-Scale Dilated Window State Space Model."""
    def __init__(self, d_model, d_state=4, window_size=8, dilations=None):
        super().__init__()
        self.ws = window_size
        self.dilations = dilations or [1, 2, 4]
        D = d_model
        n_br = len(self.dilations)
        self.branches = nn.ModuleList()
        for _ in self.dilations:
            branch = nn.ModuleDict({
                'norm':     nn.LayerNorm(D),
                'in_proj':  nn.Linear(D, D * 2, bias=False),
                'scan_fwd': _SelectiveScan1D(D, d_state),
                'scan_bwd': _SelectiveScan1D(D, d_state),
                'out_proj': nn.Linear(D, D, bias=False),
            })
            nn.init.normal_(branch['out_proj'].weight, std=0.02)
            nn.init.xavier_uniform_(branch['in_proj'].weight)
            self.branches.append(branch)
        self.fusion_gate = nn.Sequential(nn.Conv2d(D * n_br, D, 1, bias=False), nn.Sigmoid())
        self.fusion_proj = nn.Conv2d(D * n_br, D, 1, bias=False)
        self.out_norm = nn.LayerNorm(D)
        nn.init.normal_(self.fusion_gate[0].weight, std=0.01)
        nn.init.normal_(self.fusion_proj.weight, std=0.02)

    def _dilated_partition(self, x, dilation):
        B, C, H, W = x.shape
        ws, region = self.ws, self.ws * dilation
        ph, pw = (region - H % region) % region, (region - W % region) % region
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        _, _, Hp, Wp = x.shape
        nH, nW = Hp // region, Wp // region
        x = x.reshape(B, C, nH, region, nW, region).permute(0, 2, 4, 1, 3, 5)
        x = x[:, :, :, :, ::dilation, ::dilation].contiguous()
        return x.reshape(B * nH * nW, C, ws * ws).transpose(1, 2), (B, C, H, W, Hp, Wp, nH, nW)

    def _dilated_reverse(self, tokens, meta, dilation):
        B, C, H, W, Hp, Wp, nH, nW = meta
        ws, region = self.ws, self.ws * dilation
        windows = tokens.transpose(1, 2).reshape(B * nH * nW, C, ws, ws)
        if dilation > 1:
            windows = F.interpolate(windows, size=(region, region), mode='bilinear', align_corners=False)
        x = windows.reshape(B, nH, nW, C, region, region).permute(0, 3, 1, 4, 2, 5)
        return x.reshape(B, C, Hp, Wp)[:, :, :H, :W].contiguous()

    def forward(self, x):
        branch_outputs = []
        for branch, dilation in zip(self.branches, self.dilations):
            tokens, meta = self._dilated_partition(x, dilation)
            residual = tokens
            tokens_n = branch['norm'](tokens)
            xz = branch['in_proj'](tokens_n)
            x_in, z = xz.chunk(2, dim=-1)
            y_fwd = branch['scan_fwd'](x_in)
            y_bwd = branch['scan_bwd'](x_in.flip(1)).flip(1)
            y = (y_fwd + y_bwd) * F.silu(z)
            y = branch['out_proj'](y) + residual
            branch_outputs.append(self._dilated_reverse(y, meta, dilation))
        concat = torch.cat(branch_outputs, dim=1)
        gate = self.fusion_gate(concat)
        fused = self.fusion_proj(concat)
        out = fused * gate + x * (1 - gate)
        out = self.out_norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out


class _MambaBottleneck(nn.Module):
    def __init__(self, c, shortcut, d_state, window_size, dilations=None):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.cv1 = Conv(c, c, 3, 1)
        self.ssm = AtrousSSM(c, d_state=d_state, window_size=window_size, dilations=dilations)
        self.cv2 = Conv(c, c, 3, 1)
        self.add = shortcut

    def forward(self, x):
        y = self.cv2(self.ssm(self.cv1(x)))
        return x + y if self.add else y


class C3K2Mamba(nn.Module):
    """C2f-style block with AtrousSSM bottleneck. Drop-in for C3k2 in YOLO11m neck.
    YAML args: [c2, shortcut, g, e, d_state, dilations]
    After parse_model inserts n: C3K2Mamba(c1, c2, n, shortcut, g, e, d_state, dilations)
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5,
                 d_state=4, dilations=None):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.c = int(c2 * e)
        ws = _get_window_size(self.c)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            _MambaBottleneck(self.c, shortcut and c1 == c2,
                             d_state, ws, dilations=dilations or [1, 2, 4])
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

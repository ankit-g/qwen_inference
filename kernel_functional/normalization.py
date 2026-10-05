import torch
from torch.nn import functional as F


def rms_norm(x, weight, eps):
    # x: (...,D); accumulate squares in FP32, preserving all axes.
    y = x.float()
    # Reduce D: mean(x^2) -> (...,1), broadcast inverse RMS over D.
    # Elementwise x / sqrt(mean(x^2)+eps); no token/head mixing.
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    # Scale (D,) broadcasts over leading axes; effective weight is 1+w.
    return (y * (1 + weight.float())).to(x.dtype)


def gated_rms_norm(x, gate, weight, eps):
    y = x.float()
    # Reduce D: mean(x^2) -> (...,1), broadcast inverse RMS over D.
    # Elementwise x / sqrt(mean(x^2)+eps); no token/head mixing.
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    # Direct learned scale (Dv,), not 1+w; output stays (B,T,Hv,Dv).
    y = weight * y.to(x.dtype)
    # SiLU(gate)=gate*sigmoid(gate); featurewise gate has the same shape.
    return (y * F.silu(gate.float())).to(x.dtype)

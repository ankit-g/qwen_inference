from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AttentionCache:
    keys: torch.Tensor  # (B,Hkv,N,D), already rotated
    values: torch.Tensor  # (B,Hkv,N,D), without rotary transformation


@dataclass(frozen=True)
class DeltaNetCache:
    convolution: torch.Tensor  # (B,F,K-1), raw QKV tail
    recurrent_state: torch.Tensor  # (B,Hv,Dk,Dv), FP32


@dataclass(frozen=True)
class HybridCache:
    layers: tuple
    position: int
    layer_types: tuple
    owner: object = None
    batch_size: int | None = None
    device: torch.device | None = None
    dtype: torch.dtype | None = None


def create_cache(cfg):
    # One slot per layer; None means no processed tokens yet.
    # Attention grows (B,Hkv,N,D); DeltaNet keeps fixed (B,Hv,Dk,Dv).
    return HybridCache(
        (None,) * cfg.num_hidden_layers, 0, tuple(cfg.layer_types)
    )

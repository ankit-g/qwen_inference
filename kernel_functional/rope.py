from dataclasses import dataclass

import torch
import pdb

# Adapted from the existing Qwen/Hugging Face-derived implementation.
# Apache-2.0; see ../LICENSE-qwen35-transformers.txt.


def apply_partial_rope(x, cos, sin):
    # Each coordinate pair uses [[cos,-sin],[sin,cos]] @ [u,v].
    # Broadcast (1,1,T,R/2) angles over B,H; no token mixing.
    dim = cos.shape[-1] * 2
    half = dim // 2
    # split prefix into two halves and interleave matching coordinates.
    # (B,H,T,R/2,2) -> (B,H,T,R); toy [a,b,c,d] -> [a,c,b,d].
    paired = torch.stack((x[..., :half], x[..., half:dim]), -1).flatten(-2)
    # Each pair component has shape (B,H,T,R/2); tail coordinates are excluded.
    q_even, q_odd = paired[..., 0::2], paired[..., 1::2]
    rotated = torch.empty_like(paired)
    # 2D rotation at angle phi: u_new = u*cos(phi) - v*sin(phi).
    rotated[..., 0::2] = q_even * cos - q_odd * sin
    # Second coordinate: v_new = u*sin(phi) + v*cos(phi).
    rotated[..., 1::2] = q_even * sin + q_odd * cos
    # Rotated prefix (B,H,T,R) + untouched tail (B,H,T,D-R) -> (B,H,T,D).
    # The same permutation on Q and K preserves their dot products.
    return torch.cat((rotated, x[..., dim:]), dim=-1)


@dataclass(frozen=True)
class RoPETables:
    cos: torch.Tensor  # (max_positions,R/2), FP32
    sin: torch.Tensor
    signature: tuple  # rotary width, frequency base, maximum positions


@torch.inference_mode(False)
@torch.no_grad()
def build_rope_tables(cfg, device):
    from .error_handling import validate_rope_positions

    validate_rope_positions(
        0, cfg.max_position_embeddings, cfg.max_position_embeddings
    )
    # head_dim D is the feature width of each query/key head.
    # 4B: D=256; Q is (B,16,T,256), K is (B,4,T,256).
    # With partial_rotary_factor=0.25, R=64 features rotate in 32 pairs.
    # The remaining 192 features per head stay unchanged.
    # Selected cos/sin: (1,1,T,32), shared across batches and heads.
    dim = int(
        cfg.head_dim * cfg.rope_parameters.get("partial_rotary_factor", 1.0)
    )
    base = cfg.rope_parameters["rope_theta"]
    pairs = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    # Frequency theta_j = base**(-2*j/R), one value per rotary pair.
    frequencies = 1.0 / (base ** (pairs / dim))
    positions = torch.arange(
        cfg.max_position_embeddings, dtype=torch.float32, device=device
    )
    # (max_positions,1) * (1,R/2) -> (max_positions,R/2).
    angles = positions.unsqueeze(1) * frequencies.unsqueeze(0)
    return RoPETables(
        angles.cos(), angles.sin(), (dim, base, cfg.max_position_embeddings)
    )


def rope_angles(cfg, tables, x, offset, length):
    from .error_handling import validate_rope_positions

    end = offset + length
    validate_rope_positions(offset, end, cfg.max_position_embeddings)
    dim = int(
        cfg.head_dim * cfg.rope_parameters.get("partial_rotary_factor", 1.0)
    )
    signature = (
        dim,
        cfg.rope_parameters["rope_theta"],
        cfg.max_position_embeddings,
    )
    if (
        tables is None
        or tables.signature != signature
        or tables.cos.device != x.device
        or tables.cos.dtype != torch.float32
    ):
        tables = build_rope_tables(cfg, x.device)
    # Slice (T,R/2), cast if needed, then add broadcast axes (1,1,T,R/2).
    # Slicing/unsqueeze share storage; changing dtype allocates a cast copy.
    cos = tables.cos[offset:end].to(x.dtype).unsqueeze(0).unsqueeze(0)
    sin = tables.sin[offset:end].to(x.dtype).unsqueeze(0).unsqueeze(0)
    return (cos, sin), tables


if __name__ == "__main__":
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        head_dim=256,
        max_position_embeddings=262144,
        rope_parameters={
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
        },
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0)
    queries = torch.randn(1, 16, 5, 256, device=device)
    with torch.no_grad():
        # No registered buffers: tables are ordinary values passed between
        # calls.
        (cos, sin), tables = rope_angles(cfg, None, queries, offset=0, length=5)
        rotated = apply_partial_rope(queries, cos, sin)
        print("Device:", device)
        print("Prefill Q:", tuple(queries.shape))
        print("cos and sin:", tuple(cos.shape), tuple(sin.shape))
        print("Rotated Q:", tuple(rotated.shape))
        print("Position 0 cos:", cos[0, 0, 0].tolist())
        torch.testing.assert_close(rotated[..., 64:], queries[..., 64:])

        # Five tokens already processed: the next token uses absolute position
        # 5.
        next_query = torch.randn(1, 16, 1, 256, device=device)
        angles, next_tables = rope_angles(
            cfg, tables, next_query, offset=5, length=1
        )
        next_rotated = apply_partial_rope(next_query, *angles)
        assert next_tables is tables  # Full-context tables are reused.
        print("Decode cos:", tuple(angles[0].shape), "at absolute position 5")
        print("Decode Q:", tuple(next_rotated.shape))
        print("Tables reused; unrotated tail preserved.")

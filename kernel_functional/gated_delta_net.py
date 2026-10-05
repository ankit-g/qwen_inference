import math

import torch
from torch.nn import functional as F

from .cache import DeltaNetCache
from .error_handling import validate_linear_backend
from .normalization import gated_rms_norm
from .qwen35_chunked import gated_delta_chunked

# Adapted from the existing Qwen/Hugging Face-derived implementation.
# Apache-2.0; see ../LICENSE-qwen35-transformers.txt.


def gated_delta_scan(query, key, value, log_decay, beta, initial_state=None):
    input_dtype = query.dtype
    q, k, v, decay, strength, state = _prepare_scan(
        query, key, value, log_decay, beta, initial_state
    )
    outputs = []
    for t in range(q.shape[2]):
        # q_t, k_t, v_t are ROW vectors; H = Hv here.
        # 1. Retain: S_ret = alpha_t * S_(t-1), alpha_t = exp(g_t).
        #    alpha: (B,H) -> (B,H,1,1); S: (B,H,Dk,Dv).
        state = state * decay[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)
        # 2. Predict: (B,H,1,Dk) @ (B,H,Dk,Dv) -> (B,H,1,Dv).
        key_row = k[:, :, t].unsqueeze(-2)  # (B,H,1,Dk)
        prediction = (key_row @ state).squeeze(-2)  # (B,H,Dv)
        # 3. Correct: d_t = beta_t * (v_t - v_hat) -> (B,H,Dv).
        # beta: (B,H,1); one strength per head, shared across its Dv features.
        correction = (v[:, :, t] - prediction) * strength[:, :, t].unsqueeze(-1)
        # 4. Write: S_t = S_ret + k_t.T @ d_t.
        #    (B,H,Dk,1) * (B,H,1,Dv) -> rank-one update (B,H,Dk,Dv).
        state = state + k[:, :, t].unsqueeze(-1) * correction.unsqueeze(-2)
        # 5. Read AFTER writing: y_t = q_t @ S_t -> (B,H,Dv).
        #    Current token sees its own write, but no future token.
        query_row = q[:, :, t].unsqueeze(-2)  # (B,H,1,Dk)
        output = (query_row @ state).squeeze(-2)  # (B,H,Dv)
        outputs.append(output)
    # Stack T reads: (B,H,T,Dv) -> (B,T,H,Dv); cast outputs back.
    # The final memory S_T stays FP32 and has no token-length axis.
    return torch.stack(outputs, dim=2).transpose(1, 2).to(input_dtype), state


def _prepare_scan(query, key, value, log_decay, beta, initial_state):
    # Q/K (B,T,H,Dk) -> (B,H,T,Dk); V uses Dv.
    # FP32 reduces accumulated error across repeated memory updates.
    q, k, v = (t.transpose(1, 2).float() for t in (query, key, value))
    # q_hat = q / sqrt(sum_j(q_j^2) + eps); reduce Dk only.
    # This uses SUM, unlike RMSNorm's MEAN; no learned scale here.
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    # k_hat = k / sqrt(sum_j(k_j^2) + eps); same shape (B,H,T,Dk).
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    # q_read = q_hat / sqrt(Dk): additional scaling of reads, not writes.
    q = q / math.sqrt(q.shape[-1])
    # g, beta: (B,T,H) -> (B,H,T), so [:,:,t] selects one token.
    decay, strength = (
        log_decay.transpose(1, 2).float(),
        beta.transpose(1, 2).float(),
    )
    # S_0 = 0 for a fresh sequence; otherwise continue saved S.
    # Rows index key features Dk; columns index value features Dv.
    if initial_state is None:
        state = torch.zeros(
            q.shape[0],
            q.shape[1],
            q.shape[-1],
            v.shape[-1],
            device=q.device,
            dtype=torch.float32,
        )
    else:
        state = initial_state.float()
    return q, k, v, decay, strength, state


def mix_local_tokens(x, weights, cfg, cache):
    # F = 2*Hk*Dk + Hv*Dv: combined Q, K, V width; K: convolution window.
    # Linear: X @ W.T = (B,T,C) @ (C,F) -> (B,T,F).
    # W is stored as (F,C); transpose token/channel axes -> (B,F,T).
    raw = F.linear(x, weights["in_proj_qkv.weight"]).transpose(1, 2)
    # Each token needs K-1 earlier positions plus its own position.
    keep = cfg.linear_conv_kernel_dim - 1
    # History: (B,F,K-1), containing raw projected features, or initial zeros.
    history = (
        raw.new_zeros(raw.shape[0], raw.shape[1], keep)
        if cache is None
        else cache.convolution
    )
    # Join along time: (B,F,K-1) + (B,F,T) -> (B,F,T+K-1).
    window = torch.cat((history, raw), -1)
    # Depthwise convolution: each of F channels has its own K weights.
    # Per channel: (T,K) windows @ (K,1) filter -> (T,1) outputs.
    # No channel mixing: (B,F,T+K-1) -> (B,F,T), using weights (F,1,K).
    # SiLU keeps the shape; transpose restores token order -> (B,T,F).
    mixed = F.silu(
        F.conv1d(window, weights["conv1d.weight"], groups=raw.shape[1])
    ).transpose(1, 2)
    # Save the last K-1 raw positions: (B,F,K-1); K=1 needs no history.
    tail = window[..., -keep:] if keep else window[..., :0]
    # Clone the slice so the cache does not retain the entire window storage.
    return mixed, tail.clone()


def split_heads(mixed, cfg):
    # Split features only: (B,T,2*Hk*Dk+Hv*Dv); no new matrix multiply.
    b, t, _ = mixed.shape
    hk, hv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    dk, dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    q, k, v = mixed.split((hk * dk, hk * dk, hv * dv), dim=-1)
    # Expose heads: Q/K (B,T,Hk,Dk); V (B,T,Hv,Dv).
    q, k = (z.reshape(b, t, hk, dk) for z in (q, k))
    v = v.reshape(b, t, hv, dv)
    if hv // hk > 1:
        # Repeat Q/K to (B,T,Hv,Dk); each V head keeps its own memory.
        q, k = (z.repeat_interleave(hv // hk, dim=2) for z in (q, k))
    return q, k, v


def gated_delta_net(x, weights, cfg, cache=None, backend="chunked"):
    validate_linear_backend(backend)
    mixed, history = mix_local_tokens(x, weights, cfg, cache)
    q, k, v = split_heads(mixed, cfg)
    # X @ Wb.T: (B,T,C) @ (C,Hv) -> (B,T,Hv).
    # Sigmoid gives one write strength per token and value head.
    beta = F.linear(x, weights["in_proj_b.weight"]).sigmoid()
    # g = -exp(A_log) * softplus(a+dt); retained fraction = exp(g).
    # X @ Wa.T: (B,T,C) @ (C,Hv) -> (B,T,Hv).
    # A_log and dt_bias (Hv,) broadcast over B,T; g stays (B,T,Hv).
    g = -weights["A_log"].float().exp() * F.softplus(
        F.linear(x, weights["in_proj_a.weight"]).float() + weights["dt_bias"]
    )
    scan = (
        gated_delta_chunked
        if backend == "chunked" and x.shape[1] > 1
        else gated_delta_scan
    )
    # Q/K (B,T,Hv,Dk), V (B,T,Hv,Dv) -> reads Y (B,T,Hv,Dv).
    # Carry memory S (B,Hv,Dk,Dv) across calls; None starts from zeros.
    y, recurrent_state = scan(
        q, k, v, g, beta, None if cache is None else cache.recurrent_state
    )
    # X @ Wz.T: (B,T,C) @ (C,Hv*Dv) -> (B,T,Hv*Dv).
    # Reshape -> (B,T,Hv,Dv), one output gate per feature.
    gate = F.linear(x, weights["in_proj_z.weight"]).reshape(
        x.shape[0],
        x.shape[1],
        cfg.linear_num_value_heads,
        cfg.linear_value_head_dim,
    )
    # Normalize over Dv, multiply scale and SiLU(gate), then merge heads:
    # (B,T,Hv,Dv) -> (B,T,Hv*Dv).
    y = gated_rms_norm(
        y, gate, weights["norm.weight"], cfg.rms_norm_eps
    ).reshape(x.shape[0], x.shape[1], -1)
    # Y @ Wout.T: (B,T,Hv*Dv) @ (Hv*Dv,C) -> (B,T,C).
    # Cache retains raw history (B,F,K-1) and memory (B,Hv,Dk,Dv).
    output = F.linear(y, weights["out_proj.weight"])
    delta_cache = DeltaNetCache(history, recurrent_state)
    # Output (B,T,C); cache holds convolution history AND recurrent state.
    return output, delta_cache


if __name__ == "__main__":
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        hidden_size=2560,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0)
    # C=2560; Hk=16 key heads; Hv=32 value heads; Dk=Dv=128; K=4.
    # F = 2*Hk*Dk + Hv*Dv = 8192 combined Q/K/V features.
    # Linear weights are (output features, input features), without B or T.
    shapes = {
        "in_proj_qkv.weight": (8192, 2560),  # (2*Hk*Dk + Hv*Dv, C)
        "conv1d.weight": (8192, 1, 4),  # (F, 1 channel per group, K)
        "in_proj_z.weight": (4096, 2560),  # (Hv*Dv, C): output gate
        "in_proj_a.weight": (32, 2560),  # (Hv, C): decay input
        "in_proj_b.weight": (32, 2560),  # (Hv, C): write-strength input
        "A_log": (32,),  # (Hv,): log decay scale, one per value head
        "dt_bias": (32,),  # (Hv,): bias before decay softplus
        "out_proj.weight": (2560, 4096),  # (C, Hv*Dv): back to hidden width
    }
    weights = {}
    for name, shape in shapes.items():
        weights[name] = torch.randn(shape, device=device) * 0.02
    # (Dv,): learned normalization scale, shared across value heads.
    weights["norm.weight"] = torch.ones(128, device=device)
    tokens = torch.randn(1, 6, 2560, device=device)
    with torch.no_grad():
        output, delta_cache = gated_delta_net(tokens[:, :5], weights, cfg)
        print("Device:", device)
        print(
            "Prefill input/output:",
            tuple(tokens[:, :5].shape),
            tuple(output.shape),
        )
        print("Convolution history:", tuple(delta_cache.convolution.shape))
        print("Recurrent state:", tuple(delta_cache.recurrent_state.shape))
        decoded, next_cache = gated_delta_net(
            tokens[:, 5:], weights, cfg, delta_cache
        )
        complete, _ = gated_delta_net(tokens, weights, cfg)
        torch.testing.assert_close(
            decoded, complete[:, 5:], atol=1e-5, rtol=1e-4
        )
        print("Decode output:", tuple(decoded.shape))
        print(
            "Returned recurrent state:", tuple(next_cache.recurrent_state.shape)
        )
        print("Cached decode matches full sequence.")

        # Exercise the recurrence directly: normalized Q/K -> memory writes ->
        # reads.
        query, key, value = (
            torch.randn(1, 6, 32, 128, device=device) for _ in range(3)
        )
        log_decay = torch.full((1, 6, 32), -0.1, device=device)
        beta = torch.full((1, 6, 32), 0.5, device=device)
        reads, state = gated_delta_scan(
            query[:, :5],
            key[:, :5],
            value[:, :5],
            log_decay[:, :5],
            beta[:, :5],
        )
        next_read, next_state = gated_delta_scan(
            query[:, 5:],
            key[:, 5:],
            value[:, 5:],
            log_decay[:, 5:],
            beta[:, 5:],
            state,
        )
        all_reads, all_state = gated_delta_scan(
            query, key, value, log_decay, beta
        )
        torch.testing.assert_close(next_read, all_reads[:, 5:])
        torch.testing.assert_close(next_state, all_state)
        print("Scan reads/state:", tuple(reads.shape), tuple(state.shape))
        print("Resumed scan matches uninterrupted scan.")

import torch
from torch.nn import functional as F

from .cache import AttentionCache
from .normalization import rms_norm
from .rope import apply_partial_rope


def project_heads(x, weights, cfg, angles):
    # B=batch, T=new tokens, C=hidden width, D=head width.
    # Hq=query heads; Hkv=key/value heads; N=cached + new tokens.
    b, t, _ = x.shape
    hq, hk, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    # X @ Wq.T: (B,T,C) @ (C,2*Hq*D) -> (B,T,2*Hq*D).
    # Reshape (B,T,Hq,2D), then split within each head: Q/gate (B,T,Hq,D).
    q, gate = (
        F.linear(x, weights["q_proj.weight"], weights.get("q_proj.bias"))
        .reshape(b, t, hq, 2 * d)
        .split((d, d), dim=-1)
    )
    # Normalize over D, then transpose: (B,T,Hq,D) -> (B,Hq,T,D).
    q = rms_norm(q, weights["q_norm.weight"], cfg.rms_norm_eps).transpose(1, 2)
    # X @ Wk.T: (B,T,C) @ (C,Hkv*D) -> (B,T,Hkv*D).
    # Expose heads, normalize D, transpose -> (B,Hkv,T,D).
    k = F.linear(
        x, weights["k_proj.weight"], weights.get("k_proj.bias")
    ).reshape(b, t, hk, d)
    k = rms_norm(k, weights["k_norm.weight"], cfg.rms_norm_eps).transpose(1, 2)
    # X @ Wv.T: (B,T,C) @ (C,Hkv*D) -> (B,T,Hkv*D).
    # Reshape and transpose -> (B,Hkv,T,D); no normalization on V.
    v = (
        F.linear(x, weights["v_proj.weight"], weights.get("v_proj.bias"))
        .reshape(b, t, hk, d)
        .transpose(1, 2)
    )
    # Rotate current Q/K using (1,1,T,R/2) angles; shapes stay unchanged.
    q, k = (apply_partial_rope(z, *angles) for z in (q, k))
    # Final shapes: Q (B,Hq,T,D); K/V (B,Hkv,T,D); gate (B,T,Hq,D).
    return q, k, v, gate


def compute_attention_weights(q, k, offset, dropout=0.0, training=False):
    # Q @ K.T / sqrt(D): (B,Hq,T,D) @ (B,Hq,D,N) -> (B,Hq,T,N).
    scores = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    qp = offset + torch.arange(q.shape[2], device=q.device)
    kp = torch.arange(k.shape[2], device=q.device)
    # Absolute-position mask: (1,N) > (T,1) -> (T,N).
    # Broadcast over B,Hq; each query sees only current and earlier keys.
    scores = scores.masked_fill(
        kp.unsqueeze(0) > qp.unsqueeze(1), torch.finfo(scores.dtype).min
    )
    # Softmax over keys N gives attention weights: (B,Hq,T,N).
    attention_weights = F.softmax(
        scores, dim=-1, dtype=torch.float32
    ).to(q.dtype)
    # Training dropout keeps the shape; inference leaves weights intact.
    return F.dropout(attention_weights, p=dropout, training=training)


def full_attention(x, weights, cfg, angles, offset, cache=None, training=False):
    q, k, v, gate = project_heads(x, weights, cfg, angles)
    if cache is not None:
        # Append along time: (B,Hkv,N_old,D) + (B,Hkv,T,D).
        # Result (B,Hkv,N,D), where N=N_old+T; cached keys are already rotated.
        k, v = torch.cat((cache.keys, k), 2), torch.cat((cache.values, v), 2)
    # Keep compact Hkv heads in the cache; no query cache is needed.
    kv_cache = AttentionCache(k, v)
    # Repeat each KV head Hq/Hkv times: (B,Hkv,N,D) -> (B,Hq,N,D).
    k, v = (
        z.repeat_interleave(
            cfg.num_attention_heads // cfg.num_key_value_heads, dim=1
        )
        for z in (k, v)
    )
    attn_weights = compute_attention_weights(
        q, k, offset, cfg.attention_dropout, training
    )
    # (B,Hq,T,N) @ (B,Hq,N,D) -> (B,T,Hq*D).
    y = (attn_weights @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
    # Elementwise sigmoid gate: (B,T,Hq*D) * (B,T,Hq*D).
    y = y * gate.reshape(x.shape[0], x.shape[1], -1).sigmoid()
    # Y @ Wo.T: (B,T,Hq*D) @ (Hq*D,C) -> (B,T,C).
    output = F.linear(
        y, weights["o_proj.weight"], weights.get("o_proj.bias")
    )
    # Output (B,T,C); cached K/V each (B,Hkv,N,D).
    return output, kv_cache


if __name__ == "__main__":
    from types import SimpleNamespace

    from .rope import rope_angles

    cfg = SimpleNamespace(
        hidden_size=2560,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=256,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        max_position_embeddings=262144,
        rope_parameters={
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
        },
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0)
    # 4B projection shapes: (output features, input features).
    shapes = {
        "q_proj.weight": (8192, 2560),  # 16 * (256 query + 256 gate).
        "k_proj.weight": (1024, 2560),  # 4 KV heads * 256.
        "v_proj.weight": (1024, 2560),
        "o_proj.weight": (2560, 4096),
        "q_norm.weight": (256,),
        "k_norm.weight": (256,),
    }
    weights = {}
    for name, shape in shapes.items():
        weights[name] = torch.randn(shape, device=device) * 0.02
    tokens = torch.randn(1, 6, 2560, device=device)
    with torch.no_grad():
        angles, tables = rope_angles(cfg, None, tokens, offset=0, length=5)
        output, attention_cache = full_attention(
            tokens[:, :5], weights, cfg, angles, offset=0
        )
        print("Device:", device)
        print(
            "Prefill input/output:",
            tuple(tokens[:, :5].shape),
            tuple(output.shape),
        )
        print(
            "Cached keys/values:",
            tuple(attention_cache.keys.shape),
            tuple(attention_cache.values.shape),
        )

        # Cache length supplies the absolute position of the next token.
        offset = attention_cache.keys.shape[2]
        angles, tables = rope_angles(
            cfg, tables, tokens, offset=offset, length=1
        )
        decoded, next_cache = full_attention(
            tokens[:, 5:], weights, cfg, angles, offset, attention_cache
        )
        angles, tables = rope_angles(cfg, tables, tokens, offset=0, length=6)
        complete, _ = full_attention(tokens, weights, cfg, angles, offset=0)
        torch.testing.assert_close(
            decoded, complete[:, 5:], atol=1e-6, rtol=1e-5
        )
        assert attention_cache.keys.shape[2] == 5
        print("Decode output:", tuple(decoded.shape))
        print("Returned keys:", tuple(next_cache.keys.shape))
        print(
            (
                "Cached decode matches full sequence; original cache "
                "still holds five tokens."
            )
        )

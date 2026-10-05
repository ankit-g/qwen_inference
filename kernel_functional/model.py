from torch.nn import functional as F

from .cache import HybridCache
from .error_handling import validate_functional_inputs, validate_linear_backend
from .full_attention import full_attention
from .gated_delta_net import gated_delta_net
from .normalization import rms_norm
from .rope import rope_angles


def subweights(weights, prefix):
    # Select names for one layer; tensor storage is shared, not copied.
    selected_weights = {}
    for full_name, tensor in weights.items():
        if full_name.startswith(prefix):
            local_name = full_name.removeprefix(prefix)
            selected_weights[local_name] = tensor
    return selected_weights


def feed_forward(x, weights):
    # X @ Wgate.T: (B,T,C) @ (C,I) -> (B,T,I); apply SiLU.
    gate = F.silu(F.linear(x, weights["gate_proj.weight"]))
    # X @ Wup.T: (B,T,C) @ (C,I) -> (B,T,I).
    up = F.linear(x, weights["up_proj.weight"])
    # Elementwise gate*up, then @ Wdown.T: (B,T,I) @ (I,C) -> (B,T,C).
    return F.linear(gate * up, weights["down_proj.weight"])


def decoder_layer(
    x,
    weights,
    cfg,
    layer_type,
    angles,
    offset,
    layer_cache=None,
    training=False,
    backend="chunked",
):
    # Pre-norm attention: normalize C while keeping (B,T,C).
    normed = rms_norm(x, weights["input_layernorm.weight"], cfg.rms_norm_eps)
    if layer_type == "linear_attention":
        y, delta_cache = gated_delta_net(
            normed,
            subweights(weights, "linear_attn."),
            cfg,
            layer_cache,
            backend,
        )
        next_layer_cache = delta_cache
    else:
        y, kv_cache = full_attention(
            normed,
            subweights(weights, "self_attn."),
            cfg,
            angles,
            offset,
            layer_cache,
            training,
        )
        next_layer_cache = kv_cache
    # Residual addition, not matmul: (B,T,C) + (B,T,C) -> (B,T,C).
    x = x + y
    normed = rms_norm(
        x, weights["post_attention_layernorm.weight"], cfg.rms_norm_eps
    )
    # Second residual: X + MLP(RMSNorm(X)), again (B,T,C).
    output = x + feed_forward(normed, subweights(weights, "mlp."))
    return output, next_layer_cache


def model_forward(
    input_ids,
    weights,
    cfg,
    rope_tables=None,
    cache=None,
    logits_to_keep=0,
    training=False,
    backend="chunked",
):
    cfg.validate()
    validate_linear_backend(backend)
    validate_functional_inputs(input_ids, weights, cfg, cache, logits_to_keep)
    offset = 0 if cache is None else cache.position
    # Row lookup in E(Vocab,C): token IDs (B,T) -> hidden vectors (B,T,C).
    x = F.embedding(input_ids, weights["model.embed_tokens.weight"])
    # Select angles (1,1,T,R/2) once for all full-attention layers.
    angles, rope_tables = rope_angles(
        cfg, rope_tables, x, offset, input_ids.shape[1]
    )
    next_layers = []
    # Every layer preserves (B,T,C); each receives only its own cache.
    for i, kind in enumerate(cfg.layer_types):
        x, layer_cache = decoder_layer(
            x,
            subweights(weights, f"model.layers.{i}."),
            cfg,
            kind,
            angles,
            offset,
            None if cache is None else cache.layers[i],
            training,
            backend,
        )
        if cache is not None:
            next_layers.append(layer_cache)
    x = rms_norm(x, weights["model.norm.weight"], cfg.rms_norm_eps)
    # H @ Wlm.T: (B,T_keep,C) @ (C,Vocab) -> (B,T_keep,Vocab).
    # Select the last T_keep positions; zero means use all T positions.
    logits = F.linear(
        x[:, -logits_to_keep:] if logits_to_keep else x,
        weights["lm_head.weight"],
    )
    # Advance by T processed tokens; sampling alone does not extend a cache.
    next_cache = (
        None
        if cache is None
        else HybridCache(
            tuple(next_layers),
            offset + input_ids.shape[1],
            tuple(cfg.layer_types),
            weights,
            input_ids.shape[0],
            input_ids.device,
            weights["model.embed_tokens.weight"].dtype,
        )
    )
    return logits, next_cache, rope_tables


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import torch
    from transformers import AutoTokenizer

    from .cache import create_cache
    from .weights import load_weights

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt", default="What is 2 + 2?")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    weights, cfg, report = load_weights(
        args.checkpoint, dtype=dtype, device=device
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, local_files_only=True
    )
    messages = [{"role": "user", "content": args.prompt}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = torch.tensor([input_ids], device=device)
    cache = create_cache(cfg)
    tables = None
    generated_ids = []
    print("Device:", device, "Dtype:", dtype)
    print("Loaded:", report)
    print("Input tokens:", tuple(tokens.shape))

    with torch.no_grad():
        for step in range(args.max_new_tokens):
            # Prefill (1,T); later calls process only the new token (1,1).
            logits, cache, tables = model_forward(
                tokens, weights, cfg, tables, cache, logits_to_keep=1
            )
            # Last-position logits (1,1,Vocab) -> greedy next token (1,1).
            tokens = logits[:, -1].argmax(dim=-1, keepdim=True)
            token_id = tokens.item()
            if step == 0:
                print("Prefill logits:", tuple(logits.shape))
            if token_id == tokenizer.eos_token_id:
                break
            generated_ids.append(token_id)
            if cache.position >= cfg.max_position_embeddings:
                break
    print("Processed cache positions:", cache.position)
    print(
        "Response:", tokenizer.decode(generated_ids, skip_special_tokens=True)
    )

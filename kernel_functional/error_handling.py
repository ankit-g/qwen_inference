import torch


def validate_config(cfg):
    if len(cfg.layer_types) != cfg.num_hidden_layers:
        raise ValueError("Layer count mismatch")
    if set(cfg.layer_types) - {"linear_attention", "full_attention"}:
        raise ValueError(
            "Only dense linear/full attention blocks are supported"
        )
    # Head repetition must map whole groups: Hq/Hkv and Hv/Hk integers.
    if cfg.num_attention_heads % cfg.num_key_value_heads:
        raise ValueError("Query heads must be divisible by KV heads")
    if cfg.linear_num_value_heads % cfg.linear_num_key_heads:
        raise ValueError("Value heads must be divisible by linear key heads")
    if cfg.hidden_act != "silu" or not cfg.attn_output_gate:
        raise ValueError(
            "This implementation requires SiLU and attention output gating"
        )
    if cfg.rope_parameters.get("rope_type", "default") != "default":
        raise ValueError("Only default text RoPE is implemented")
    rotary_dim = int(
        cfg.head_dim * cfg.rope_parameters.get("partial_rotary_factor", 1.0)
    )
    if rotary_dim <= 0 or rotary_dim > cfg.head_dim or rotary_dim % 2:
        raise ValueError(
            "Rotary dimension must be positive, even, and at most head_dim"
        )


def validate_config_type(raw):
    if raw.get("model_type", "qwen3_5_text") != "qwen3_5_text":
        raise ValueError("Expected a Qwen3.5 dense text configuration")


def validate_linear_backend(backend):
    if backend not in ("chunked", "recurrent"):
        raise ValueError("Expected 'chunked' or 'recurrent'")


def validate_rope_positions(offset, end, max_positions):
    # Requested table slice [offset:end] represents the current T positions.
    if offset < 0 or end > max_positions:
        raise ValueError("Positions exceed configured context length")


def validate_chunk_inputs(query, chunk_size):
    # query is (B,T,H,Dk); T and the block width must both be positive.
    if chunk_size < 1 or query.shape[1] < 1:
        raise ValueError("Chunk size and sequence length must be positive")


def validate_checkpoint_files(files, directory):
    if not files:
        raise FileNotFoundError(f"No safetensors in {directory}")


def validate_checkpoint_tensor(key, mapped, expected, loaded, handle):
    if mapped not in expected:
        raise ValueError(f"Unexpected checkpoint tensor: {key}")
    if mapped in loaded:
        raise ValueError(f"Duplicate checkpoint tensor: {key}")
    if tuple(handle.get_slice(key).get_shape()) != tuple(
        expected[mapped].shape
    ):
        raise ValueError(f"Checkpoint shape mismatch: {key}")


def validate_tied_weights(loaded, embedding):
    # Tied embedding and output matrices must agree elementwise: (Vocab,C).
    if "lm_head.weight" in loaded and not torch.equal(
        loaded["lm_head.weight"], embedding
    ):
        raise ValueError("Tied output and embedding weights disagree")


def validate_checkpoint_complete(expected, loaded):
    missing = set(expected) - set(loaded)
    if missing:
        raise ValueError(f"Missing text tensors: {sorted(missing)}")


def validate_functional_inputs(input_ids, weights, cfg, cache, logits_to_keep):
    if input_ids.ndim != 2 or not input_ids.shape[0] or not input_ids.shape[1]:
        raise ValueError("Expected nonempty unpadded (batch, tokens) IDs")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Token IDs must be integers")
    if logits_to_keep < 0:
        raise ValueError("logits_to_keep must be nonnegative")
    weight = weights["model.embed_tokens.weight"]
    if input_ids.device != weight.device:
        raise ValueError("Token IDs and weights must share a device")
    if cache is not None:
        if torch.is_grad_enabled():
            raise ValueError(
                "Use torch.no_grad() with cache; train with cache=None"
            )
        if (
            cache.layer_types != tuple(cfg.layer_types)
            or len(cache.layers) != cfg.num_hidden_layers
        ):
            raise ValueError("Cache architecture mismatch")
        if cache.owner is not None and cache.owner is not weights:
            raise ValueError(
                "Cache belongs to different weights; create a fresh cache"
            )
        if cache.position and (
            cache.batch_size != input_ids.shape[0]
            or cache.device != input_ids.device
            or cache.dtype != weight.dtype
        ):
            raise ValueError("Cached batch size, device or dtype changed")


def validate_tensor_shape(key, mapped, shapes, loaded, handle):
    if mapped not in shapes:
        raise ValueError(f"Unexpected checkpoint tensor: {key}")
    if mapped in loaded:
        raise ValueError(f"Duplicate checkpoint tensor: {key}")
    if tuple(handle.get_slice(key).get_shape()) != shapes[mapped]:
        raise ValueError(f"Checkpoint shape mismatch: {key}")

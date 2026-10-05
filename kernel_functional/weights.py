import json
from pathlib import Path

import torch

from .config import Qwen35Config
from .error_handling import (
    validate_checkpoint_complete,
    validate_checkpoint_files,
    validate_tied_weights,
)


def weight_shapes(cfg):
    # Linear weights use (output,input); forward multiplies X @ W.T.
    # C=hidden width, I=MLP width; these shapes do not include B or T.
    c, i = cfg.hidden_size, cfg.intermediate_size
    shapes = {
        "model.embed_tokens.weight": (cfg.vocab_size, c),
        "lm_head.weight": (cfg.vocab_size, c),
        "model.norm.weight": (c,),
    }
    for index, kind in enumerate(cfg.layer_types):
        local = {
            "input_layernorm.weight": (c,),
            "post_attention_layernorm.weight": (c,),
            "mlp.gate_proj.weight": (i, c),
            "mlp.up_proj.weight": (i, c),
            "mlp.down_proj.weight": (c, i),
        }
        if kind == "linear_attention":
            # ks=Hk*Dk, vs=Hv*Dv; combined QKV width F=2*ks+vs.
            ks = cfg.linear_num_key_heads * cfg.linear_key_head_dim
            vs = cfg.linear_num_value_heads * cfg.linear_value_head_dim
            hv, dv = cfg.linear_num_value_heads, cfg.linear_value_head_dim
            local.update(
                {
                    f"linear_attn.{n}": shape
                    for n, shape in {
                        "in_proj_qkv.weight": (2 * ks + vs, c),
                        "conv1d.weight": (
                            2 * ks + vs,
                            1,
                            cfg.linear_conv_kernel_dim,
                        ),
                        "in_proj_z.weight": (vs, c),
                        "in_proj_a.weight": (hv, c),
                        "in_proj_b.weight": (hv, c),
                        "A_log": (hv,),
                        "dt_bias": (hv,),
                        "norm.weight": (dv,),
                        "out_proj.weight": (c, vs),
                    }.items()
                }
            )
        else:
            # qw=Hq*D, kw=Hkv*D; query projection also produces qw gate values.
            qw = cfg.num_attention_heads * cfg.head_dim
            kw = cfg.num_key_value_heads * cfg.head_dim
            for name, shape in {
                "q_proj": (2 * qw, c),
                "k_proj": (kw, c),
                "v_proj": (kw, c),
                "o_proj": (c, qw),
            }.items():
                local[f"self_attn.{name}.weight"] = shape
                if cfg.attention_bias:
                    local[f"self_attn.{name}.bias"] = (shape[0],)
            local["self_attn.q_norm.weight"] = (cfg.head_dim,)
            local["self_attn.k_norm.weight"] = (cfg.head_dim,)
        shapes.update(
            {
                f"model.layers.{index}.{name}": shape
                for name, shape in local.items()
            }
        )
    return shapes


def load_weights(
    directory, dtype=torch.bfloat16, device="cpu", trainable=False
):
    from safetensors import safe_open

    from .error_handling import validate_tensor_shape

    directory = Path(directory)
    cfg = Qwen35Config.from_dict(
        json.loads((directory / "config.json").read_text())
    )
    shapes = weight_shapes(cfg)
    files = sorted(directory.glob("*.safetensors"))
    validate_checkpoint_files(files, directory)
    loaded, excluded = {}, []
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(
                    ("model.visual.", "visual.", "mtp.", "model.mtp.")
                ):
                    excluded.append(key)
                    continue
                mapped = key.replace("model.language_model.", "model.", 1)
                validate_tensor_shape(key, mapped, shapes, loaded, handle)
                # Keep checkpoint axes; move/cast without transposing.
                # Detach creates leaf weights for optional training.
                loaded[mapped] = (
                    handle.get_tensor(key)
                    .to(device=device, dtype=dtype)
                    .detach()
                    .requires_grad_(trainable)
                )
    if cfg.tie_word_embeddings and "model.embed_tokens.weight" in loaded:
        embedding = loaded["model.embed_tokens.weight"]
        validate_tied_weights(loaded, embedding)
        # Share E(Vocab,C): embeddings select rows; logits use H @ E.T.
        loaded["lm_head.weight"] = embedding
    validate_checkpoint_complete(shapes, loaded)
    report = {
        "checkpoint": str(directory),
        "loaded_state_entries": len(loaded),
        "excluded_tensors": len(excluded),
    }
    return loaded, cfg, report


def trainable_tensors(weights):
    # Deduplicate shared tensors so an optimizer updates tied weights once.
    return list(
        {id(t): t for t in weights.values() if t.requires_grad}.values()
    )

# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights
# reserved.
# Licensed under Apache-2.0; see ../LICENSE-qwen35-transformers.txt.


from dataclasses import dataclass, fields

from .error_handling import validate_config, validate_config_type


@dataclass
class Qwen35Config:
    # Values below were read from the local Qwen3.5-4B checkpoint config.json
    # (text_config). They describe the pretrained text architecture, not tensor
    # contents.
    # Required fields remain configurable; the 4B examples do not set new
    # defaults.

    # Number of token rows in the embedding table and output logits.
    # 4B: 248320; embedding/LM-head weight shape = (248320, 2560).
    vocab_size: int
    # Residual-stream width C: features carried by every token between blocks.
    # 4B: 2560; hidden tensors have shape (B, T, 2560).
    hidden_size: int
    # MLP expansion width I, used by both gate and up projections.
    # 4B: 9216; MLP flow is 2560 -> 9216 -> 2560 per token.
    intermediate_size: int
    # Number of decoder blocks, each with attention, two norms, and an MLP.
    # 4B: 32 total = 24 DeltaNet blocks + 8 full-attention blocks.
    num_hidden_layers: int
    # Full-attention query-head count Hq (separate from DeltaNet heads).
    # 4B: 16; concatenated query width Hq*D = 16*256 = 4096.
    num_attention_heads: int
    # Full-attention key/value-head count Hkv, stored in the KV cache.
    # 4B: 4; each KV head serves Hq/Hkv = 4 query heads.
    num_key_value_heads: int
    # Width D of each full-attention query/key/value head.
    # 4B: 256, explicitly configured; it is NOT hidden_size/num_attention_heads.
    # K/V projection width = 4*256 = 1024; Q+gate projection width = 16*2*256 =
    # 8192.
    head_dim: int
    # Ordered attention type for each decoder block, using zero-based indices.
    # 4B: ["linear_attention"]*3 + ["full_attention"], repeated 8 times.
    # Full attention is at indices 3,7,11,15,19,23,27,31; other blocks use
    # DeltaNet.
    layer_types: list[str]
    # DeltaNet query/key-head count Hk before repetition to match value heads.
    # 4B: 16; each Q/K projection has Hk*Dk = 16*128 = 2048 features.
    linear_num_key_heads: int
    # DeltaNet value-head count Hv, also the number of recurrent memory
    # matrices.
    # 4B: 32; Q/K heads repeat Hv/Hk = 2 times; V width = 32*128 = 4096.
    linear_num_value_heads: int
    # DeltaNet key/query width Dk; the row dimension of each memory matrix S.
    # 4B: 128; S has shape (B,32,128,128), with rows indexed by key features.
    linear_key_head_dim: int
    # DeltaNet value width Dv; the column dimension of S and each readout width.
    # 4B: 128; q(1,128) @ S(128,128) -> one 128-feature value per head.
    linear_value_head_dim: int
    # Causal depthwise convolution window K, including the current token.
    # 4B: 4; retain K-1 = 3 previous raw projected tokens per channel.
    # F = 2*16*128 + 32*128 = 8192 channels; conv weight shape = (8192,1,4).
    linear_conv_kernel_dim: int
    # Position-rotation settings for full attention; exact 4B dictionary:
    # {"rope_type": "default", "rope_theta": 10000000,
    #  "partial_rotary_factor": 0.25, "mrope_interleaved": True,
    #  "mrope_section": [11, 11, 10]}.
    # rope_theta sets angular speeds: theta_j = base^(-2j/R).
    # partial_rotary_factor gives R = 256*0.25 = 64 rotated coordinates (32
    # pairs).
    # rope_type="default" selects the unscaled frequency schedule.
    # The mrope fields describe multimodal axis layout; this text-only path uses
    # one shared position for all axes and does not implement multimodal grids.
    rope_parameters: dict
    # Configured position limit, not an allocation size or measured context
    # capacity.
    # 4B: 262144; valid positions run from 0 through 262143.
    # RoPE tables cover all positions from the first call; full-attention KV
    # memory still grows with sequence length.
    max_position_embeddings: int = 262144
    # Epsilon inside sqrt(mean(x^2) + eps), preventing division by zero in
    # RMSNorm.
    # 4B: 1e-6; used by residual, Q/K, and gated output RMS normalization.
    rms_norm_eps: float = 1e-6
    # Whether input embedding and output vocabulary head share the same
    # Parameter.
    # 4B: True; input uses E[id], output uses h @ E.T with E=(248320,2560).
    tie_word_embeddings: bool = True
    # Whether full-attention Q/K/V/output linear projections include additive
    # biases.
    # 4B: False; these projections compute X @ W.T without a bias vector.
    attention_bias: bool = False
    # Drop probability applied to softmax attention weights during training.
    # 4B: 0.0; no attention weights are dropped. Evaluation always disables
    # dropout.
    attention_dropout: float = 0.0
    # Required activation family for this implementation.
    # 4B: "silu", where SiLU(x)=x*sigmoid(x); used in the gated MLP and
    # DeltaNet.
    hidden_act: str = "silu"
    # Whether full attention uses a feature-wise sigmoid gate on the retrieved
    # output.
    # 4B: True; output = (softmax(QK.T/sqrt(D)) @ V) * sigmoid(gate).
    # This implementation requires the gate; False is rejected by validation.
    attn_output_gate: bool = True

    @classmethod
    def from_dict(cls, raw):
        # Read dimensions and layer order without allocating tensors.
        raw = raw.get("text_config", raw)
        validate_config_type(raw)
        allowed = {f.name for f in fields(cls)}
        cfg = cls(**{k: v for k, v in raw.items() if k in allowed})
        cfg.validate()
        return cfg

    def validate(self):
        validate_config(self)

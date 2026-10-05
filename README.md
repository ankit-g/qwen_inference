# Qwen inference

A readable, functional PyTorch implementation of the Qwen3.5 text model.
No `torch.nn.Module` or `nn.Parameter` objects are constructed. Weights,
RoPE tables, and caches are explicit inputs and outputs.

## Setup

Use Python 3.10 or newer. Install a PyTorch build appropriate for your GPU,
then install the dependencies:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell.
CUDA validation used PyTorch 2.11.0 and Transformers 5.17.0.
Transformers supplies the tokenizer; our code performs the forward pass.

## Run

Provide a local Qwen3.5 checkpoint directory containing its config,
safetensors weights, tokenizer files, and chat template. Weights are not
included or automatically downloaded.

```sh
python -m kernel_functional.model /path/to/Qwen3.5-0.8B \
  --device cuda --prompt "What is 2 + 2?" --max-new-tokens 16
```

Devices: `cuda`, `mps`, or `cpu`. GPU inference uses BF16; CPU uses FP32.
The default device preference is CUDA, MPS, then CPU.

## Functional API

```python
import torch
from kernel_functional import create_cache, load_weights, model_forward

weights, cfg, report = load_weights(
    "/path/to/Qwen3.5-0.8B", device="cuda", dtype=torch.bfloat16
)
cache = create_cache(cfg)
tables = None
# input_ids: unpadded (batch, tokens), using the matching tokenizer.
with torch.no_grad():
    logits, cache, tables = model_forward(
        input_ids, weights, cfg, tables, cache, logits_to_keep=1
    )
    next_ids = logits[:, -1].argmax(-1, keepdim=True)
    logits, cache, tables = model_forward(
        next_ids, weights, cfg, tables, cache, logits_to_keep=1
    )
```

Pass `cache=None` for independent decisions without retained history.
Capture returned caches when decoding. RoPE tables can be reused across
independent requests. Cache containers are frozen, but their tensors are
mutable; do not modify them in place.

## Components

- `model.py`: feed-forward layers, decoder blocks, and model forward.
- `full_attention.py`: grouped-query causal attention.
- `gated_delta_net.py`: local convolution and recurrent DeltaNet.
- `qwen35_chunked.py`: chunked DeltaNet for prefill.
- `rope.py`: partial rotary embeddings and reusable tables.
- `weights.py`, `config.py`: checkpoint loading and configuration.
- `cache.py`, `normalization.py`, `error_handling.py`: supporting code.

Runnable layer demonstrations use random weights with actual 4B dimensions:

```sh
python -m kernel_functional.rope
python -m kernel_functional.full_attention
python -m kernel_functional.gated_delta_net
```

These layer examples select MPS when available, otherwise CPU. Python may
emit a runpy warning because package exports import the module before its
example is executed.

## Validation and limits

In the originating project, six functional tests passed on CUDA. The 0.8B
FP32 comparison against Transformers passed 26 checks. BF16 cached decoding
had a next-token discrepancy, so exact reference parity is not established.
Those reference-dependent tests are not included in this standalone export.

On an RTX 4070 Ti, BF16, batch one, a reused 1,000-case Pong supplied-distance
test scored 964/1,000 for 0.8B and 1,000/1,000 for 4B. This tested textual
selection among Python-calculated distances, not learned physics or live
self-play. Throughput was 15.1 and 12.2 decisions/sec respectively.

Scope: dense text and equal-length unpadded batches, default partial RoPE,
explicit full attention, and recurrent/chunked DeltaNet. No vision, paged
KV cache, request scheduler, or beam-cache reordering. Long prefills can
require substantial attention memory.

## Attribution

Parts derive from the existing Qwen/Hugging Face implementation. The
source preserves its attribution; see LICENSE-qwen35-transformers.txt
for the accompanying Apache-2.0 license.

![ci](https://github.com/emiliomunozai/SequenceFactorBenchmark/actions/workflows/ci.yml/badge.svg)

# FB-Bench

Factor Bottleneck Benchmark for sequence models under a fixed information budget.

## Overview

Every model must compress a sequence into a fixed bottleneck vector before reconstruction.

- An **encoder** maps tokens `[B, L]` to a latent `[B, bottleneck_dim]`.
- A **decoder** reconstructs logits `[B, L, V]` from that latent.

Encoders and decoders are independent — any encoder can pair with any decoder.

Active families:

- Encoders: `mlp`, `lstm`, `bi_lstm`, `rnn`, `transformer`
- Decoders: `mlp`, `lstm`, `rnn`, `transformer`

The transformer encoder uses a learned bottleneck token and the decoder expands it back into position-indexed memory slots, while still keeping the single latent vector contract.

## Setup

Python 3.12+.

```powershell
py -m pip install --user uv
uv sync --group dev --group cpu   # or --group gpu for CUDA
```

## CLI

Two commands:

```powershell
uv run sfb run -c configs/single_model_example.yaml
uv run sfb predict data/checkpoints/copy_lstm_rnn_0.pt -n 5
```

`sfb run` takes one or more YAML configs. List-valued fields are swept (Cartesian product). There is no separate sweep command.

Options:

```
-c, --config YAML [YAML ...]   Config file(s)
-o, --output PATH              Results CSV (default: data/results/results.csv)
--overwrite                    Replace existing results
--no-save-model                Skip checkpoint writing
--device {auto,cpu,cuda}
```

## Config patterns

### Single model

```yaml
task: copy
loss: cross_entropy

encoder: lstm
decoder: rnn

sequence_length: 32
vocabulary_size: 64
input_noise: 0.0
corruption_mode: replace

d_model: 128
bottleneck_dim: 32
n_layers: 2

batch_size: 64
steps: 5000
eval_every: 500
seed: 42
```

### Multiple explicit models

Use `models:` to define specific encoder-decoder pairs instead of sweeping all combinations:

```yaml
task: copy
loss: cross_entropy
sequence_length: [64, 128, 256]
vocabulary_size: [64, 128, 256]
input_noise: [0.0, 0.1]
corruption_mode: replace

models:
  - encoder: mlp
    decoder: mlp

  - encoder: lstm
    decoder: lstm
    n_layers: 1

  - encoder: bi_lstm
    decoder: lstm
    n_layers: 1

  - encoder: rnn
    decoder: rnn
    n_layers: 1

  - encoder: transformer
    decoder: transformer
    n_layers: 2
    n_heads: 4

d_model: 64
bottleneck_dim: 32
batch_size: 128
steps: 5000
eval_every: 1000
seed: 42
```

List-valued fields (`sequence_length`, `vocabulary_size`, `input_noise` above) are crossed with each model pair.

### Defaults

Any key omitted from a config falls back to the built-in defaults. `bottleneck_dim` defaults to `d_model` when not specified.

## Output

Each run writes:

- `data/results/results.csv` — one summary row per run
- `data/results/runs_detail.jsonl` — full config, step history, and checkpoint path per run
- `data/checkpoints/` — model weights (`.pt`)

Runs already present in the CSV (matched by config signature) are skipped automatically. Use `--overwrite` to start fresh.

## Validation

Two eval modes are tracked at each checkpoint:

- **clean**: clean input → clean target
- **noisy**: corrupted input → clean target

The noisy score is the primary robustness metric and drives early stopping.

## Extending

Add encoders in `src/sfb/models/encoders/`, decoders in `src/sfb/models/decoders/`. Register with the `@register_encoder` / `@register_decoder` decorators.

Contract:
- Encoders return `EncoderOutput(z=...)` with shape `[B, bottleneck_dim]`
- Decoders map that latent to logits `[B, L, V]`

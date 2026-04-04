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

## Project description

### Motivation

Comparing sequence model architectures is difficult when models differ simultaneously in depth, width, parameter count, and representational capacity. Performance differences may reflect architectural merit or simply differences in information throughput. FB-Bench isolates architecture as the experimental variable by forcing every model through a fixed-dimensional bottleneck, equalizing the information budget available for sequence reconstruction.

### Formulation

Let \( x \in \{0, \dots, V{-}1\}^L \) be an input sequence of length \( L \) over a vocabulary of size \( V \). The benchmark decomposes every model into two stages:

1. **Encoder** \( f_\theta : \mathbb{Z}^{L} \to \mathbb{R}^{d} \). Maps the discrete input to a single latent vector \( z = f_\theta(x) \) of fixed dimension \( d \) (the bottleneck dimension), regardless of sequence length.
2. **Decoder** \( g_\phi : \mathbb{R}^{d} \to \mathbb{R}^{L \times V} \). Reconstructs token-level logits from \( z \) alone, with no residual connection or skip path from the encoder.

The full model is their composition \( g_\phi(f_\theta(x)) \), trained end-to-end with a token-level loss.

### Tasks

Each task defines a deterministic mapping from the clean input \( x \) to a target sequence \( y \):

| Task | Target | What it measures |
|------|--------|------------------|
| **Copy** | \( y = x \) | Whether the bottleneck can preserve token identity and order |
| **Reverse** | \( y_t = x_{L-1-t} \) | Whether the bottleneck supports non-trivial reordering |
| **Sorting** | \( y = \text{sort}(x) \) | Whether the bottleneck captures distributional content independent of position |

Input sequences are sampled uniformly at random from \( \{0, \dots, V{-}1\}^L \). Targets are always computed from the clean (uncorrupted) input.

### Input corruption

To probe robustness, the encoder input may be corrupted at a configurable noise rate \( \eta \in [0, 1] \). Each token is independently corrupted with probability \( \eta \). Two corruption modes are supported:

- **Replace**: the token is replaced by a uniformly random token guaranteed to differ from the original.
- **Mask**: the token is replaced by a fixed mask token ID.

Corruption is applied to the encoder input only; the target \( y \) is always derived from the clean sequence. During training, corruption is applied by default. At evaluation time, two scores are reported: clean (no corruption) and noisy (corruption applied), allowing separation of modeling capacity from noise robustness.

### Loss functions

- **Cross-entropy**: standard token-level cross-entropy over the vocabulary, averaged across positions and batch.
- **Shift-tolerant cross-entropy**: at each position \( t \), the one-hot target is replaced by a soft distribution blending targets from neighboring positions \( [t{-}w, t{+}w] \), weighted by a Gaussian kernel with standard deviation \( \sigma = w/2 \). This relaxes the positional alignment requirement: a copy shifted by one position incurs a small penalty rather than full cross-entropy loss, while a perfectly aligned prediction still achieves the global optimum.

### Experimental protocol

A sweep configuration specifies sets of values for each hyperparameter (encoder, decoder, task, sequence length, vocabulary size, noise rate, etc.). The benchmark takes the Cartesian product of all list-valued fields and trains one model per combination. For each run:

1. A random seed is fixed (when provided) for reproducibility.
2. The encoder and decoder are instantiated with shared architectural parameters (\( d_\text{model} \), \( d \), number of layers, number of attention heads where applicable).
3. The model is trained with Adam (lr = 1e-3) for a fixed number of steps.
4. At regular intervals, clean and noisy validation accuracy are computed. Early stopping monitors noisy validation accuracy with a patience window and a no-hope threshold (minimum accuracy after a configurable number of evaluations).
5. The final reported metrics are taken from the last evaluation step, or from the best noisy validation checkpoint if early stopping triggered.

### Measured quantities

Each run produces a summary row containing:

- Architecture identifiers (encoder, decoder, task, loss)
- Data parameters (sequence length, vocabulary size, noise rate, corruption mode)
- Model parameters (d_model, bottleneck_dim, n_layers, n_heads, trainable parameter count)
- Training parameters (batch size, steps, eval frequency, seed)
- Final metrics: train loss/accuracy, clean validation loss/accuracy, noisy validation loss/accuracy
- Training time, early stopping status, and step at which training stopped

### Controlled variables

The bottleneck dimension \( d \) is the primary controlled variable. By fixing \( d \) across architectures, the benchmark ensures that all models operate under the same information constraint. Differences in final accuracy reflect how effectively each encoder-decoder family utilizes a fixed-capacity latent representation, rather than differences in raw model size or width.

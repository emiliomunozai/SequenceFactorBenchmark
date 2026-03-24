![ci](https://github.com/emiliomunozai/SequenceFactorBenchmark/actions/workflows/ci.yml/badge.svg)

# SeqFactorBench

Sequence model benchmark for copy, sorting, and reverse tasks under controlled scale, breadth, structure, and noise.

---

## Setup

**Python 3.12+.** Dependencies and venv via `[uv](https://github.com/astral-sh/uv)`.

### Install uv

- **Quick (Python already installed):** `py -m pip install --user uv`
- **Fresh machines (standalone, recommended):**

  ```powershell
  powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

  On Windows, if `uv` is not found, add the install dir to your user PATH (then open a new terminal):

  ```powershell
  [Environment]::SetEnvironmentVariable(
    "Path",
    $env:Path + ";C:\Users\$env:USERNAME\.local\bin",
    [EnvironmentVariableTarget]::User
  )
  ```

### Sync environment


| Goal                          | Command                           |
| ----------------------------- | --------------------------------- |
| Tests/lint only (no torch)    | `uv sync --group dev`             |
| Run benchmarks (with PyTorch) | `uv sync --group dev --group gpu` |


Use the CLI with `uv run sfb ...` (no need to activate the venv). Check GPU: `uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"`.

**Dependency layers:** Core (pandas, matplotlib, PyYAML, etc.) + optional dev (pytest, ruff) + optional GPU (PyTorch). CPU-only setups skip the GPU group.

---

## CLI overview

After `uv sync --group dev --group gpu`, run `sfb` via `**uv run sfb ...`** or activate the venv and run `sfb ...`.

**List tasks, models, losses, defaults:**

```bash
sfb list --tasks    # copy, sorting, reverse, ...
sfb list --models   # simple_nn, gru, ...
sfb list --losses   # cross_entropy, shift_tolerant_ce
sfb list --config   # default config values
sfb version
```

**Data layout** (under `data/`):


| Path                | Purpose                                                               |
| ------------------- | --------------------------------------------------------------------- |
| `data/results/`     | Results table (`results.csv`); `run` and `sweep` append here          |
| `data/traces/`      | Step-level history (e.g. `sfb run --trace`) for learning curves       |
| `data/checkpoints/` | Saved model weights (saved by default; use `--no-save-model` to skip) |
| `data/figures/`     | Saved plots; `sfb report --save name.png` writes here by default      |


---

## Run a single experiment

Good for trying one task/model and inspecting behavior. **Copy task** is a minimal example: the model must output the input sequence unchanged.

**Basic copy run (defaults: seq_len 32, vocab 64, 5000 steps):**

```bash
sfb run -t copy -m simple_nn
```

Results: one row appended to `data/results/results.csv`, and a checkpoint saved to `data/checkpoints/` (e.g. `copy_simple_nn_0.pt`).

**Copy with explicit size and training settings:**

```bash
sfb run -t copy -m gru --seq-len 32 --vocab-size 64 -n 5000 -b 64 --eval-every 1000
```

**Save step-level history** for learning-curve plots (writes to `data/traces/` or a path you give):

```bash
sfb run -t copy -m simple_nn --trace
# or
sfb run -t copy -m simple_nn -o my_trace.csv
```

**See example predictions** after training:

```bash
sfb run -t copy -m gru --show-examples 5
```

**Other tasks:** same pattern. Examples:

```bash
sfb run -t sorting -m simple_nn --seq-len 32 --vocab-size 64
sfb run -t reverse -m gru --seq-len 32 -n 5000
```

**YAML config** (overrides with CLI): `sfb run -t copy -c configs/generation_default.yaml --steps 2000`.

**Skip saving checkpoint:** `sfb run -t copy -m simple_nn --no-save-model`

---

## Run a full sweep

Run many (task × model × hyperparameter) combinations from one or more YAML configs. List values in a config become grid dimensions (Cartesian product). You can pass multiple configs; all runs are merged and deduplicated.

```bash
sfb sweep -c configs/run_sweep.yaml
# Exhaustive copy: all models, one scale (7 runs)
sfb sweep -c configs/copy_all_models.yaml
# Multiple configs in one go
sfb sweep -c configs/copy_all_models.yaml -c configs/run_sweep.yaml
```

- Results append to `data/results/results.csv` (or `-o path.csv`). Use `--overwrite` to replace the file.
- Each run saves a checkpoint by default; use `--no-save-model` to disable.

**Sweep options:**

| Option            | Description                                                          |
| ----------------- | -------------------------------------------------------------------- |
| `-c` / `--config` | One or more sweep YAML paths (required); list values = grid; multiple files = union of runs |
| `-o` / `--output` | Results CSV (default: data/results/results.csv) |
| `--overwrite`     | Replace results file instead of appending       |
| `--no-save-model` | Do not save checkpoints                         |
| `--device`        | auto | cpu | cuda (default: auto)               |


---

## Report and visualize

```bash
sfb report                                    # Metrics grid (default filters)
sfb report --save metrics.png                  # Save to data/figures/
sfb report curve data/traces/copy_simple_nn_*.csv --save curve.png
sfb report noise --save noise.png              # Metric vs target_noise
sfb report seq-len --save seq_len.png          # Metric vs seq_len
sfb report vocab --save vocab.png              # Metric vs vocab_size
```

Filter by task/model/seq_len/vocab_size/target_noise; set `--x`, `--y`, `--metric`, `--facet-by` for grids. Example: `sfb report --x seq_len --y vocab_size --facet-by model --save grid.png`.

**Predict from a checkpoint:** `sfb predict data/checkpoints/copy_gru_0.pt -n 5`

---

## `sfb run` options (reference)


| Option                | Short | Description                                        |
| --------------------- | ----- | -------------------------------------------------- |
| `--task`              | `-t`  | Task: copy, sorting, reverse (required)            |
| `--model`             | `-m`  | Model (default: simple_nn)                         |
| `--loss`              | `-l`  | cross_entropy or shift_tolerant_ce (default: cross_entropy) |
| `--seq-len`           |       | Sequence length                                    |
| `--vocab-size`        |       | Vocabulary size                                    |
| `--target-noise`      |       | Label noise [0–1] during training                  |
| `--d-model`           |       | Model hidden dimension                             |
| `--steps`             | `-n`  | Training steps (default: 5000)                     |
| `--batch-size`        | `-b`  | Batch size                                         |
| `--eval-every`        |       | Eval every N steps (default: 1000)                 |
| `-o` / `--output`     |       | Path for step history (optional)                   |
| `--trace`             |       | Save step history to data/traces/                  |
| `--no-append-summary` |       | Do not append to results.csv                       |
| `--no-save-model`     |       | Do not save checkpoint (default: save)             |
| `-c` / `--config`     |       | YAML config path                                   |
| `--device`            |       | auto | cpu | cuda (default: auto)                  |
| `--seed`              |       | Random seed                                        |
| `--show-examples`     |       | Show N example predictions at end                  |


---

## Adding custom models and tasks

Models and tasks self-register via decorators. No CLI changes needed.

**Model** — create `src/sfb/models/my_model.py`:

```python
from sfb.registry import register_model
from sfb.models.base import BaseModel

@register_model(
    "my_model",
    display_params=["d_model"],
    constructor_params=["vocab_size", "seq_len", "d_model"],
    param_defaults={"d_model": 64},
)
class MyModel(BaseModel):
    # Implement forward(), reset_state()
    ...
```

**Task** — create `src/sfb/tasks/my_task.py`:

```python
from sfb.registry import register_task
from sfb.tasks.base import BaseTask

@register_task("my_task", description="what it does")
class MyTask(BaseTask):
    # Implement get_batch(), loss(), evaluate()
    ...
```

Then `sfb list --tasks` and `sfb list --models` show the new entries. Registry: **display_params** (shown in CSV), **constructor_params** (passed to `__init_`_), **param_defaults** (sweep/run defaults).
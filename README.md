![Build](https://github.com/emiliomunozai/SequenceFactorBenchmark/actions/workflows/build.yml/badge.svg)
![Tests](https://github.com/emiliomunozai/SequenceFactorBenchmark/actions/workflows/test.yml/badge.svg)
# SeqFactorBench

Sequence model benchmark for copy, sorting, and reverse tasks under controlled scale, breadth, structure, and noise.

---

## Setup Instructions

This project requires **Python 3.14.3** (see `pyproject.toml`).  
The required version is enforced via:

```toml
requires-python = "==3.14.3.*"
```

All dependencies and the virtual environment are managed using [`uv`](https://github.com/astral-sh/uv).

---

### 1. Using `uv` (Recommended)

#### Install `uv`

```bash
pip install uv
```

#### Create a virtual environment with Python 3.14

```bash
uv venv .venv --python 3.14
```

#### Activate the environment

- **Linux/macOS**
  ```bash
  source .venv/bin/activate
  ```
- **Windows (PowerShell)**
  ```powershell
  .venv\Scripts\activate
  ```

#### Install the project (CPU by default)

```bash
uv sync
```

For **running tests** (pytest, build):

```bash
uv sync --extra dev
```

Then run `uv run pytest` or `pytest`.

For **CUDA** (new GPU hardware, e.g. RTX 50 series):

```bash
uv sync
uv pip uninstall torch
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

This replaces the default CPU-only PyTorch with the CUDA 12.8 build. Use `cu124`, `cu121`, or `cu118` for older GPUs. Verify with:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

This automatically:
- Resolves dependencies from `pyproject.toml`
- Creates/updates `uv.lock`
- Installs the project in *editable* mode  
  (_No need to run `pip install -e .`_)

---

### 2. CLI Usage (`sfb`)

After running `uv sync`, the `sfb` CLI command is available.

**List tasks, models, and losses:**

```bash
sfb list               # Summary
sfb list --tasks       # Show available tasks
sfb list --models      # Show available models
sfb list --losses      # Show available loss functions
sfb list --config      # Show default config
```

**Show version:**

```bash
sfb version
```

**Run a benchmark:**

**Data layout** (folders under `data/`, relative to project root):
- `data/results/` — results table (`results.csv`), one row per experiment; `sfb run` and `sfb sweep` append here by default
- `data/traces/` — step-level history from `sfb run --trace` for learning curve plots
- `data/figures/` — saved plots; `sfb report --save name.png` writes to `data/figures/name.png` by default

```bash
# Sorting task — summary appended to data/results/results.csv
sfb run --task sorting --seq-len 32 --vocab-size 64 --steps 5000

# Save step history for learning curves
sfb run -t sorting --trace

# Reverse task with GRU
sfb run -t reverse -m gru --seq-len 32 --steps 5000

# Copy task
sfb run -t copy -b 128 --eval-every 200

# YAML config
sfb run -t sorting -c configs/generation_default.yaml --steps 2000
```

**Sweep and report:**

```bash
sfb sweep -c configs/run_sweep.yaml

# Visualize (metrics grid, learning curve, noise sensitivity)
sfb report
sfb report --save metrics.png                    # saves to data/figures/metrics.png
sfb report curve data/traces/sorting_simple_nn_*.csv --save curve.png   # plot trace, save to data/figures/

# Analyze the three dials: faceted heatmaps or line plots
sfb report --x seq_len --y vocab_size --facet-by target_noise --save heatmaps_by_noise.png
sfb report noise --save noise_sensitivity.png      # line plot: metric vs target_noise
sfb report seq-len --save seq_len_sensitivity.png  # line plot: metric vs seq_len
sfb report vocab --save vocab_sensitivity.png      # line plot: metric vs vocab_size
```

---

#### Common `sfb run` Options

| Option         | Short | Description                       |
| -------------- |-------|-----------------------------------|
| `--task`       | `-t`  | sorting, copy, reverse            |
| `--model`      | `-m`  | simple_nn or gru (default: simple_nn) |
| `--loss`       | `-l`  | Loss (default: cross_entropy)     |
| `--seq-len`    |       | Sequence length                   |
| `--vocab-size` |       | Vocabulary size                   |
| `--target-noise` |     | Label noise rate [0-1] during training (0 = none) |
| `--d-model`    |       | Model hidden dimension            |
| `--steps`      | `-n`  | Training steps                    |
| `--batch-size` | `-b`  | Batch size                        |
| `--eval-every` |       | Evaluation interval (steps)       |
| `--output`     | `-o`  | Save step history to path (optional) |
| `--trace`      |       | Save step history to data/traces/ |
| `--no-append-summary` | | Skip appending to data/results/results.csv |
| `--config`     | `-c`  | YAML config path                  |
| `--device`     |       | auto, cpu, or cuda                |
| `--seed`       |       | Random seed                       |
| `--show-examples` |   | Show N example predictions at end (e.g. `--show-examples` or `--show-examples 3`) |

#### Analyzing the three dials (seq_len, vocab_size, target_noise)

| Report mode | Command | Use case |
| ----------- | ------- | -------- |
| Metrics heatmap | `sfb report --x seq_len --y vocab_size --facet-by model` | 2D grid, one subplot per model |
| Heatmaps by noise | `sfb report --x seq_len --y vocab_size --facet-by target_noise` | Same grid, one subplot per noise level |
| Noise sensitivity | `sfb report noise --save noise.png` | Line plot: metric vs target_noise |
| Seq length sensitivity | `sfb report seq-len --save seq_len.png` | Line plot: metric vs seq_len |
| Vocab size sensitivity | `sfb report vocab --save vocab.png` | Line plot: metric vs vocab_size |
| Dial on axis | `sfb report --x target_noise --y task` | Any dial on x-axis (e.g. target_noise: 0, 0.1, 0.2) |

#### `sfb sweep` Options

| Option     | Short | Description                                    |
| ---------- |-------|------------------------------------------------|
| `--config` | `-c`  | Sweep YAML (list values = grid dimensions)     |
| `--output` | `-o`  | Results path (default: data/results/results.csv) |
| `--overwrite` |     | Replace results file instead of appending |
| `--device` |       | auto, cpu, or cuda                             |

**Device / CUDA:** The CLI uses `--device auto` by default (GPU if available). If you see "cpu (GPU not available)", replace torch with the CUDA build (see [Install the project](#install-the-project-cpu-by-default) above).

---

### 3. Adding Custom Models and Tasks (Registry)

Models and tasks self-register via decorators. **No CLI changes needed** when adding a new model or task.

**Add a model:** create `src/seqfacben/models/my_model.py`:

```python
from seqfacben.registry import register_model
from seqfacben.models.base import BaseModel

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

**Add a task:** create `src/seqfacben/tasks/my_task.py`:

```python
from seqfacben.registry import register_task
from seqfacben.tasks.base import BaseTask

@register_task("my_task", description="what it does")
class MyTask(BaseTask):
    # Implement get_batch(), loss(), evaluate()
    ...
```

Registry options:
- **display_params** – params shown in sweep CSV model column (e.g. `simple_nn(d_model=64)`)
- **constructor_params** – args passed to model `__init__`
- **param_defaults** – defaults for sweep/run (e.g. `n_layers=1` for GRU)

Run `sfb list --tasks` and `sfb list --models` to see registered items.

---
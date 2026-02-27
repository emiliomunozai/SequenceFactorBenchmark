![Tests](https://github.com/emiliomunozai/SequenceFactorBenchmark/actions/workflows/test.yml/badge.svg)
![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)

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

For **CUDA** (new GPU hardware, e.g. RTX 50 series):

```bash
uv sync
uv pip uninstall torch -y
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

#### (Optional) Install developer extras

```bash
uv sync --extra dev
```

#### (Optional) Parquet support for compact results

```bash
uv sync --extra parquet
```

Then use `-o data/results.parquet` to save as Parquet; default is CSV.

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

All experiment summaries go to `data/results.*` by default (same as sweep).

```bash
# Sorting task — summary appended to data/results.*
sfb run --task sorting --seq-len 32 --vocab-size 64 --steps 5000

# Reverse task with GRU (sequence-oriented)
sfb run -t reverse -m gru --seq-len 32 --steps 5000

# Copy task with custom batch size and eval interval
sfb run -t copy -b 128 --eval-every 200

# Use a YAML config file
sfb run -t sorting -c configs/generation_default.yaml --steps 2000
```

**Optional:** Use `-o path` only when you need the step-level learning curve for one run (e.g. for plotting). The summary still goes to `data/results.*`.

**Run a parameter sweep (many experiments from one config):**

```bash
# Sweep over task and sequence_length (lists in YAML → Cartesian product)
sfb sweep -c configs/run_sweep.yaml
```

Use list values in the YAML to sweep; scalars are fixed. Results are saved to `data/results.csv` by default.

**Results layout:**
- `data/results.csv` — single file with one row per experiment (config + final metrics). Both `sfb run` and `sfb sweep` append here by default.
- `-o path` (optional) — saves step-level history for one run (train_loss, train_acc per eval step) for plotting learning curves. Use only when needed.

---

#### Common `sfb run` Options

| Option         | Short | Description                       |
| -------------- |-------|-----------------------------------|
| `--task`       | `-t`  | sorting, copy, reverse            |
| `--model`      | `-m`  | simple_nn or gru (default: simple_nn) |
| `--loss`       | `-l`  | Loss (default: cross_entropy)     |
| `--seq-len`    |       | Sequence length                   |
| `--vocab-size` |       | Vocabulary size                   |
| `--d-model`    |       | Model hidden dimension            |
| `--steps`      | `-n`  | Training steps                    |
| `--batch-size` | `-b`  | Batch size                        |
| `--eval-every` |       | Evaluation interval (steps)       |
| `--output`     | `-o`  | Save step-level learning curve to path (optional, for plotting) |
| `--no-append-summary` | | Skip appending summary to data/results.* |
| `--config`     | `-c`  | YAML config path                  |
| `--device`     |       | auto, cpu, or cuda                |
| `--seed`       |       | Random seed                       |

#### `sfb sweep` Options

| Option     | Short | Description                                    |
| ---------- |-------|------------------------------------------------|
| `--config` | `-c`  | Sweep YAML (list values = grid dimensions)     |
| `--output` | `-o`  | Results path (default: data/results.csv) |
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
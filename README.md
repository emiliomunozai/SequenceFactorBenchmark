![ci](https://github.com/emiliomunozai/SequenceFactorBenchmark/actions/workflows/ci.yml/badge.svg)

# SeqFactorBench

Sequence model benchmark for copy, sorting, and reverse tasks under controlled scale, breadth, structure, and noise.

---

## Setup Instructions

This project requires **Python 3.12+** (see `pyproject.toml`). Dependencies and the virtual environment are managed using [`uv`](https://github.com/astral-sh/uv).

---

### 1. Using `uv` (Recommended)

#### Install `uv`

`uv` is used to create virtual envs, manage Python versions, manage dependencies, and replace pip/pip-tools/venv/pipx. Installing it via `pip` is slightly circular (using old Python tooling to install the tool that replaces it). Not wrong—just less clean.

**When each is appropriate**

| Use `pip install uv` when … | Use the PowerShell installer when … |
| ---------------------------- | ------------------------------------ |
| Python is already installed | You want the official recommended install |
| You want the fastest path | You want uv independent of Python |
| Corporate environment blocks script installers | You want cleaner machine bootstrap |
| You want zero external installer logic | You may later use `uv python install` |
| CI already has Python and just needs uv | You want fewer interpreter/path edge cases. **Best for fresh machines.** |

**Fast pragmatic install** (Python already on the machine):

```powershell
py -m pip install --user uv
```

**Cleaner long-term install** (standalone; recommended for fresh machines):

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Why the PATH step exists (Windows):** The installer typically puts `uv` in `C:\Users\<user>\.local\bin`. If that directory is not on PATH, Windows won’t find `uv`. Add it once if needed:

```powershell
[Environment]::SetEnvironmentVariable(
  "Path",
  $env:Path + ";C:\Users\$env:USERNAME\.local\bin",
  [EnvironmentVariableTarget]::User
)
```

Then open a new terminal so `uv` is callable globally.

**Summary:** Need quick and simple → `py -m pip install --user uv`. Need proper toolchain / fresh machine / best practice → use the [Astral PowerShell installer](https://astral.sh/uv/install.ps1), not pip.

#### Dependency structure

Dependencies are split so the environment is reproducible and torch stays optional and machine-specific:

| Layer | Purpose |
| ----- | -------- |
| **Core** | Universal runtime: PyYAML, tqdm, spacy, pandas, matplotlib. No torch. |
| **Dev group** | Tooling only (not needed at benchmark runtime): build, pytest, ruff, ipykernel. |
| **GPU group** | PyTorch (CUDA build). Heavy and environment-specific; kept out of core so CPU-only setups don’t pull it. |

That gives:

- **CPU/basic setup** — core + dev tooling, no torch. Use when you only run tests/lint.
- **GPU setup** — core + dev + torch (CUDA). Use when you run benchmarks (CLI uses GPU if available; the same torch build also runs on CPU).

#### Best command flow

**Local dev without GPU** (tests, lint, no benchmark runs):

```bash
uv sync --group dev
```

**Local dev with GPU** (run benchmarks; torch from CUDA index):

```bash
uv sync --group dev --group gpu
```

Then run the CLI with `uv run sfb ...`; no need to activate the venv. To confirm PyTorch sees the GPU:

```bash
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

`uv sync` (with or without groups) resolves from `pyproject.toml`, updates `uv.lock`, and installs the project in editable mode.

---

### 2. CLI Usage (`sfb`)

After running `uv sync --group dev --group gpu` (or at least `--group gpu` so torch is installed), the `sfb` CLI is available. Either **activate the venv** (e.g. `.venv\Scripts\activate` on Windows) and run `sfb ...`, or run **`uv run sfb ...`** without activating—both use the project environment.

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

**Device / CUDA:** The CLI uses `--device auto` by default (GPU if available). To use GPU, run `uv sync --group dev --group gpu`.

---

### 3. Adding Custom Models and Tasks (Registry)

Models and tasks self-register via decorators. **No CLI changes needed** when adding a new model or task.

**Add a model:** create `src/sfb/models/my_model.py`:

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

**Add a task:** create `src/sfb/tasks/my_task.py`:

```python
from sfb.registry import register_task
from sfb.tasks.base import BaseTask

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
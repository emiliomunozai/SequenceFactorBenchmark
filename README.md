![Tests](https://github.com/emiliomunozai/SequenceFactorBenchmark/actions/workflows/test.yml/badge.svg)
![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)

# SeqFactorBench

Sequence model benchmark for copy and sorting tasks under controlled scale, breadth, structure, and noise.

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

```bash
# Sorting task, 5000 steps
sfb run --task sorting --seq-len 32 --vocab-size 64 --steps 5000 --output results.csv

# Copy task with custom batch size and custom eval interval
sfb run -t copy -b 128 --eval-every 200 -o copy_run.csv

# Use a YAML config file
sfb run -t sorting -c configs/generation_default.yaml --steps 2000 -o out.csv
```

**Run a parameter sweep (many experiments from one config):**

```bash
# Sweep over task and sequence_length (lists in YAML → Cartesian product)
sfb sweep -c configs/run_sweep.yaml -o data/sweep_results.csv
```

Use list values in the YAML to sweep; scalars are fixed. Output is saved to `data/sweep_results.csv` by default (or the path given by `-o`).

**Incremental sweeps:** If the output CSV already exists, the sweep loads it, skips configurations that were already run, and appends only new results. You can safely re-run a sweep (e.g., after adding parameters to the YAML) without re-running completed experiments.

---

#### Common `sfb run` Options

| Option         | Short | Description                       |
| -------------- |-------|-----------------------------------|
| `--task`       | `-t`  | sorting or copy                   |
| `--model`      | `-m`  | Model (default: simple_nn)        |
| `--loss`       | `-l`  | Loss (default: cross_entropy)     |
| `--seq-len`    |       | Sequence length                   |
| `--vocab-size` |       | Vocabulary size                   |
| `--d-model`    |       | Model hidden dimension            |
| `--steps`      | `-n`  | Training steps                    |
| `--batch-size` | `-b`  | Batch size                        |
| `--eval-every` |       | Evaluation interval (steps)       |
| `--output`     | `-o`  | CSV output path                   |
| `--config`     | `-c`  | YAML config path                  |
| `--device`     |       | auto, cpu, or cuda                |
| `--seed`       |       | Random seed                       |

#### `sfb sweep` Options

| Option     | Short | Description                                    |
| ---------- |-------|------------------------------------------------|
| `--config` | `-c`  | Sweep YAML (list values = grid dimensions)     |
| `--output` | `-o`  | Summary CSV (default: data/sweep_results.csv)  |
| `--device` |       | auto, cpu, or cuda                             |

**Device / CUDA:** The CLI uses `--device auto` by default (GPU if available). If you see "cpu (GPU not available)", replace torch with the CUDA build (see [Install the project](#install-the-project-cpu-by-default) above).

---
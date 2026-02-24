# AGENT.md — SeqFactorBench (`sfb`)

> This file is the authoritative context document for any agent working on the `sfb` project.
> Read it fully before taking any action. Follow every constraint exactly.

---

## 1. What This Project Is

**SeqFactorBench** is a Python benchmarking framework for sequence models.
It tests models on controlled tasks (copy, sorting) across combinations of sequence length, vocabulary size, model dimension, and noise.
The CLI tool is `sfb`. The Python package lives under `src/seqfacben/`.

---

## 2. Hard Constraints — Never Violate These

- **Python version is `3.14.3` exactly.** Do not use system Python, `pyenv`, `conda`, or `pip` directly. Only `uv`.
- **All environment operations must go through `uv`.** No bare `pip install` commands.
- **Never edit `uv.lock` by hand.** Let `uv sync` manage it.
- **Never install packages globally.** Always work inside `.venv`.
- **Source code lives in `src/seqfacben/`.** Do not create files outside this tree unless they are configs, tests, or data outputs.
- **Tests live in `tests/`.** Run them with `pytest` after any code change.
- **Output CSVs go to `data/`.** Do not scatter output files in the project root.

---

## 3. Environment Setup (Run Once)

```bash
pip install uv                                  # install uv itself
uv venv .venv --python 3.14                     # create venv, exact Python version
source .venv/bin/activate                       # Linux/macOS
# .venv\Scripts\activate                        # Windows PowerShell

uv sync                                         # install all deps (CPU PyTorch by default)
uv sync --extra dev                             # also install pytest, build tools
```

**CUDA (GPU) setup — only if CUDA is available:**

```bash
uv sync
uv pip uninstall torch -y
uv pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 40/50 series
# cu124 / cu121 / cu118 for older GPUs

# Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**Confirm the environment is healthy:**

```bash
sfb version
sfb list
```

If `sfb` is not found, the venv is not activated or `uv sync` was not run.

---

## 4. Project Layout

```
seqfacbench/
├── src/
│   └── seqfacben/          ← all source code lives here
│       ├── cli.py           ← sfb CLI entrypoint (do not rename)
│       ├── tasks/           ← task definitions
│       │   └── __init__.py  ← TASK_REGISTRY dict
│       ├── models/          ← model definitions
│       │   └── __init__.py  ← MODEL_REGISTRY dict
│       └── losses/          ← loss function definitions
│           └── __init__.py  ← LOSS_REGISTRY dict
├── configs/                 ← YAML configs for runs and sweeps
├── data/                    ← output CSVs (do not commit large files)
├── tests/                   ← pytest tests
├── pyproject.toml           ← single source of truth for deps and metadata
└── uv.lock                  ← auto-managed, never edit manually
```

---

## 5. Registries — How Components Connect

Every model, task, and loss must be **registered** to be usable from the CLI.

### Adding a Model

1. Create `src/seqfacben/models/my_model.py` with a `nn.Module` subclass.
2. Open `src/seqfacben/models/__init__.py` and add it to `MODEL_REGISTRY`:

```python
from .my_model import MyModel

MODEL_REGISTRY = {
    "simple_nn": SimpleNN,
    "my_model":  MyModel,    # ← new entry
}
```

3. The model constructor **must accept** `vocab_size`, `seq_len`, `d_model` as keyword arguments.
4. `forward(self, x)` must accept `x` of shape `(batch, seq_len)` and return logits of shape `(batch, seq_len, vocab_size)`.

### Adding a Task

1. Create `src/seqfacben/tasks/my_task.py`.
2. The class must implement `sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]` returning `(input_ids, target_ids)` as `LongTensor` of shape `(batch, seq_len)`.
3. Register in `src/seqfacben/tasks/__init__.py` under `TASK_REGISTRY`.

### Adding a Loss

1. Create `src/seqfacben/losses/my_loss.py` with a `nn.Module` subclass.
2. `forward(logits, targets)` must accept `(B, L, V)` logits and `(B, L)` targets, return a scalar.
3. Register in `src/seqfacben/losses/__init__.py` under `LOSS_REGISTRY`.

---

## 6. CLI Reference

### Inspect what's registered

```bash
sfb list               # everything
sfb list --tasks
sfb list --models
sfb list --losses
sfb list --config      # default config values
sfb version
```

### Single run

```bash
sfb run \
  --task sorting \
  --model simple_nn \
  --loss cross_entropy \
  --seq-len 32 \
  --vocab-size 64 \
  --d-model 128 \
  --steps 5000 \
  --batch-size 64 \
  --eval-every 500 \
  --seed 42 \
  --device auto \
  -o data/run_001.csv
```

All flags are optional — unset flags use defaults from `sfb list --config`.

### YAML config run

```bash
sfb run -c configs/my_experiment.yaml -o data/run_001.csv
```

YAML keys mirror CLI flags exactly (use underscores, e.g. `seq_len` not `seq-len`).

### Parameter sweep

```bash
sfb sweep -c configs/sweep.yaml -o data/sweep_results.csv
```

In the sweep YAML, **list values are swept axes** (Cartesian product), **scalars are fixed**.

```yaml
# configs/sweep.yaml example
task: [copy, sorting]
seq_len: [16, 32, 64]
vocab_size: 64           # fixed
d_model: [128, 256]
steps: 5000
seed: 0
```

This produces `2 × 3 × 2 = 12` runs.

### Device flag

| Value | Behaviour |
|-------|-----------|
| `auto` | GPU if available, else CPU (default) |
| `cpu` | Force CPU |
| `cuda` | Force GPU (fails if unavailable) |

---

## 7. Config Files

Store all configs under `configs/`. Name them descriptively.

```yaml
# configs/example.yaml
task: sorting
model: simple_nn
loss: cross_entropy
seq_len: 32
vocab_size: 64
d_model: 128
steps: 5000
batch_size: 64
eval_every: 500
seed: 42
device: auto
```

---

## 8. Output Format

All runs write a CSV to the path given by `-o`. Each row is one evaluation checkpoint.
Expected columns (confirm with `sfb list --config`):

| Column | Description |
|--------|-------------|
| `step` | Training step |
| `loss` | Training loss at checkpoint |
| `accuracy` | Token-level accuracy on eval set |
| `task` | Task name |
| `model` | Model name |
| `seq_len` | Sequence length |
| `vocab_size` | Vocabulary size |
| `d_model` | Model hidden dim |
| `seed` | Random seed |

Sweep runs append all experiments into a single CSV with the same schema.

---

## 9. Dependencies

Managed entirely via `pyproject.toml`. To add a dependency:

```bash
# Add to [project.dependencies] in pyproject.toml, then:
uv sync
```

Never run `uv pip install <pkg>` for project dependencies — it won't be tracked in `pyproject.toml`.
Only use `uv pip install` for one-off tools (like swapping the torch build for CUDA).

Current dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | `>=2.0` | Model training |
| `pandas` | `>=3.0.0` | CSV output |
| `tqdm` | `4.67.1` | Progress bars |
| `PyYAML` | `>=6.0` | Config parsing |
| `spacy` | `3.8.11` | NLP preprocessing (if used) |

Dev extras (`uv sync --extra dev`): `pytest>=7.0`, `build>=1.0`

---

## 10. Testing

```bash
pytest                              # run all tests
pytest tests/test_models.py -v      # single file, verbose
pytest -k "test_sorting" -v         # filter by name
```

Run tests after every code change. Do not commit if tests fail.

---

## 11. Common Agent Tasks & How to Execute Them

### "Add a new model"
1. Write the class in `src/seqfacben/models/<name>.py`.
2. Register it in `src/seqfacben/models/__init__.py`.
3. Verify: `sfb list --models` shows the new name.
4. Run: `sfb run --model <name> --task copy --steps 100 -o data/smoke.csv`
5. Run: `pytest`

### "Run an experiment"
1. Write or reuse a YAML config in `configs/`.
2. Run: `sfb run -c configs/<file>.yaml -o data/<name>.csv`
3. Inspect results: `python -c "import pandas as pd; print(pd.read_csv('data/<name>.csv').tail())"`

### "Run a sweep"
1. Create a sweep YAML with list values for the axes to search.
2. Run: `sfb sweep -c configs/<sweep>.yaml -o data/sweep_results.csv`
3. Analyze: load `data/sweep_results.csv` with pandas and group by the swept axes.

### "Add a dependency"
1. Add the package to `[project.dependencies]` in `pyproject.toml`.
2. Run: `uv sync`
3. Confirm: `python -c "import <pkg>"`

### "Debug a failing run"
1. Re-run with `--steps 50 --eval-every 10` to get fast feedback.
2. Check `sfb list --config` to confirm defaults are what you expect.
3. Add `--device cpu` to rule out GPU issues.
4. Print model output shapes in `forward()` to catch shape mismatches.

---

## 12. What NOT to Do

- Do not run `python setup.py` or `pip install -e .` — `uv sync` handles editable install.
- Do not hardcode paths. Use `pathlib.Path` relative to the project root.
- Do not commit `data/*.csv` files with large sweep results.
- Do not modify `uv.lock` manually.
- Do not create a second `requirements.txt` — `pyproject.toml` is the only dependency file.
- Do not use `print` for logging inside library code — use Python's `logging` module.
- Do not change the `sfb` entrypoint path (`seqfacben.cli:main`) without updating `pyproject.toml`.
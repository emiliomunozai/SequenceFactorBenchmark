
## Setup Instructions

This project requires **Python 3.14.3** (see `pyproject.toml` and `environment.yml`). Please ensure you are using the correct Python version for reproducibility.

You can set up your environment using either Python virtual environments (`venv`/`uv`) or Conda.

---

### 1. Using Python venv (with [uv](https://github.com/astral-sh/uv) or `venv`)

```bash
pip install uv  # (optional) install uv for fast environment setup
uv venv .venv --python 3.14  # or: python3.14 -m venv .venv
source .venv/bin/activate
pip install -e .  # install the project and dependencies in editable mode
```

If you require additional developer dependencies:
```bash
pip install .[dev]
```

---

### 2. Using Conda

```bash
conda env create -f environment.yml
conda activate seqfacben
```

This will install Python 3.14.3 and all main dependencies.

---

**Notes:**
- Package dependencies are managed in `pyproject.toml` (PEP 621). Editable install (`pip install -e .`) is recommended for development.
- To install optional developer tools (e.g., testing and building), use the `[dev]` extra as shown above.
- Python 3.14.3 is required (`requires-python = "==3.14.3.*"`).

---

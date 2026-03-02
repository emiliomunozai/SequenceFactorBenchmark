"""
Results and data paths. All CSV.
"""
from pathlib import Path


def _project_root() -> Path:
    """Project root (dir containing pyproject.toml); fallback to cwd when not found."""
    cur = Path(__file__).resolve().parent
    for _ in range(5):
        if (cur / "pyproject.toml").exists():
            return cur
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
    return Path.cwd()


# data/
#   results/   – results table (appending summary rows)
#   traces/    – step-level history (sfb run --trace)
#   figures/   – saved plots from sfb report --save
DATA_DIR = _project_root() / "data"
RESULTS_DIR = DATA_DIR / "results"
TRACES_DIR = DATA_DIR / "traces"
FIGURES_DIR = DATA_DIR / "figures"

RESULTS_CSV = RESULTS_DIR / "results.csv"
_LEGACY_RESULTS = DATA_DIR / "results.csv"


def default_results_path() -> Path:
    """Path for the results summary table."""
    if not RESULTS_CSV.exists() and _LEGACY_RESULTS.exists():
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        _LEGACY_RESULTS.replace(RESULTS_CSV)  # migrate legacy data/results.csv
    return RESULTS_CSV


def traces_dir() -> Path:
    """Directory for step-level trace CSVs."""
    return TRACES_DIR


def figures_dir() -> Path:
    """Directory for saved analysis plots."""
    return FIGURES_DIR


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    import pandas as pd
    df = pd.read_csv(path)
    rows = df.to_dict("records")
    for row in rows:
        if "model" not in row or (isinstance(row.get("model"), float) and pd.isna(row.get("model"))):
            row["model"] = "simple_nn"
        if "n_layers" not in row or (isinstance(row.get("n_layers"), float) and pd.isna(row.get("n_layers"))):
            row["n_layers"] = 1
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, index=False)


def append_results(rows: list[dict], path: Path | None = None) -> Path:
    """Append rows to results CSV. Returns path written to."""
    path = Path(path) if path else default_results_path()
    existing = _read_csv(path)
    _write_csv(path, existing + rows)
    return path


def overwrite_results(rows: list[dict], path: Path | None = None) -> Path:
    """Overwrite results CSV. Returns path written to."""
    path = Path(path) if path else default_results_path()
    _write_csv(path, rows)
    return path


def load_existing_rows(path: Path, config_signature_fn) -> tuple[list[dict], set[tuple]]:
    """Load existing rows and their config signatures."""
    if not path.exists():
        return [], set()
    rows = _read_csv(path)
    return rows, {config_signature_fn(r) for r in rows}


def get_next_run_id(path: Path | None = None) -> int:
    """Next run_id (max + 1, or 0)."""
    path = Path(path) if path else default_results_path()
    if not path.exists():
        return 0
    import pandas as pd
    df = pd.read_csv(path)
    if "run_id" not in df.columns or len(df) == 0:
        return 0
    return int(df["run_id"].max()) + 1

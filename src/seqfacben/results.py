"""
Unified results handling: append/overwrite. Default format is CSV.
All experiment summaries go to data/results.csv by default.
"""
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("data")
RESULTS_FILE = "results"
DEFAULT_RESULTS_PATH = RESULTS_DIR / "results"


def default_results_path() -> Path:
    """Default path for main results file. Uses CSV. Prefers existing file."""
    base = Path(DEFAULT_RESULTS_PATH)
    csv_path = base.with_suffix(".csv")
    parquet_path = base.with_suffix(".parquet")
    if csv_path.exists():
        return csv_path
    if parquet_path.exists():
        return parquet_path  # Keep using existing parquet if present
    return csv_path


def _read_results(path: Path) -> list[dict]:
    """Load existing results. Backfills missing columns for older formats."""
    if not path.exists():
        return []
    try:
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        rows = df.to_dict("records")
        for row in rows:
            if "model" not in row or (isinstance(row.get("model"), float) and pd.isna(row.get("model"))):
                row["model"] = "simple_nn"
            if "n_layers" not in row or (isinstance(row.get("n_layers"), float) and pd.isna(row.get("n_layers"))):
                row["n_layers"] = 1
        return rows
    except Exception as e:
        raise IOError(f"Could not load results from {path}: {e}") from e


def _write_results(path: Path, rows: list[dict]) -> None:
    """Write results to path. Creates parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def append_results(rows: list[dict], path: Path | None = None) -> Path:
    """Append rows to results file. Returns path written to."""
    path = Path(path) if path else default_results_path()
    existing = _read_results(path) if path.exists() else []
    all_rows = existing + rows
    _write_results(path, all_rows)
    return path


def overwrite_results(rows: list[dict], path: Path | None = None) -> Path:
    """Overwrite results file with rows. Returns path written to."""
    path = Path(path) if path else default_results_path()
    _write_results(path, rows)
    return path


def load_existing_rows(path: Path, config_signature_fn) -> tuple[list[dict], set[tuple]]:
    """Load existing rows and explored signatures. config_signature_fn(row) -> tuple."""
    if not path.exists():
        return [], set()
    rows = _read_results(path)
    signatures = {config_signature_fn(r) for r in rows}
    return rows, signatures


def get_next_run_id(path: Path | None = None) -> int:
    """Next run_id when appending (max existing + 1, or 0)."""
    path = Path(path) if path else default_results_path()
    if not path.exists():
        return 0
    try:
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        if "run_id" not in df.columns or len(df) == 0:
            return 0
        return int(df["run_id"].max()) + 1
    except Exception:
        return 0

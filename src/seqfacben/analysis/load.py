"""
Load and filter benchmark results.
"""
from pathlib import Path
from typing import Any

import pandas as pd

from seqfacben.results import default_results_path

CONFIG_COLS = ["task", "loss", "seq_len", "vocab_size", "d_model", "model"]


def load_results(path: str | Path | None = None) -> pd.DataFrame:
    """Load results from CSV. Default: data/results/results.csv"""
    p = Path(path) if path else default_results_path()
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def last_row_per_config(df: pd.DataFrame) -> pd.DataFrame:
    """Keep last row per unique config."""
    if df.empty or len(df) == 1:
        return df
    cols = [c for c in CONFIG_COLS if c in df.columns]
    if not cols:
        return df
    if "run_id" in df.columns:
        df = df.sort_values("run_id")
    return df.groupby(cols, dropna=False).last().reset_index()


def filter_results(
    df: pd.DataFrame,
    *,
    task: str | list[str] | None = None,
    model: str | list[str] | None = None,
    seq_len: int | list[int] | None = None,
    vocab_size: int | list[int] | None = None,
    d_model: int | list[int] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Filter by config. Model uses prefix match."""
    out = df.copy()
    for k, v in [
        ("task", task),
        ("model", model),
        ("seq_len", seq_len),
        ("vocab_size", vocab_size),
        ("d_model", d_model),
        *kwargs.items(),
    ]:
        if v is None or k not in out.columns:
            continue
        if isinstance(v, list):
            mask = (
                out[k].astype(str).apply(lambda x: any(str(x).startswith(str(m)) or x == m for m in v))
                if k == "model"
                else out[k].isin(v)
            )
        else:
            mask = (
                out[k].astype(str).str.startswith(str(v)) | (out[k].astype(str) == str(v))
                if k == "model"
                else out[k] == v
            )
        out = out[mask]
    return out.reset_index(drop=True)

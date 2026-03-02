"""
Visualize benchmark results and learning curves.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from seqfacben.analysis.load import last_row_per_config


def plot_metrics_grid(
    df: pd.DataFrame,
    x: str = "seq_len",
    y: str = "task",
    metric: str = "final_val_acc",
    facet_by: str | None = "model",
    figsize: tuple[float, float] | None = None,
    cmap: str = "RdYlGn",
    save: str | Path | None = None,
) -> plt.Figure | None:
    """Plot metrics as heatmap. facet_by splits into subplots."""
    if df.empty or metric not in df.columns:
        return None

    df = last_row_per_config(df)
    agg_df = df.groupby([y, x], as_index=False)[metric].mean()
    pivot = agg_df.pivot(index=y, columns=x, values=metric)

    if facet_by and facet_by in df.columns and df[facet_by].nunique() > 1:
        facets = sorted(df[facet_by].dropna().unique(), key=str)
        n_facets = len(facets)
        ncols = min(2, n_facets)
        nrows = (n_facets + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (5 * ncols, 4 * nrows))
        axarr = np.atleast_1d(axes).flatten()
        vmin = 0 if "acc" in metric.lower() else None
        vmax = 1 if "acc" in metric.lower() else None
        for i, facet_val in enumerate(facets):
            ax = axarr[i]
            sub = df[df[facet_by] == facet_val]
            pvt = sub.groupby([y, x], as_index=False)[metric].mean().pivot(index=y, columns=x, values=metric)
            im = ax.imshow(pvt.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(pvt.columns)))
            ax.set_xticklabels(pvt.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(pvt.index)))
            ax.set_yticklabels(pvt.index)
            ax.set_title(f"{facet_by}={facet_val}", fontsize=10)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            plt.colorbar(im, ax=ax, label=metric)
        for j in range(n_facets, len(axarr)):
            axarr[j].set_visible(False)
    else:
        fig, ax = plt.subplots(figsize=figsize or (8, 5))
        vmin = 0 if "acc" in metric.lower() else None
        vmax = 1 if "acc" in metric.lower() else None
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(metric)
        plt.colorbar(im, ax=ax, label=metric)

    fig.tight_layout()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_learning_curve(
    path: str | Path,
    figsize: tuple[float, float] | None = (10, 4),
    save: str | Path | None = None,
) -> plt.Figure | None:
    """Plot learning curve from step-level trace CSV."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")
    df = pd.read_csv(p)
    required = {"step", "train_loss", "val_loss"}
    if not required.issubset(df.columns):
        raise ValueError(f"Trace must have columns: {required}")

    fig, axes = plt.subplots(1, 2, figsize=figsize or (10, 4))
    ax_loss, ax_acc = axes

    ax_loss.plot(df["step"], df["train_loss"], label="train", color="C0")
    ax_loss.plot(df["step"], df["val_loss"], label="val", color="C1")
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Learning curve: Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    if "train_acc" in df.columns and "val_acc" in df.columns:
        ax_acc.plot(df["step"], df["train_acc"], label="train", color="C0")
        ax_acc.plot(df["step"], df["val_acc"], label="val", color="C1")
    ax_acc.set_xlabel("Step")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Learning curve: Accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    fig.suptitle(p.name, fontsize=10, y=1.02)
    fig.tight_layout()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


_DIAL_COLS = ["seq_len", "vocab_size", "target_noise", "task", "model"]


def _plot_sensitivity_impl(
    df: pd.DataFrame,
    x_col: str,
    metric: str,
    facet_by: str | None,
    figsize: tuple[float, float] | None,
    save: str | Path | None,
) -> plt.Figure | None:
    """Line plot: metric vs x_col. One line per combo of other varying dims."""
    if df.empty or x_col not in df.columns or metric not in df.columns:
        return None

    df = last_row_per_config(df)
    if df[x_col].nunique() < 2:
        return None

    line_candidates = [c for c in _DIAL_COLS if c != x_col]
    x_label = x_col.replace("_", " ")

    if facet_by and facet_by in df.columns and df[facet_by].nunique() > 1:
        facets = sorted(df[facet_by].dropna().unique(), key=str)
        n_facets = len(facets)
        ncols = min(2, n_facets)
        nrows = (n_facets + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (5 * ncols, 4 * nrows))
        axarr = np.atleast_1d(axes).flatten()
        for i, facet_val in enumerate(facets):
            ax = axarr[i]
            sub = df[df[facet_by] == facet_val]
            line_cols = [c for c in line_candidates if c in sub.columns and sub[c].nunique() > 1]
            for _, grp in sub.groupby(line_cols, dropna=False):
                agg = grp.sort_values(x_col).groupby(x_col, as_index=False)[metric].mean()
                if len(agg) < 2:
                    continue
                label = "_".join(str(grp[c].iloc[0]) for c in line_cols) if line_cols else str(facet_val)
                ax.plot(agg[x_col], agg[metric], "-o", label=label, markersize=4)
            ax.set_xlabel(x_label)
            ax.set_ylabel(metric)
            ax.set_title(f"{facet_by}={facet_val}")
            ax.legend(loc="best", fontsize=7)
            ax.grid(True, alpha=0.3)
            if "acc" in metric.lower():
                ax.set_ylim(0, 1)
        for j in range(n_facets, len(axarr)):
            axarr[j].set_visible(False)
    else:
        fig, ax = plt.subplots(figsize=figsize or (8, 5))
        line_cols = [c for c in line_candidates if c in df.columns and df[c].nunique() > 1]
        for keys, grp in df.groupby(line_cols or ["task"], dropna=False):
            agg = grp.sort_values(x_col).groupby(x_col, as_index=False)[metric].mean()
            if len(agg) < 2:
                continue
            label = "_".join(str(k) for k in (keys if isinstance(keys, tuple) else [keys]))[:30]
            ax.plot(agg[x_col], agg[metric], "-o", label=label, markersize=4)
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric)
        ax.set_title(f"Metric vs {x_label}")
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)
        if "acc" in metric.lower():
            ax.set_ylim(0, 1)

    fig.tight_layout()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_sensitivity(
    df: pd.DataFrame,
    x_col: str = "target_noise",
    metric: str = "final_val_acc",
    facet_by: str | None = "task",
    figsize: tuple[float, float] | None = None,
    save: str | Path | None = None,
) -> plt.Figure | None:
    """Line plot: metric vs x_col (seq_len, vocab_size, or target_noise)."""
    return _plot_sensitivity_impl(df, x_col, metric, facet_by, figsize, save)


def plot_noise_sensitivity(
    df: pd.DataFrame,
    metric: str = "final_val_acc",
    facet_by: str | None = "task",
    figsize: tuple[float, float] | None = None,
    save: str | Path | None = None,
) -> plt.Figure | None:
    """Line plot: metric vs target_noise (backward compatible)."""
    return _plot_sensitivity_impl(df, "target_noise", metric, facet_by, figsize, save)

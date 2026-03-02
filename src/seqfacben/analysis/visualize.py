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

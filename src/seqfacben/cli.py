"""
CLI for SeqFactorBench. Invoke as: sfb run ... | sfb sweep ... | sfb report ...
"""
import argparse
import csv
import itertools
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from importlib.metadata import version, PackageNotFoundError

import pandas as pd
import torch

from seqfacben.generators.random import RandomSequenceGenerator
from seqfacben.results import (
    default_results_path,
    append_results,
    overwrite_results,
    load_existing_rows,
    get_next_run_id,
    traces_dir,
    figures_dir,
)
from seqfacben.registry import (
    get_model,
    get_task,
    all_model_names,
    all_task_names,
    all_model_param_keys,
    param_defaults_from_models,
)
from seqfacben.manager.task_manager import TaskManager
from seqfacben.losses import cross_entropy

LOSSES = {"cross_entropy": cross_entropy}


def _load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


# Keys that can be swept (list = dimension) or fixed (scalar). Used by sweep command.
# model_params.* are flattened into the config (e.g. model_params.d_model -> d_model).
SWEEP_PARAM_KEYS = [
    "model", "task", "loss", "sequence_length", "vocabulary_size", "target_noise",
    *sorted(all_model_param_keys()),
    "batch_size", "steps", "eval_every", "seed",
]
SWEEP_DEFAULTS = {
    "model": "simple_nn",
    "task": "sorting",
    "loss": "cross_entropy",
    "sequence_length": 32,
    "vocabulary_size": 64,
    "target_noise": 0.0,
    "batch_size": 64,
    "steps": 5000,
    "eval_every": 1000,
    "seed": None,
    **param_defaults_from_models(),
}


def _flatten_model_params(config: dict) -> dict:
    """Merge model_params into flat config. model_params.d_model -> d_model."""
    cfg = dict(config)
    if "model_params" in cfg and isinstance(cfg["model_params"], dict):
        params = cfg.pop("model_params")
        for k, v in params.items():
            cfg[k] = v
    return cfg


def _expand_sweep_config(config: dict) -> list[dict]:
    """Expand a config into a list of run configs. List values → Cartesian product."""
    cfg = _flatten_model_params(config)
    fixed = {}
    dims = {}
    for k in SWEEP_PARAM_KEYS:
        v = cfg.get(k, SWEEP_DEFAULTS.get(k))
        if isinstance(v, list):
            dims[k] = v
        else:
            fixed[k] = v
    if not dims:
        return [fixed]
    keys = list(dims.keys())
    return [
        {**fixed, **dict(zip(keys, combo))}
        for combo in itertools.product(*(dims[k] for k in keys))
    ]


def _base_model_name(model_val) -> str:
    """Extract base model name from display string like 'simple_nn(d_model=64)'."""
    s = str(model_val or "simple_nn")
    return s.split("(")[0].strip() if "(" in s else s


def _format_model_column(run_config: dict) -> str:
    """Build model column value: base name + params, e.g. 'simple_nn(d_model=64)'."""
    model = run_config.get("model", "simple_nn")
    base = _base_model_name(model)
    model_info = get_model(base)
    param_names = model_info["display_params"] if model_info else []
    if not param_names:
        return base
    parts = [f"{k}={run_config.get(k)}" for k in param_names if run_config.get(k) is not None]
    if not parts:
        return base
    return f"{base}({', '.join(parts)})"


def _config_signature(cfg: dict) -> tuple:
    """Hashable signature for deduplication. cfg can be run_config (sequence_length/vocabulary_size) or row (seq_len/vocab_size)."""
    model_val = cfg.get("model", "simple_nn")
    model = _base_model_name(model_val)
    seq_len = cfg.get("sequence_length") or cfg.get("seq_len")
    vocab_size = cfg.get("vocabulary_size") or cfg.get("vocab_size")
    seed = cfg.get("seed")
    if seed is None or (isinstance(seed, float) and pd.isna(seed)):
        seed = None
    elif seed is not None:
        seed = int(seed)
    target_noise = cfg.get("target_noise")
    if target_noise is not None and not (isinstance(target_noise, float) and pd.isna(target_noise)):
        target_noise = float(target_noise)
    else:
        target_noise = None
    return (
        model,
        str(cfg.get("task", "")),
        str(cfg.get("loss", "")),
        int(seq_len) if seq_len is not None else None,
        int(vocab_size) if vocab_size is not None else None,
        int(cfg.get("d_model")) if cfg.get("d_model") is not None else None,
        int(cfg.get("n_layers")) if cfg.get("n_layers") is not None and not (isinstance(cfg.get("n_layers"), float) and pd.isna(cfg.get("n_layers"))) else 1,
        int(cfg.get("batch_size")) if cfg.get("batch_size") is not None else None,
        int(cfg.get("steps")) if cfg.get("steps") is not None else None,
        int(cfg.get("eval_every")) if cfg.get("eval_every") is not None else None,
        target_noise,
        seed,
    )


def _get_device(name: str) -> torch.device:
    if name == "auto":
        use_cuda = torch.cuda.is_available()
        dev = torch.device("cuda" if use_cuda else "cpu")
        msg = "cuda (GPU)" if use_cuda else "cpu (GPU not available)"
        print(f"Using device: {msg}")
        return dev
    print(f"Using device: {name}")
    return torch.device(name)


def _build_task_and_model(args, config: dict):
    """Resolve seq_len, vocab_size, model from args + config. Build generator, task, model."""
    seq_len = getattr(args, "seq_len", None) or config.get("sequence_length", 32)
    vocab_size = getattr(args, "vocab_size", None) or config.get("vocabulary_size", 64)
    d_model = getattr(args, "d_model", None) or config.get("d_model", 64)
    model_name = (getattr(args, "model", None) or config.get("model", "simple_nn")).lower()

    loss_name = (getattr(args, "loss", None) or config.get("loss", "cross_entropy")).lower()
    loss_fn = LOSSES.get(loss_name)
    if loss_fn is None:
        raise SystemExit(f"Unknown loss: {loss_name}. Use: {', '.join(LOSSES)}.")

    model_info = get_model(model_name)
    if model_info is None:
        raise SystemExit(f"Unknown model: {model_name}. Use: {', '.join(all_model_names())}.")

    generator = RandomSequenceGenerator(seq_len=seq_len, vocab_size=vocab_size)
    task_name = (getattr(args, "task", None) or "").lower()
    task_info = get_task(task_name)
    if task_info is None:
        raise SystemExit(f"Unknown task: {args.task}. Use: {', '.join(all_task_names())}.")

    task = task_info["cls"](generator, loss_fn=loss_fn)

    # Build model from constructor params
    param_names = model_info["constructor_params"]
    defaults = model_info.get("param_defaults", {})
    model_kwargs = {
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        **{k: config.get(k, defaults.get(k, SWEEP_DEFAULTS.get(k))) for k in param_names if k not in ("vocab_size", "seq_len")},
    }
    model = model_info["cls"](**model_kwargs)
    return task, model


def _run_config_from_args(args, config: dict) -> dict:
    """Build run_config dict from args + config (for single run summary row)."""
    return {
        "model": getattr(args, "model", "simple_nn"),
        "task": getattr(args, "task", "sorting"),
        "loss": getattr(args, "loss", "cross_entropy"),
        "sequence_length": getattr(args, "seq_len") or config.get("sequence_length", 32),
        "vocabulary_size": getattr(args, "vocab_size") or config.get("vocabulary_size", 64),
        "target_noise": getattr(args, "target_noise", None) or config.get("target_noise", 0.0),
        "d_model": getattr(args, "d_model") or config.get("d_model", 64),
        "n_layers": config.get("n_layers", 1),
        "batch_size": getattr(args, "batch_size") or config.get("batch_size", 64),
        "steps": getattr(args, "steps", 5000),
        "eval_every": getattr(args, "eval_every", 1000),
        "seed": getattr(args, "seed"),
    }


def _args_from_run_config(run_config: dict):
    """Build an args-like object from a flat run config (for sweep)."""
    return SimpleNamespace(
        model=run_config.get("model", "simple_nn"),
        task=run_config.get("task", "sorting"),
        loss=run_config.get("loss", "cross_entropy"),
        seq_len=run_config.get("sequence_length"),
        vocab_size=run_config.get("vocabulary_size"),
        target_noise=run_config.get("target_noise", 0.0),
        d_model=run_config.get("d_model"),
        n_layers=run_config.get("n_layers", 1),
        batch_size=run_config.get("batch_size"),
        steps=run_config.get("steps"),
        eval_every=run_config.get("eval_every"),
        seed=run_config.get("seed"),
    )


def _run_one_experiment(run_config: dict, device: torch.device, run_id: int) -> dict:
    """Run a single experiment from a run config; return one summary row (params + metrics)."""
    if run_config.get("seed") is not None:
        torch.manual_seed(run_config["seed"])
    args = _args_from_run_config(run_config)
    task, model = _build_task_and_model(args, run_config)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    target_noise = float(run_config.get("target_noise", 0.0))
    manager = TaskManager(task=task, model=model, optimizer=optimizer, device=device, target_noise=target_noise)
    steps = int(run_config.get("steps", SWEEP_DEFAULTS["steps"]))
    batch_size = int(run_config.get("batch_size", SWEEP_DEFAULTS["batch_size"]))
    eval_every = int(run_config.get("eval_every", SWEEP_DEFAULTS["eval_every"]))
    history = manager.train(n_steps=steps, batch_size=batch_size, eval_every=eval_every)
    if not history:
        final_train_loss = final_train_acc = final_val_loss = final_val_acc = None
    else:
        last = history[-1]
        final_train_loss = last["train_loss"]
        final_train_acc = last["train_acc"]
        final_val_loss = last["val_loss"]
        final_val_acc = last["val_acc"]
    row = {
        "run_id": run_id,
        "model": _format_model_column(run_config),
        "task": run_config["task"],
        "loss": run_config["loss"],
        "seq_len": run_config["sequence_length"],
        "vocab_size": run_config["vocabulary_size"],
        "target_noise": run_config.get("target_noise", 0.0),
        "d_model": run_config["d_model"],
        "n_layers": run_config.get("n_layers", 1),
        "batch_size": run_config["batch_size"],
        "steps": run_config["steps"],
        "eval_every": run_config["eval_every"],
        "seed": run_config.get("seed"),
        "final_train_loss": final_train_loss,
        "final_train_acc": final_train_acc,
        "final_val_loss": final_val_loss,
        "final_val_acc": final_val_acc,
    }
    return row


def cmd_run(args):
    config = {}
    if getattr(args, "config", None):
        config = _load_config(args.config)
    if getattr(args, "seed", None) is not None:
        torch.manual_seed(args.seed)

    task, model = _build_task_and_model(args, config)
    device = _get_device(getattr(args, "device", "auto"))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    target_noise = getattr(args, "target_noise", None) or config.get("target_noise", 0.0)
    manager = TaskManager(task=task, model=model, optimizer=optimizer, device=device, target_noise=target_noise)
    steps = getattr(args, "steps", 5000)
    batch_size = getattr(args, "batch_size", None) or config.get("batch_size", 64)
    eval_every = getattr(args, "eval_every", 1000)

    history = manager.train(n_steps=steps, batch_size=batch_size, eval_every=eval_every)

    out_path = getattr(args, "output", None)
    if getattr(args, "trace", False) and not out_path:
        trace_dir = traces_dir()
        trace_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_name = getattr(args, "task", "run")
        model_name = getattr(args, "model", "model")
        out_path = trace_dir / f"{task_name}_{model_name}_{ts}.csv"
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["step", "train_loss", "train_acc", "val_loss", "val_acc"],
            )
            w.writeheader()
            w.writerows(history)
        print(f"Wrote trace to {out_path}")

    # Append summary to main results (unless --no-append-summary)
    append_summary = not getattr(args, "no_append_summary", False)
    if append_summary:
        run_config = _run_config_from_args(args, config)
        if history:
            last = history[-1]
            final_train_loss = last["train_loss"]
            final_train_acc = last["train_acc"]
            final_val_loss = last["val_loss"]
            final_val_acc = last["val_acc"]
        else:
            final_train_loss = final_train_acc = final_val_loss = final_val_acc = None
        row = {
            "run_id": get_next_run_id(),
            "model": _format_model_column(run_config),
            "task": run_config["task"],
            "loss": run_config["loss"],
            "seq_len": run_config["sequence_length"],
            "vocab_size": run_config["vocabulary_size"],
            "target_noise": run_config.get("target_noise", 0.0),
            "d_model": run_config["d_model"],
            "n_layers": run_config.get("n_layers", 1),
            "batch_size": run_config["batch_size"],
            "steps": run_config["steps"],
            "eval_every": run_config["eval_every"],
            "seed": run_config.get("seed"),
            "final_train_loss": final_train_loss,
            "final_train_acc": final_train_acc,
            "final_val_loss": final_val_loss,
            "final_val_acc": final_val_acc,
        }
        results_path = append_results([row])
        print(f"Appended summary to {results_path}")

    show_n = getattr(args, "show_examples", None)
    if show_n is not None:
        manager.show_examples(n_examples=show_n if isinstance(show_n, int) else 5)


def cmd_sweep(args):
    """Run many experiments from one sweep config (list values = grid)."""
    config = _load_config(args.config)
    # Support nested: merge top-level defaults with sweep section (sweep overrides)
    if "sweep" in config and isinstance(config.get("sweep"), dict):
        expand_config = {k: v for k, v in config.items() if k != "sweep"}
        expand_config.update(config["sweep"])
    else:
        expand_config = config
    device = _get_device(args.device)
    run_configs = _expand_sweep_config(expand_config)

    out_path = Path(args.output) if args.output else default_results_path()
    overwrite = getattr(args, "overwrite", False)

    existing_rows: list[dict] = []
    explored_signatures: set[tuple] = set()
    if not overwrite and out_path.exists():
        existing_rows, explored_signatures = load_existing_rows(out_path, _config_signature)
        print(f"Found {len(existing_rows)} existing result(s) in {out_path}")

    # Filter to configs not yet explored
    to_run = [rc for rc in run_configs if _config_signature(rc) not in explored_signatures]
    skipped = len(run_configs) - len(to_run)
    if skipped:
        print(f"Skipping {skipped} already-explored configuration(s)")

    if not to_run:
        print("No new configurations to run.")
        return

    n = len(to_run)
    print(f"Sweep: {n} new run(s) from {args.config}")
    base_run_id = 0 if overwrite else len(existing_rows)
    for i, run_config in enumerate(to_run):
        print(f"  [{i + 1}/{n}] model={run_config.get('model', 'simple_nn')} task={run_config['task']} seq_len={run_config['sequence_length']}")
        row = _run_one_experiment(run_config, device, run_id=base_run_id + i)
        if overwrite and i == 0:
            overwrite_results([row], out_path)
        else:
            append_results([row], out_path)

    print(f"Sweep complete: {len(to_run)} result(s) saved to {out_path}")


def cmd_list(args):
    if args.tasks:
        print("Tasks:")
        for name in all_task_names():
            info = get_task(name)
            desc = info["description"] if info else ""
            print(f"  {name:12} – {desc}")
        return
    if args.losses:
        print("Losses:")
        print("  cross_entropy – flattened CE over sequence")
        return
    if args.models:
        print("Models:")
        for name in all_model_names():
            info = get_model(name)
            params = info["constructor_params"] if info else []
            print(f"  {name:12} – ({', '.join(params)})")
        return
    if args.config:
        # Try repo config if running from source; else show CLI defaults
        repo_root = Path(__file__).resolve().parent.parent.parent
        default_path = repo_root / "configs" / "generation_default.yaml"
        if default_path.exists():
            cfg = _load_config(str(default_path))
            print("Default config (configs/generation_default.yaml):")
            for k, v in sorted(cfg.items()):
                print(f"  {k}: {v}")
        else:
            print("CLI defaults (use sfb run -c path/to/config.yaml to override):")
            print("  sequence_length: 32")
            print("  vocabulary_size: 64")
            print("  batch_size: 64")
            print("  d_model: 64")
            print("  steps: 5000")
            print("  eval_every: 100")
        return
    # default: summary
    print("SeqFactorBench (sfb) – sequence model benchmark")
    print()
    print(f"Tasks:   {', '.join(all_task_names())}")
    print(f"Models:  {', '.join(all_model_names())}")
    print("Losses:  cross_entropy")
    print()
    print("Usage:   sfb run --task <task> [options]")
    print("         sfb sweep -c <sweep.yaml> [-o summary.csv]  (one config, many params)")
    print("         sfb list [--tasks | --models | --losses | --config]")


def cmd_version(args):
    try:
        print(version("sfb"))
    except PackageNotFoundError:
        print("Version unknown (package not installed)")


def _resolve_save_path(save: str | None) -> Path | None:
    """Resolve --save path: bare filenames go to data/figures/, else use as given."""
    if not save:
        return None
    p = Path(save)
    if len(p.parts) <= 1 or p.parts[0] == ".":
        return figures_dir() / (p.name or save)
    return p


def cmd_report(args):
    """Report and visualize results."""
    from seqfacben.analysis import (
        load_results,
        filter_results,
        plot_metrics_grid,
        plot_learning_curve,
        plot_sensitivity,
    )

    sub = getattr(args, "report_subcommand", "metrics")
    save_path = _resolve_save_path(args.save)

    if sub == "curve":
        path = getattr(args, "trace_path", None)
        if not path:
            raise SystemExit("Usage: sfb report curve <path_to_trace.csv>")
        path = Path(path)
        if not path.exists():
            raise SystemExit(f"Trace file not found: {path}")
        fig = plot_learning_curve(path, save=save_path)
        if fig and not save_path:
            import matplotlib.pyplot as plt
            plt.show()
        elif save_path:
            print(f"Saved learning curve plot to {save_path}")
    elif sub in ("noise", "seq-len", "vocab"):
        df = load_results(args.results)
        if df.empty:
            print("No results found. Run experiments first.")
            return
        df = filter_results(
            df,
            task=args.task or None,
            model=args.model or None,
            seq_len=args.seq_len or None,
            vocab_size=args.vocab_size or None,
            target_noise=args.target_noise if getattr(args, "target_noise", None) is not None else None,
        )
        if df.empty:
            print("No results match the filters.")
            return
        x_col = {"noise": "target_noise", "seq-len": "seq_len", "vocab": "vocab_size"}.get(sub, "target_noise")
        fig = plot_sensitivity(
            df,
            x_col=x_col,
            metric=args.metric,
            facet_by=args.facet_by if getattr(args, "facet_by", None) else "task",
            save=save_path,
        )
        if fig and not save_path:
            import matplotlib.pyplot as plt
            plt.show()
        elif save_path and fig:
            print(f"Saved sensitivity plot to {save_path}")
        elif not fig:
            print(f"Need at least 2 {x_col} levels in the data. Sweep over {x_col}.")
    else:
        # metrics grid
        df = load_results(args.results)
        if df.empty:
            print("No results found. Run experiments first.")
            return
        df = filter_results(
            df,
            task=args.task or None,
            model=args.model or None,
            seq_len=args.seq_len or None,
            vocab_size=args.vocab_size or None,
            target_noise=args.target_noise if getattr(args, "target_noise", None) is not None else None,
        )
        if df.empty:
            print("No results match the filters.")
            return
        fig = plot_metrics_grid(
            df,
            x=args.x,
            y=args.y,
            metric=args.metric,
            facet_by=args.facet_by if args.facet_by else None,
            save=save_path,
        )
        if fig and not save_path:
            import matplotlib.pyplot as plt
            plt.show()
        elif save_path:
            print(f"Saved metrics grid to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        prog="sfb",
        description="SeqFactorBench CLI: run benchmarks and list tasks/models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    run_p = subparsers.add_parser("run", help="Run a single benchmark (task + model)")
    run_p.add_argument("-t", "--task", required=True, choices=all_task_names(), help="Task name")
    run_p.add_argument("-m", "--model", default="simple_nn", help="Model name (default: simple_nn)")
    run_p.add_argument("-l", "--loss", default="cross_entropy", choices=list(LOSSES), help="Loss function (default: cross_entropy)")
    run_p.add_argument("--seq-len", type=int, default=None, help="Sequence length")
    run_p.add_argument("--vocab-size", type=int, default=None, help="Vocabulary size")
    run_p.add_argument("--target-noise", type=float, default=None, help="Label noise rate [0-1] during training (0 = none)")
    run_p.add_argument("--d-model", type=int, default=None, help="Model hidden dimension")
    run_p.add_argument("-n", "--steps", type=int, default=5000, help="Training steps")
    run_p.add_argument("-b", "--batch-size", type=int, default=None, help="Batch size")
    run_p.add_argument("--eval-every", type=int, default=1000, help="Eval every N steps")
    run_p.add_argument("-o", "--output", default=None, help="Path for step-level history (optional)")
    run_p.add_argument("--trace", action="store_true", help="Save step history to data/traces/")
    run_p.add_argument("--no-append-summary", action="store_true", help="Do not append to data/results/results.csv")
    run_p.add_argument("-c", "--config", default=None, help="YAML config path (overrides with CLI)")
    run_p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device")
    run_p.add_argument("--seed", type=int, default=None, help="Random seed")
    run_p.add_argument(
        "--show-examples",
        nargs="?",
        const=5,
        type=int,
        default=None,
        metavar="N",
        help="Show N example predictions at end (default: 5)",
    )
    run_p.set_defaults(func=cmd_run)

    # sweep: one config file, list values = parameter grid (Cartesian product)
    sweep_p = subparsers.add_parser(
        "sweep",
        help="Run many experiments from one YAML; use lists to sweep parameters.",
    )
    sweep_p.add_argument(
        "-c", "--config",
        required=True,
        help="Sweep YAML path (e.g. configs/sweep.yaml). List values = grid dimension.",
    )
    sweep_p.add_argument(
        "-o", "--output",
        default=None,
        help="Results path (default: data/results/results.csv)",
    )
    sweep_p.add_argument("--overwrite", action="store_true", help="Replace results file instead of appending")
    sweep_p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device")
    sweep_p.set_defaults(func=cmd_sweep)

    # report: visualize results
    report_p = subparsers.add_parser("report", help="Visualize results and learning curves")
    report_sub = report_p.add_subparsers(dest="report_subcommand", required=False)
    # metrics (default): sfb report [filters] [--save]
    report_p.add_argument("-r", "--results", default=None, help="Results path (default: data/results/results.csv)")
    report_p.add_argument("--task", type=str, default=None, help="Filter by task")
    report_p.add_argument("--model", type=str, default=None, help="Filter by model (base name)")
    report_p.add_argument("--seq-len", type=int, default=None, help="Filter by seq_len")
    report_p.add_argument("--vocab-size", type=int, default=None, help="Filter by vocab_size")
    report_p.add_argument("--target-noise", type=float, default=None, help="Filter by target_noise")
    report_p.add_argument("--x", default="seq_len", help="Column for x axis (default: seq_len)")
    report_p.add_argument("--y", default="task", help="Column for y axis (default: task)")
    report_p.add_argument("--metric", default="final_val_acc", help="Metric to show (default: final_val_acc)")
    report_p.add_argument("--facet-by", default="model", help="Subplot by column (default: model)")
    report_p.add_argument("--save", default=None, help="Save figure to path (bare filename -> data/figures/)")
    report_p.set_defaults(func=cmd_report, report_subcommand="metrics")
    # curve: sfb report curve <path>
    curve_p = report_sub.add_parser("curve", help="Plot learning curve from trace file")
    curve_p.add_argument("trace_path", help="Path to trace CSV (from sfb run --trace or -o)")
    curve_p.add_argument("--save", default=None, help="Save figure to path (bare filename -> data/figures/)")
    curve_p.set_defaults(func=cmd_report, report_subcommand="curve")
    # noise: sfb report noise — metric vs target_noise (line plot)
    noise_p = report_sub.add_parser("noise", help="Plot metric vs target_noise (noise sensitivity)")
    noise_p.add_argument("--metric", default="final_val_acc", help="Metric (default: final_val_acc)")
    noise_p.add_argument("--facet-by", default="task", help="Subplot by column (default: task)")
    noise_p.add_argument("--save", default=None, help="Save figure to path (bare filename -> data/figures/)")
    noise_p.set_defaults(func=cmd_report, report_subcommand="noise")
    # seq-len: sfb report seq-len — metric vs seq_len (sequence length sensitivity)
    seqlen_p = report_sub.add_parser("seq-len", help="Plot metric vs seq_len (sequence length sensitivity)")
    seqlen_p.add_argument("--metric", default="final_val_acc", help="Metric (default: final_val_acc)")
    seqlen_p.add_argument("--facet-by", default="task", help="Subplot by column (default: task)")
    seqlen_p.add_argument("--save", default=None, help="Save figure to path (bare filename -> data/figures/)")
    seqlen_p.set_defaults(func=cmd_report, report_subcommand="seq-len")
    # vocab: sfb report vocab — metric vs vocab_size (vocabulary size sensitivity)
    vocab_p = report_sub.add_parser("vocab", help="Plot metric vs vocab_size (vocabulary size sensitivity)")
    vocab_p.add_argument("--metric", default="final_val_acc", help="Metric (default: final_val_acc)")
    vocab_p.add_argument("--facet-by", default="task", help="Subplot by column (default: task)")
    vocab_p.add_argument("--save", default=None, help="Save figure to path (bare filename -> data/figures/)")
    vocab_p.set_defaults(func=cmd_report, report_subcommand="vocab")

    # list
    list_p = subparsers.add_parser("list", help="List tasks, models, or default config")
    list_p.add_argument("--tasks", action="store_true", help="List available tasks")
    list_p.add_argument("--models", action="store_true", help="List available models")
    list_p.add_argument("--losses", action="store_true", help="List available loss functions")
    list_p.add_argument("--config", action="store_true", help="Show default config values")
    list_p.set_defaults(func=cmd_list)

    # version
    version_p = subparsers.add_parser("version", help="Show SeqFactorBench version")
    version_p.set_defaults(func=cmd_version)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

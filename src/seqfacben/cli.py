"""
CLI for SeqFactorBench. Invoke as: sfb run ... | sfb sweep ... | sfb list ...
"""
import argparse
import csv
import itertools
import sys
from pathlib import Path
from types import SimpleNamespace
from importlib.metadata import version, PackageNotFoundError

import pandas as pd
import torch

from seqfacben.generators.random import RandomSequenceGenerator
from seqfacben.tasks.sorting import SortingTask
from seqfacben.tasks.copy import CopyTask
from seqfacben.models.simple_nn import SimpleNN
from seqfacben.manager.task_manager import TaskManager
from seqfacben.losses import cross_entropy

LOSSES = {"cross_entropy": cross_entropy}


def _load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


# Keys that can be swept (list = dimension) or fixed (scalar). Used by sweep command.
SWEEP_PARAM_KEYS = [
    "task", "loss", "sequence_length", "vocabulary_size", "d_model",
    "batch_size", "steps", "eval_every", "seed",
]
SWEEP_DEFAULTS = {
    "task": "sorting",
    "loss": "cross_entropy",
    "sequence_length": 32,
    "vocabulary_size": 64,
    "d_model": 64,
    "batch_size": 64,
    "steps": 5000,
    "eval_every": 1000,
    "seed": None,
}


def _expand_sweep_config(config: dict) -> list[dict]:
    """Expand a config into a list of run configs. List values → Cartesian product."""
    fixed = {}
    dims = {}
    for k in SWEEP_PARAM_KEYS:
        v = config.get(k, SWEEP_DEFAULTS.get(k))
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
    """Resolve seq_len, vocab_size from args + config. Build generator, task, model."""
    seq_len = getattr(args, "seq_len", None) or config.get("sequence_length", 32)
    vocab_size = getattr(args, "vocab_size", None) or config.get("vocabulary_size", 64)
    d_model = getattr(args, "d_model", None) or config.get("d_model", 64)

    loss_name = (getattr(args, "loss", None) or config.get("loss", "cross_entropy")).lower()
    loss_fn = LOSSES.get(loss_name)
    if loss_fn is None:
        raise SystemExit(f"Unknown loss: {loss_name}. Use: {', '.join(LOSSES)}.")

    generator = RandomSequenceGenerator(seq_len=seq_len, vocab_size=vocab_size)
    task_name = (getattr(args, "task", None) or "").lower()
    if task_name == "sorting":
        task = SortingTask(generator, loss_fn=loss_fn)
    elif task_name == "copy":
        task = CopyTask(generator, loss_fn=loss_fn)
    else:
        raise SystemExit(f"Unknown task: {args.task}. Use: sorting, copy.")

    model = SimpleNN(vocab_size=vocab_size, seq_len=seq_len, d_model=d_model)
    return task, model


def _args_from_run_config(run_config: dict):
    """Build an args-like object from a flat run config (for sweep)."""
    return SimpleNamespace(
        task=run_config.get("task", "sorting"),
        loss=run_config.get("loss", "cross_entropy"),
        seq_len=run_config.get("sequence_length"),
        vocab_size=run_config.get("vocabulary_size"),
        d_model=run_config.get("d_model"),
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
    manager = TaskManager(task=task, model=model, optimizer=optimizer, device=device)
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
        "task": run_config["task"],
        "loss": run_config["loss"],
        "seq_len": run_config["sequence_length"],
        "vocab_size": run_config["vocabulary_size"],
        "d_model": run_config["d_model"],
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

    manager = TaskManager(task=task, model=model, optimizer=optimizer, device=device)
    steps = getattr(args, "steps", 5000)
    batch_size = getattr(args, "batch_size", None) or config.get("batch_size", 64)
    eval_every = getattr(args, "eval_every", 1000)

    history = manager.train(n_steps=steps, batch_size=batch_size, eval_every=eval_every)

    out_path = getattr(args, "output", None)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["step", "train_loss", "train_acc", "val_loss", "val_acc"],
            )
            w.writeheader()
            w.writerows(history)
        print(f"Wrote history to {out_path}")


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
    n = len(run_configs)
    print(f"Sweep: {n} run(s) from {args.config}")
    rows = []
    for i, run_config in enumerate(run_configs):
        print(f"  [{i + 1}/{n}] task={run_config['task']} seq_len={run_config['sequence_length']} (eval_every={run_config['eval_every']})")
        row = _run_one_experiment(run_config, device, run_id=i)
        rows.append(row)
    df = pd.DataFrame(rows)
    out_path = args.output or "data/sweep_results.csv"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} rows to {out_path}")


def cmd_list(args):
    if args.tasks:
        print("Tasks:")
        print("  sorting  – sort input sequence")
        print("  copy     – copy input to output")
        return
    if args.losses:
        print("Losses:")
        print("  cross_entropy – flattened CE over sequence")
        return
    if args.models:
        print("Models:")
        print("  simple_nn  – MLP over flattened sequence (vocab_size, seq_len, d_model)")
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
    print("Tasks:   sorting, copy")
    print("Models:  simple_nn")
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


def main():
    parser = argparse.ArgumentParser(
        prog="sfb",
        description="SeqFactorBench CLI: run benchmarks and list tasks/models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    run_p = subparsers.add_parser("run", help="Run a single benchmark (task + model)")
    run_p.add_argument("-t", "--task", required=True, choices=["sorting", "copy"], help="Task name")
    run_p.add_argument("-m", "--model", default="simple_nn", help="Model name (default: simple_nn)")
    run_p.add_argument("-l", "--loss", default="cross_entropy", choices=list(LOSSES), help="Loss function (default: cross_entropy)")
    run_p.add_argument("--seq-len", type=int, default=None, help="Sequence length")
    run_p.add_argument("--vocab-size", type=int, default=None, help="Vocabulary size")
    run_p.add_argument("--d-model", type=int, default=None, help="Model hidden dimension")
    run_p.add_argument("-n", "--steps", type=int, default=5000, help="Training steps")
    run_p.add_argument("-b", "--batch-size", type=int, default=None, help="Batch size")
    run_p.add_argument("--eval-every", type=int, default=1000, help="Eval every N steps")
    run_p.add_argument("-o", "--output", default=None, help="CSV path for step/loss/accuracy history")
    run_p.add_argument("-c", "--config", default=None, help="YAML config path (overrides with CLI)")
    run_p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device")
    run_p.add_argument("--seed", type=int, default=None, help="Random seed")
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
        help="Output CSV path for summary (default: data/sweep_results.csv)",
    )
    sweep_p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device")
    sweep_p.set_defaults(func=cmd_sweep)

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

"""
Registry for models and tasks. Models and tasks self-register via decorators.
No CLI changes needed when adding a new model or task.

To add a model: create `models/my_model.py` with @register_model("name", ...) on the class.
To add a task: create `tasks/my_task.py` with @register_task("name", description="...") on the class.
"""
import importlib
import pkgutil
from typing import Any

# name -> {cls, display_params, constructor_params}
MODELS: dict[str, dict[str, Any]] = {}
# name -> {cls, description}
TASKS: dict[str, dict[str, Any]] = {}

_models_loaded = False
_tasks_loaded = False


def _load_models():
    global _models_loaded
    if _models_loaded:
        return
    import seqfacben.models as models_pkg
    for importer, modname, _ in pkgutil.iter_modules(models_pkg.__path__, prefix="seqfacben.models."):
        if "base" not in modname:  # skip base.py
            importlib.import_module(modname)
    _models_loaded = True


def _load_tasks():
    global _tasks_loaded
    if _tasks_loaded:
        return
    import seqfacben.tasks as tasks_pkg
    for importer, modname, _ in pkgutil.iter_modules(tasks_pkg.__path__, prefix="seqfacben.tasks."):
        if "base" not in modname:
            importlib.import_module(modname)
    _tasks_loaded = True


def register_model(
    name: str,
    *,
    display_params: list[str] | None = None,
    constructor_params: list[str] | None = None,
    param_defaults: dict[str, Any] | None = None,
):
    """Register a model class. Use as @register_model('gru', display_params=[...], constructor_params=[...])."""

    def decorator(cls):
        MODELS[name] = {
            "cls": cls,
            "display_params": display_params or ["d_model"],
            "constructor_params": constructor_params or ["vocab_size", "seq_len", "d_model"],
            "param_defaults": param_defaults or {},
        }
        return cls

    return decorator


def register_task(name: str, *, description: str = ""):
    """Register a task class. Use as @register_task('reverse', description='...')."""

    def decorator(cls):
        TASKS[name] = {"cls": cls, "description": description}
        return cls

    return decorator


def get_model(name: str):
    """Get model class and metadata. Loads models on first call."""
    _load_models()
    return MODELS.get(name)


def get_task(name: str):
    """Get task class and metadata. Loads tasks on first call."""
    _load_tasks()
    return TASKS.get(name)


def all_model_names() -> list[str]:
    _load_models()
    return list(MODELS.keys())


def all_task_names() -> list[str]:
    _load_tasks()
    return list(TASKS.keys())


def all_model_param_keys() -> set[str]:
    """Union of constructor params from all models (for sweep config)."""
    _load_models()
    keys: set[str] = set()
    for info in MODELS.values():
        keys.update(info.get("constructor_params", []))
    return keys - {"vocab_size", "seq_len"}  # these come from sequence_length, vocabulary_size


def param_defaults_from_models() -> dict[str, Any]:
    """Merge param_defaults from all models (for sweep defaults)."""
    _load_models()
    merged: dict[str, Any] = {}
    for info in MODELS.values():
        merged.update(info.get("param_defaults", {}))
    return merged

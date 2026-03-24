"""
Benchmark-ready models: encoder + decoder wired as :class:`sfb.models.codec.ComposedCodecModel`.

Each submodule calls ``@register_model``; the registry loads these via ``registry._load_models``.
Import from here only when you need a concrete class (e.g. ``from sfb.models.composed.simple_nn import SimpleNNCodecModel``).
"""

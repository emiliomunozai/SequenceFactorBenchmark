"""Regression tests for the FB-Bench run path."""

import math

import torch

from sfb.cli import (
    _config_signature,
    _expand_sweep,
    _model_label,
    _run_one,
)

REPRO_CONFIG = {
    "encoder": "mlp",
    "decoder": "mlp",
    "task": "sorting",
    "loss": "cross_entropy",
    "sequence_length": 8,
    "vocabulary_size": 16,
    "d_model": 16,
    "bottleneck_dim": 16,
    "n_layers": 1,
    "n_heads": 4,
    "batch_size": 8,
    "steps": 20,
    "eval_every": 20,
    "seed": 42,
}

METRIC_KEYS = [
    "final_train_loss",
    "final_train_acc",
    "final_val_clean_loss",
    "final_val_clean_acc",
    "final_val_noisy_loss",
    "final_val_noisy_acc",
]


class TestReproducibility:
    def test_sorting_task_reproducible(self):
        device = torch.device("cpu")
        row1, _, _ = _run_one(REPRO_CONFIG, device, run_id=0)
        row2, _, _ = _run_one(REPRO_CONFIG, device, run_id=1)

        for key in METRIC_KEYS:
            assert row1[key] == row2[key], f"{key} differs: {row1[key]} vs {row2[key]}"

    def test_model_column_includes_encoder_decoder(self):
        assert _model_label(REPRO_CONFIG, 16) == "mlp->mlp(d_model=16, bottleneck_dim=16, n_layers=1)"

    def test_copy_task_reproducible(self):
        device = torch.device("cpu")
        config = {**REPRO_CONFIG, "task": "copy", "encoder": "lstm", "decoder": "lstm"}
        row1, _, _ = _run_one(config, device, run_id=0)
        row2, _, _ = _run_one(config, device, run_id=1)

        for key in METRIC_KEYS:
            assert row1[key] == row2[key], f"{key} differs: {row1[key]} vs {row2[key]}"

    def test_reverse_task_reproducible(self):
        device = torch.device("cpu")
        config = {**REPRO_CONFIG, "task": "reverse", "encoder": "rnn", "decoder": "rnn"}
        row1, _, _ = _run_one(config, device, run_id=0)
        row2, _, _ = _run_one(config, device, run_id=1)

        for key in METRIC_KEYS:
            assert row1[key] == row2[key], f"{key} differs: {row1[key]} vs {row2[key]}"

    def test_different_seeds_diverge(self):
        device = torch.device("cpu")
        row1, _, _ = _run_one({**REPRO_CONFIG, "seed": 123}, device, run_id=0)
        row2, _, _ = _run_one({**REPRO_CONFIG, "seed": 456}, device, run_id=1)

        diffs = [row1[key] != row2[key] for key in METRIC_KEYS]
        assert any(diffs), "Expected different seeds to produce different results"

    def test_input_noise_runs(self):
        device = torch.device("cpu")
        row_clean, _, _ = _run_one({**REPRO_CONFIG, "input_noise": 0.0}, device, run_id=0)
        row_noisy, _, _ = _run_one({**REPRO_CONFIG, "input_noise": 0.2}, device, run_id=1)

        for row in (row_clean, row_noisy):
            for key in METRIC_KEYS:
                value = row[key]
                assert isinstance(value, (int, float)) and math.isfinite(float(value))

    def test_transformer_pair_reproducible(self):
        device = torch.device("cpu")
        config = {**REPRO_CONFIG, "task": "copy", "encoder": "transformer", "decoder": "transformer"}
        row1, _, _ = _run_one(config, device, run_id=0)
        row2, _, _ = _run_one(config, device, run_id=1)

        for key in METRIC_KEYS:
            assert row1[key] == row2[key], f"{key} differs: {row1[key]} vs {row2[key]}"


class TestConfigSignature:
    def test_same_config_matches(self):
        cfg = {
            "encoder": "mlp",
            "decoder": "lstm",
            "task": "sorting",
            "sequence_length": 16,
            "vocabulary_size": 32,
            "input_noise": 0.0,
            "d_model": 32,
            "bottleneck_dim": 16,
            "n_layers": 1,
            "n_heads": 4,
            "seed": 42,
        }
        assert _config_signature(cfg) == _config_signature(dict(cfg))

    def test_different_encoder_decoder_pairs_differ(self):
        cfg_a = {"encoder": "mlp", "decoder": "mlp", "task": "sorting"}
        cfg_b = {"encoder": "rnn", "decoder": "mlp", "task": "sorting"}
        assert _config_signature(cfg_a) != _config_signature(cfg_b)

    def test_different_input_noise_differs(self):
        cfg_a = {"encoder": "mlp", "decoder": "mlp", "task": "sorting", "input_noise": 0.0}
        cfg_b = {"encoder": "mlp", "decoder": "mlp", "task": "sorting", "input_noise": 0.2}
        assert _config_signature(cfg_a) != _config_signature(cfg_b)

    def test_transformer_head_count_affects_signature(self):
        cfg_a = {"encoder": "transformer", "decoder": "mlp", "task": "sorting", "n_heads": 2}
        cfg_b = {"encoder": "transformer", "decoder": "mlp", "task": "sorting", "n_heads": 4}
        assert _config_signature(cfg_a) != _config_signature(cfg_b)


class TestExpandSweep:
    def test_all_fixed_returns_single_config(self):
        result = _expand_sweep({"task": "sorting", "encoder": "mlp", "decoder": "mlp", "sequence_length": 32})
        assert len(result) == 1
        assert result[0]["task"] == "sorting"
        assert result[0]["sequence_length"] == 32

    def test_list_expands_cartesian(self):
        result = _expand_sweep({
            "encoder": ["mlp", "lstm"],
            "decoder": ["mlp", "rnn"],
            "task": ["sorting", "copy"],
        })
        assert len(result) == 8
        assert {r["encoder"] for r in result} == {"mlp", "lstm"}
        assert {r["decoder"] for r in result} == {"mlp", "rnn"}
        assert {r["task"] for r in result} == {"sorting", "copy"}

    def test_models_block_expands_explicit_pairs(self):
        result = _expand_sweep({
            "task": "copy",
            "sequence_length": [8, 16],
            "models": [
                {"encoder": "mlp", "decoder": "mlp"},
                {"encoder": "lstm", "decoder": "transformer", "n_layers": 2},
            ],
        })
        assert len(result) == 4
        pairs = {(r["encoder"], r["decoder"]) for r in result}
        assert pairs == {("mlp", "mlp"), ("lstm", "transformer")}
        assert all(r["task"] == "copy" for r in result)

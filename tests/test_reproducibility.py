"""
Unit tests to verify that experiments are reproducible given a fixed config and seed.
"""
import pytest
import torch

from seqfacben.cli import (
    _run_one_experiment,
    _config_signature,
    _expand_sweep_config,
    SWEEP_DEFAULTS,
)


# Fixed config for reproducibility tests. Small steps for fast tests.
REPRO_CONFIG = {
    "task": "sorting",
    "loss": "cross_entropy",
    "sequence_length": 16,
    "vocabulary_size": 32,
    "d_model": 32,
    "batch_size": 32,
    "steps": 200,
    "eval_every": 100,
    "seed": 42,
}


def _metric_keys():
    """Keys that should match across runs for the same config."""
    return ["final_train_loss", "final_train_acc", "final_val_loss", "final_val_acc"]


class TestReproducibility:
    """Verify same config + seed produces identical results."""

    def test_sorting_task_reproducible(self):
        """Running sorting task twice with same seed yields identical metrics."""
        device = torch.device("cpu")
        row1 = _run_one_experiment(REPRO_CONFIG, device, run_id=0)
        row2 = _run_one_experiment(REPRO_CONFIG, device, run_id=1)

        for key in _metric_keys():
            assert row1[key] == row2[key], f"{key} differs: {row1[key]} vs {row2[key]}"

    def test_copy_task_reproducible(self):
        """Running copy task twice with same seed yields identical metrics."""
        config = {**REPRO_CONFIG, "task": "copy"}
        device = torch.device("cpu")

        row1 = _run_one_experiment(config, device, run_id=0)
        row2 = _run_one_experiment(config, device, run_id=1)

        for key in _metric_keys():
            assert row1[key] == row2[key], f"{key} differs: {row1[key]} vs {row2[key]}"

    def test_different_seeds_diverge(self):
        """Different seeds produce different results (sanity check)."""
        device = torch.device("cpu")
        config1 = {**REPRO_CONFIG, "seed": 123}
        config2 = {**REPRO_CONFIG, "seed": 456}

        row1 = _run_one_experiment(config1, device, run_id=0)
        row2 = _run_one_experiment(config2, device, run_id=0)

        # At least one metric should differ
        diffs = [row1[k] != row2[k] for k in _metric_keys()]
        assert any(diffs), "Expected different seeds to produce different results"


class TestConfigSignature:
    """Verify config signature for sweep deduplication."""

    def test_run_config_and_row_match(self):
        """Signature from run_config (sequence_length/vocabulary_size) matches row (seq_len/vocab_size)."""
        run_config = {
            "task": "sorting",
            "loss": "cross_entropy",
            "sequence_length": 32,
            "vocabulary_size": 64,
            "d_model": 64,
            "batch_size": 64,
            "steps": 5000,
            "eval_every": 1000,
            "seed": 42,
        }
        row = {
            "task": "sorting",
            "loss": "cross_entropy",
            "seq_len": 32,
            "vocab_size": 64,
            "d_model": 64,
            "batch_size": 64,
            "steps": 5000,
            "eval_every": 1000,
            "seed": 42,
        }
        assert _config_signature(run_config) == _config_signature(row)

    def test_different_configs_differ(self):
        """Different configs produce different signatures."""
        cfg_a = {"task": "sorting", "sequence_length": 32, "vocabulary_size": 64}
        cfg_b = {"task": "sorting", "sequence_length": 64, "vocabulary_size": 64}
        assert _config_signature(cfg_a) != _config_signature(cfg_b)


class TestExpandSweepConfig:
    """Verify sweep config expansion."""

    def test_all_fixed_returns_single_config(self):
        """No list values yields one config."""
        config = {"task": "sorting", "sequence_length": 32}
        result = _expand_sweep_config(config)
        assert len(result) == 1
        assert result[0]["task"] == "sorting"
        assert result[0]["sequence_length"] == 32

    def test_list_expands_cartesian(self):
        """List values expand to Cartesian product."""
        config = {
            "task": ["sorting", "copy"],
            "sequence_length": [16, 32],
        }
        result = _expand_sweep_config(config)
        assert len(result) == 4  # 2 * 2
        tasks = {r["task"] for r in result}
        seq_lens = {r["sequence_length"] for r in result}
        assert tasks == {"sorting", "copy"}
        assert seq_lens == {16, 32}

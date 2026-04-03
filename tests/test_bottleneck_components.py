"""Shape and registry checks for the active FB-Bench model family."""

import torch

from sfb.cli import _build_task_and_model
from sfb.registry import all_decoder_names, all_encoder_names


def test_phase1_registry_names():
    assert set(all_encoder_names()) == {"mlp", "lstm", "bi_lstm", "rnn", "transformer"}
    assert set(all_decoder_names()) == {"mlp", "lstm", "rnn", "transformer"}


def test_all_encoder_decoder_pairs_build_and_forward():
    x = torch.randint(0, 16, (4, 8))
    for encoder in all_encoder_names():
        for decoder in all_decoder_names():
            _, model, _ = _build_task_and_model(
                {
                    "encoder": encoder,
                    "decoder": decoder,
                    "task": "copy",
                    "loss": "cross_entropy",
                    "sequence_length": 8,
                    "vocabulary_size": 16,
                    "d_model": 16,
                    "bottleneck_dim": 12,
                    "n_layers": 1,
                    "n_heads": 4,
                }
            )
            z = model.encoder.encode(x).z
            logits = model(x)
            assert z.shape == (4, 12)
            assert logits.shape == (4, 8, 16)

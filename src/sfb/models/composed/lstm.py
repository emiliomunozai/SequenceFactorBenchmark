"""Registered LSTM codec (encoder / decoder building blocks live under ``encoders`` / ``decoders``)."""
from sfb.models.codec import ComposedCodecModel, resolve_d_bottleneck
from sfb.models.decoders import build_sequence_decoder
from sfb.models.encoders.lstm import LSTMSequenceEncoder
from sfb.registry import register_model


@register_model(
    "lstm",
    display_params=["d_model", "d_bottleneck", "n_layers", "decoder"],
    constructor_params=[
        "vocab_size",
        "seq_len",
        "d_model",
        "n_layers",
        "d_bottleneck",
        "decoder",
    ],
    param_defaults={"n_layers": 1, "decoder": "linear"},
)
class LSTMCodecModel(ComposedCodecModel):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        n_layers: int = 1,
        d_bottleneck: int | None = None,
        decoder: str = "linear",
    ):
        db = resolve_d_bottleneck(d_model, d_bottleneck)
        enc = LSTMSequenceEncoder(vocab_size, seq_len, d_model, db, n_layers)
        dec = build_sequence_decoder(
            decoder,
            z_dim=enc.out_dim,
            seq_len=seq_len,
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
        )
        super().__init__(enc, dec)

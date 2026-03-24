from sfb.models.codec import ComposedCodecModel, resolve_d_bottleneck
from sfb.models.decoders import build_sequence_decoder
from sfb.models.encoders.simple_nn import SimpleNNSequenceEncoder
from sfb.registry import register_model


@register_model(
    "simple_nn",
    display_params=["d_model", "d_bottleneck", "decoder"],
    constructor_params=[
        "vocab_size",
        "seq_len",
        "d_model",
        "d_bottleneck",
        "decoder",
    ],
    param_defaults={"decoder": "linear"},
)
class SimpleNNCodecModel(ComposedCodecModel):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        d_bottleneck: int | None = None,
        decoder: str = "linear",
    ):
        db = resolve_d_bottleneck(d_model, d_bottleneck)
        enc = SimpleNNSequenceEncoder(vocab_size, seq_len, d_model, db)
        dec = build_sequence_decoder(
            decoder,
            z_dim=enc.out_dim,
            seq_len=seq_len,
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=1,
        )
        super().__init__(enc, dec)

"""Registered MLP autoencoder codec."""
from sfb.models.codec import ComposedCodecModel
from sfb.models.decoders.autoencoder import AutoencoderSequenceDecoder
from sfb.models.encoders.autoencoder import AutoencoderSequenceEncoder
from sfb.registry import register_model


@register_model(
    "autoencoder",
    display_params=["d_model", "d_bottleneck", "n_layers"],
    constructor_params=[
        "vocab_size",
        "seq_len",
        "d_model",
        "n_layers",
        "bottleneck_dim",
        "d_bottleneck",
    ],
    param_defaults={"n_layers": 2, "bottleneck_dim": 16},
)
class AutoencoderCodecModel(ComposedCodecModel):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        n_layers: int = 2,
        bottleneck_dim: int = 16,
        d_bottleneck: int | None = None,
    ):
        bd = int(d_bottleneck) if d_bottleneck is not None else int(bottleneck_dim)
        enc = AutoencoderSequenceEncoder(vocab_size, seq_len, d_model, n_layers, bd)
        dec = AutoencoderSequenceDecoder(bd, seq_len, d_model, vocab_size, n_layers, enc.tail_dim)
        super().__init__(enc, dec)

"""
GRU encoder: processes the sequence step-by-step with a Gated Recurrent Unit.
Well-suited for sequence-oriented tasks like reverse.
"""
from torch import nn
from seqfacben.models.base import BaseModel
from seqfacben.registry import register_model


@register_model(
    "gru",
    display_params=["d_model", "n_layers"],
    constructor_params=["vocab_size", "seq_len", "d_model", "n_layers"],
    param_defaults={"n_layers": 1},
)
class GRUEncoder(BaseModel):
    """GRU encoder that reads the sequence left-to-right and outputs logits per position."""

    def __init__(self, vocab_size: int, seq_len: int, d_model: int, n_layers: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        self.embed = nn.Embedding(vocab_size, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: [batch, seq_len]
        x = self.embed(x)  # [batch, seq_len, d_model]
        out, _ = self.gru(x)  # out: [batch, seq_len, d_model]
        logits = self.head(out)  # [batch, seq_len, vocab_size]
        return logits

    def reset_state(self):
        """Stateless per forward (hidden is not carried across batches)."""
        pass

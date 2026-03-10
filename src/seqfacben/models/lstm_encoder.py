"""
LSTM encoder: processes the sequence step-by-step with a Long Short-Term Memory unit.
Stronger gating than GRU — cell state allows selective memory retention.
"""
from torch import nn
from seqfacben.registry import register_model


@register_model(
    "lstm",
    display_params=["d_model", "n_layers"],
    constructor_params=["vocab_size", "seq_len", "d_model", "n_layers"],
    param_defaults={"n_layers": 1},
)
class LSTMEncoder(nn.Module):
    """LSTM encoder that reads the sequence left-to-right and outputs logits per position."""

    def __init__(self, vocab_size: int, seq_len: int, d_model: int, n_layers: int = 1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        return self.head(out)

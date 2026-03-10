"""
Bidirectional GRU encoder: processes the sequence in both directions.
Each output position has context from the full sequence, giving it an advantage
on tasks where future tokens inform the current output (e.g. copy, sorting).
"""
from torch import nn
from seqfacben.registry import register_model


@register_model(
    "bigru",
    display_params=["d_model", "n_layers"],
    constructor_params=["vocab_size", "seq_len", "d_model", "n_layers"],
    param_defaults={"n_layers": 1},
)
class BiGRUEncoder(nn.Module):
    """Bidirectional GRU — output dim is 2*d_model, projected back to vocab."""

    def __init__(self, vocab_size: int, seq_len: int, d_model: int, n_layers: int = 1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.head = nn.Linear(d_model * 2, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.gru(x)
        return self.head(out)

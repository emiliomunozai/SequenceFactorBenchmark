from torch import nn
from seqfacben.models.base import BaseModel
from seqfacben.registry import register_model


@register_model("simple_nn", display_params=["d_model"], param_defaults={"d_model": 64})
class SimpleNN(BaseModel):
    def __init__(self, vocab_size: int, seq_len: int, d_model: int):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # Network sees the ENTIRE flattened sequence
        self.net = nn.Sequential(
            nn.Linear(d_model * seq_len, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, vocab_size * seq_len)
        )

    def forward(self, x):
        # x: [batch, seq_len]
        batch_size = x.size(0)
        x = self.embed(x)  # [batch, seq_len, d_model]
        x = x.view(batch_size, -1)  # [batch, seq_len * d_model] - FLATTEN
        x = self.net(x)  # [batch, vocab_size * seq_len]
        x = x.view(batch_size, self.seq_len, self.vocab_size)  # [batch, seq_len, vocab_size]
        return x

    def reset_state(self):
        """Stateless model: no-op."""
        pass
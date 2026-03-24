"""LSTM encoder: bottleneck = stacked final hidden states (``n_layers`` × ``d_bottleneck``)."""

from torch import nn

from sfb.models.codec import EncoderOutput, SequenceEncoder


class LSTMSequenceEncoder(SequenceEncoder):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        d_bottleneck: int,
        n_layers: int,
    ):
        super().__init__(vocab_size, seq_len)
        self.d_model = d_model
        self.d_bottleneck = d_bottleneck
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.LSTM(d_model, d_bottleneck, num_layers=n_layers, batch_first=True)
        self.out_dim = n_layers * d_bottleneck

    def encode(self, x):
        x = self.embed(x)
        _, (h_n, _) = self.rnn(x)
        z = h_n.transpose(0, 1).reshape(x.size(0), -1)
        return EncoderOutput(z=z)

import torch
from seqfacben.generators.base import BaseGenerator

class RandomSequenceGenerator(BaseGenerator):
    def sample(self, batch_size: int):
        x = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(batch_size, self.seq_len),
            dtype=torch.long
        )
        return x

import torch
from sfb.generators.base import BaseGenerator

class RandomSequenceGenerator(BaseGenerator):
    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        *,
        input_noise: float = 0.0,
        corruption_mode: str = "replace",
        mask_token_id: int = 0,
        pad_token_id: int | None = None,
    ):
        super().__init__(
            seq_len,
            vocab_size,
            input_noise=input_noise,
            corruption_mode=corruption_mode,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
        )

    def sample(self, batch_size: int):
        x = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(batch_size, self.seq_len),
            dtype=torch.long
        )
        return x

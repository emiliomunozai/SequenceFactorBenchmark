from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        """
        x: LongTensor [B, T]
        returns logits: FloatTensor [B, T, V]
        """
        pass

    @abstractmethod
    def reset_state(self):
        """
        For stateful models (RNN, memory, etc).
        Stateless models can no-op.
        """
        pass

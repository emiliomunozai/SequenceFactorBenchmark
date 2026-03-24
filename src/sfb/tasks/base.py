from abc import ABC, abstractmethod
from collections.abc import Callable

import torch


class BaseTask(ABC):
    """Subclasses set ``generator`` and ``loss_fn``; token-level loss/eval are shared."""

    generator: object
    loss_fn: Callable[..., torch.Tensor]

    @abstractmethod
    def get_batch(self, batch_size: int, split: str = "train"):
        """
        split: 'train' | 'eval'
        """
        pass

    def loss(self, model, batch):
        x, y = batch
        return self.loss_fn(model(x), y)

    def evaluate(self, model, batch):
        x, y = batch
        preds = model(x).argmax(dim=-1)
        return (preds == y).float().mean().item()

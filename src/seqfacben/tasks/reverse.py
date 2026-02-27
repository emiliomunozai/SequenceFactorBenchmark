"""
Reverse task: output the input sequence in reversed order.
Sequence-oriented — the model must process temporal structure.
"""
import torch
from seqfacben.tasks.base import BaseTask
from seqfacben.registry import register_task


@register_task("reverse", description="output sequence in reversed order (sequence-oriented)")
class ReverseTask(BaseTask):
    """Reverse the input sequence. Input [a,b,c,d] -> Target [d,c,b,a]."""

    def __init__(self, generator, loss_fn):
        self.generator = generator
        self.loss_fn = loss_fn

    def get_batch(self, batch_size: int, split: str = "train"):
        x = self.generator.sample(batch_size)
        y = torch.flip(x, dims=[1])
        return x, y

    def loss(self, model, batch):
        x, y = batch
        logits = model(x)
        return self.loss_fn(logits, y)

    def evaluate(self, model, batch):
        x, y = batch
        preds = model(x).argmax(dim=-1)
        return (preds == y).float().mean().item()

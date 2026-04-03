"""
Reverse task: output the input sequence in reversed order.
Sequence-oriented — the model must process temporal structure.
"""
import torch
from sfb.tasks.base import BaseTask
from sfb.registry import register_task


@register_task("reverse", description="output sequence in reversed order (sequence-oriented)")
class ReverseTask(BaseTask):
    """Reverse the input sequence. Input [a,b,c,d] -> Target [d,c,b,a]."""

    def __init__(self, generator, loss_fn):
        self.generator = generator
        self.loss_fn = loss_fn

    def get_batch(
        self,
        batch_size: int,
        split: str = "train",
        *,
        apply_input_corruption: bool | None = None,
    ):
        x_clean = self.generator.sample(batch_size)
        y = torch.flip(x_clean, dims=[1])
        x = self.generator.apply_input_corruption(
            x_clean,
            split,
            apply_input_corruption=apply_input_corruption,
        )
        return x, y

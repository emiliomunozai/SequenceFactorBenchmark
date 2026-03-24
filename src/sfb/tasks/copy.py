from sfb.tasks.base import BaseTask
from sfb.registry import register_task


@register_task("copy", description="copy input to output")
class CopyTask(BaseTask):

    def __init__(self, generator, loss_fn):
        self.generator = generator
        self.loss_fn = loss_fn

    def get_batch(self, batch_size: int, split: str = "train"):
        x = self.generator.sample(batch_size)
        y = x.clone()
        return x, y

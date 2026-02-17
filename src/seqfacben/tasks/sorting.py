import torch
from seqfacben.tasks.base import BaseTask


class SortingTask(BaseTask):

    def __init__(self, generator, loss_fn):
        self.generator = generator
        self.loss_fn = loss_fn

    def get_batch(self, batch_size: int, split: str = "train"):
        x = self.generator.sample(batch_size)
        y = torch.sort(x, dim=1).values
        return x, y

    def loss(self, model, batch):
        x, y = batch
        logits = model(x)
        return self.loss_fn(logits, y)

    def evaluate(self, model, batch):
        x, y = batch
        preds = model(x).argmax(dim=-1)
        return (preds == y).float().mean().item()

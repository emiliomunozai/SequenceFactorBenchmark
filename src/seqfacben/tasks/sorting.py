import torch
import torch.nn.functional as F
from seqfacben.tasks.base import BaseTask

class SortingTask(BaseTask):

    def __init__(self, generator):
        self.generator = generator

    def get_batch(self, batch_size: int, split: str = "train"):
        x = self.generator.sample(batch_size)
        y = torch.sort(x, dim=1).values
        return x, y

    def loss(self, model, batch):
        x, y = batch
        logits = model(x)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )

    def evaluate(self, model, batch):
        x, y = batch
        preds = model(x).argmax(dim=-1)
        correct = (preds == y).all(dim=1)
        return correct.float().mean().item()

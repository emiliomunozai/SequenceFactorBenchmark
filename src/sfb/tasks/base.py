from abc import ABC, abstractmethod

class BaseTask(ABC):

    @abstractmethod
    def get_batch(self, batch_size: int, split: str = "train"):
        """
        split: 'train' | 'eval'
        """
        pass

    @abstractmethod
    def loss(self, model, batch):
        pass

    @abstractmethod
    def evaluate(self, model, batch):
        pass

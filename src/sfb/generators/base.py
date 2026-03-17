from abc import ABC, abstractmethod

class BaseGenerator(ABC):
    def __init__(self, seq_len: int, vocab_size: int):
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    @abstractmethod
    def sample(self, batch_size: int):
        """
        Returns x: input tensor [batch_size, seq_len] (LongTensor).
        The task computes targets y from x (e.g. sort, copy).
        """
        pass

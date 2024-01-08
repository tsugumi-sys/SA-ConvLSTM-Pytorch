from abc import ABC, abstractmethod

from torch.utils.data import DataLoader


class BaseRunner(ABC):
    @abstractmethod
    def run(self, *args, **kwargs):
        pass


class BasePipeline(ABC):
    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass


class BaseDataLoaders(ABC):
    @property
    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        pass

    @property
    @abstractmethod
    def validation_dataloader(self) -> DataLoader:
        pass

    @property
    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        pass

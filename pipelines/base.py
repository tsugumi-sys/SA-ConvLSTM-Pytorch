from abc import ABC, abstractmethod

from torch.utils.data import DataLoader


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


class BaseRunner(ABC):
    @abstractmethod
    def run(self, *args, **kwargs):
        pass

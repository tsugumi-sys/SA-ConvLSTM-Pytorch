import torch
from torch.utils.data import DataLoader, Dataset

from data_loaders.base import BaseDataLoaders
from data_loaders.moving_mnist import VideoPredictionDataset


class MockMovingMNIST(Dataset):
    def __init__(self, dataset_length: int = 20):
        self.data = torch.rand((dataset_length, 20, 1, 64, 64)).to(torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class MockMovingMNISTDataLoaders(BaseDataLoaders):
    def __init__(
        self,
        dataset_length: int,
        train_batch_size: int,
        split_ratio: int = 10,
        shuffle: bool = True,
    ):
        self.dataset_length = dataset_length
        self.train_batch_size = train_batch_size
        self.split_ratio = split_ratio
        self.shuffle = shuffle

        self.train_dataset = VideoPredictionDataset(
            MockMovingMNIST(self.dataset_length), self.split_ratio
        )
        self.valid_dataset = VideoPredictionDataset(
            MockMovingMNIST(self.dataset_length), self.split_ratio
        )
        self.test_dataset = VideoPredictionDataset(
            MockMovingMNIST(self.dataset_length), self.split_ratio
        )

    @property
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle
        )

    @property
    def validation_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_dataset, batch_size=1, shuffle=self.shuffle)

    @property
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=1, shuffle=self.shuffle)


def mock_data_loader(batch_size: int = 5, dataset_length: int = 10):
    return DataLoader(
        VideoPredictionDataset(MockMovingMNIST(dataset_length=dataset_length)),
        batch_size=batch_size,
    )

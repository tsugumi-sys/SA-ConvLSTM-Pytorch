from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from data_loaders.base import BaseDataLoaders
from data_loaders.moving_mnist import VideoPredictionDataset


class MockMovingMNIST(Dataset):
    def __init__(self, dataset_length: int = 20):
        self.data = torch.rand((dataset_length, 20, 1, 64, 64))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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

        train_dataset, valid_dataset, test_dataset = random_split(
            MockMovingMNIST(self.dataset_length),
            [0.7, 0.2, 0.1],
            generator=torch.Generator().manual_seed(42),
        )

        self.train_dataset = VideoPredictionDataset(
            self.dataset_length, self.split_ratio
        )
        self.valid_dataset = VideoPredictionDataset(
            self.dataset_length, self.split_ratio
        )
        self.test_dataset = VideoPredictionDataset(
            self.dataset_length, self.split_ratio
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


def mock_data_loader(batch_size: int = 5, data_length: int = 10):
    return DataLoader(MockMovingMNIST(data_length=data_length), batch_size=batch_size)

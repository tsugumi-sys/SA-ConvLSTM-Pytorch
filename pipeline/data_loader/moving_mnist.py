from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MovingMNIST


class VideoPredictionDataset(Dataset):
    def __init__(self, data: torch.Tensor, input_frames: int = 10):
        self.data = data
        self.input_frames = input_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_frames = self.data[idx][: self.input_frames, :, :, :]
        label_frames = self.data[idx][self.input_frames :, :, :, :]
        return (
            torch.swapaxes(input_frames, 0, 1),
            torch.swapaxes(label_frames, 0, 1),
        )


class MovingMNISTDataLoaders:
    def __init__(
        self, train_batch_size: int, input_frames: int = 10, shuffle: bool = True
    ):
        self.train_batch_size = train_batch_size
        self.input_frames = input_frames
        self.shuffle = shuffle

        moving_mnist = MovingMNIST(root="./data", download=True)
        train_dataset, valid_dataset, test_dataset = random_split(
            moving_mnist,
            [0.7, 0.299, 0.001],
            generator=torch.Generator().manual_seed(42),
        )
        self.train_dataset = VideoPredictionDataset(train_dataset, self.input_frames)
        self.valid_dataset = VideoPredictionDataset(valid_dataset, self.input_frames)
        self.test_dataset = VideoPredictionDataset(test_dataset, self.input_frames)

    @property
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle
        )

    @property
    def valid_dataloader(self):
        return DataLoader(
            self.valid_dataset, batch_size=len(self.valid_dataset), shuffle=self.shuffle
        )

    @property
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=self.shuffle)

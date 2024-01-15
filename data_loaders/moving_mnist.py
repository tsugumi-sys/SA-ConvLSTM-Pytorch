import math
from typing import List, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import MovingMNIST

from data_loaders.base import BaseDataLoaders


class VideoPredictionDataset(Dataset):
    def __init__(
        self,
        data: Union[Subset, Dataset],
        input_frames: int = 10,
        label_frames: int | None = None,
    ):
        self.data = data
        self.input_frames = input_frames
        self.label_frames = label_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_frames = self.data[idx][: self.input_frames, :, :, :].to(torch.float32)
        label_frames = self.data[idx][self.input_frames :, :, :, :].to(torch.float32)
        if self.label_frames is not None:
            label_frames = label_frames[: self.label_frames, :]
        return (
            torch.swapaxes(input_frames, 0, 1),
            torch.swapaxes(label_frames, 0, 1),
        )


class MovingMNISTDataLoaders(BaseDataLoaders):
    def __init__(
        self,
        train_batch_size: int,
        validation_batch_size: int = 1,
        input_frames: int = 10,
        label_frames: int | None = None,
        split_ratios: List[float] | None = None,
        shuffle: bool = True,
    ):
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.input_frames = input_frames
        self.label_frames = label_frames
        self.shuffle = shuffle
        if split_ratios is None:
            split_ratios = [0.7, 0.2, 0.1]
        if not math.isclose(sum(split_ratios), 1) and sum(split_ratios) > 1:
            raise ValueError(
                f"Sum of `split_ratios` (train data, validation data, test data) must be 1.0, but got sum({split_ratios}) = {sum(split_ratios)}."
            )
        self.split_ratios = split_ratios

        moving_mnist = MovingMNIST(root="./data", download=True)
        train_dataset, valid_dataset, test_dataset = random_split(
            moving_mnist,
            [*self.split_ratios],
            generator=torch.Generator().manual_seed(42),
        )
        self.train_dataset = VideoPredictionDataset(
            train_dataset, self.input_frames, self.label_frames
        )
        self.valid_dataset = VideoPredictionDataset(
            valid_dataset, self.input_frames, self.label_frames
        )
        self.test_dataset = VideoPredictionDataset(test_dataset, self.input_frames)

    @property
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle
        )

    @property
    def validation_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.validation_batch_size,
            shuffle=self.shuffle,
        )

    @property
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=1, shuffle=self.shuffle)

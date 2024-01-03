from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class MockMovingMNIST(Dataset):
    def __init__(self, data_length: int = 20, split_ratio: int = 10):
        self.data = torch.rand((data_length, 20, 1, 64, 64))
        self.split_ratio = split_ratio

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_frames = self.data[idx, : self.split_ratio].to(torch.float32)
        label_frames = self.data[idx, self.split_ratio :].to(torch.float32)
        return (torch.swapaxes(input_frames, 0, 1), torch.swapaxes(label_frames, 0, 1))


def mock_data_loader(batch_size: int = 5, data_length: int = 10):
    return DataLoader(MockMovingMNIST(data_length=data_length), batch_size=batch_size)

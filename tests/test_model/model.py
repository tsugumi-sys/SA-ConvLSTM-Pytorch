from typing import Iterator

import torch
from torch import nn

from core.constants import DEVICE


class TestModel(nn.Module):
    __test__ = False

    def __init__(self, return_sequences: bool = True):
        super().__init__()
        self.return_sequences = return_sequences
        # NOTE: Set device not only torch.ones but also nn.parameter.Parameter (https://github.com/pytorch/pytorch/issues/50402)
        self.W = nn.parameter.Parameter(torch.ones(1, dtype=torch.float, device=DEVICE))
        self.B = nn.parameter.Parameter(torch.ones(1, dtype=torch.float, device=DEVICE))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.return_sequences:
            return torch.sigmoid(X * self.W + self.B)

        output = torch.sigmoid(X[:, :, -1] * self.W + self.B)
        batch_size, out_channels, height, width = output.size()
        return torch.reshape(output, (batch_size, out_channels, 1, height, width))

    def parameters(self, recurse: bool = True) -> Iterator[nn.parameter.Parameter]:
        return iter((self.W, self.B))

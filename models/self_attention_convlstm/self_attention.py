from typing import Union, Tuple, Optional
import torch
from torch import nn

import sys

sys.path.append(".")
from common.constans import DEVICE, WeightsInitializer


class SelfAttentionWithConv2d(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(SelfAttentionWithConv2d, self).__init__()
        self.query_layer = nn.Conv2d(input_dim, hidden_dim, 1, device=DEVICE)
        self.key_layer = nn.Conv2d(input_dim, hidden_dim, 1, device=DEVICE)
        self.value_layer = nn.Conv2d(input_dim, input_dim, 1, device=DEVICE)
        self.z_layer = nn.Conv2d(input_dim, input_dim, 1, device=DEVICE)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, h) -> torch.Tensor:
        batch_size, _, H, W = h.shape
        key = self.key_layer(h)
        query = self.query_layer(h)
        value = self.value_layer(h)

        key = key.view(batch_size, self.hidden_dim, H * W)
        query = query.view(batch_size, self.hidden_dim, H * W).transpose(1, 2)
        value = value.view(batch_size, self.input_dim, H * W)

        attention = torch.softmax(
            torch.bmm(query, key), dim=-1
        )  # the shape is (batch_size, H*W, H*W)

        z = torch.matmul(attention, value.permute(0, 2, 1))
        z = z.transpose(1, 2).view(batch_size, self.input_dim, H, W)

        new_h = self.z_layer(z) + h

        return new_h

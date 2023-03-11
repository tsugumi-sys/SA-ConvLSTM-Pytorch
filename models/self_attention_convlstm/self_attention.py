import sys
from typing import Tuple

import torch
from torch import nn

sys.path.append(".")
from common.constants import DEVICE  # noqa: E402


class SelfAttention(nn.Module):
    """Self-Attention module implementation."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(SelfAttention, self).__init__()
        self.query_h = nn.Conv2d(
            input_dim, hidden_dim, 1, padding="same", device=DEVICE
        )
        self.key_h = nn.Conv2d(input_dim, hidden_dim, 1, padding="same", device=DEVICE)
        self.value_h = nn.Conv2d(input_dim, input_dim, 1, padding="same", device=DEVICE)
        self.z = nn.Conv2d(input_dim, input_dim, 1, padding="same", device=DEVICE)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, h) -> Tuple:
        batch_size, _, H, W = h.shape
        k_h = self.key_h(h)
        q_h = self.query_h(h)
        v_h = self.value_h(h)

        k_h = k_h.view(batch_size, self.hidden_dim, H * W)
        q_h = q_h.view(batch_size, self.hidden_dim, H * W).transpose(1, 2)
        v_h = v_h.view(batch_size, self.input_dim, H * W)

        attention = torch.softmax(
            torch.bmm(q_h, k_h), dim=-1
        )  # the shape is (batch_size, H*W, H*W)

        new_h = torch.matmul(attention, v_h.permute(0, 2, 1))
        new_h = new_h.transpose(1, 2).view(batch_size, self.input_dim, H, W)
        new_h = self.z(new_h)

        return new_h, attention

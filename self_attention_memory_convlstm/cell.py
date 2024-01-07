from typing import Tuple, Union

import torch
from torch import nn

from core.constants import DEVICE, WeightsInitializer
from core.convlstm_cell import BaseConvLSTMCell
from self_attention_memory_convlstm.self_attention_memory import (
    SelfAttentionMemory,
)


class SAMConvLSTMCell(nn.Module):
    def __init__(
        self,
        attention_hidden_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        padding: Union[int, Tuple, str],
        activation: str,
        frame_size: Tuple,
        weights_initializer: WeightsInitializer = WeightsInitializer.Zeros,
    ) -> None:
        super().__init__()
        self.convlstm_cell = BaseConvLSTMCell(
            in_channels,
            out_channels,
            kernel_size,
            padding,
            activation,
            frame_size,
            weights_initializer,
        )
        self.attention_memory = SelfAttentionMemory(out_channels, attention_hidden_dims)

    def forward(
        self,
        X: torch.Tensor,
        prev_h: torch.Tensor,
        prev_cell: torch.Tensor,
        prev_memory: torch.Tensor,
    ) -> Tuple:
        new_h, new_cell = self.convlstm_cell(X, prev_h, prev_cell)
        new_h, new_memory, attention_h = self.attention_memory(new_h, prev_memory)
        return (
            new_h.to(DEVICE),
            new_cell.to(DEVICE),
            new_memory.to(DEVICE),
            attention_h.to(DEVICE),
        )

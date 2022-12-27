from typing import Tuple, Union, Optional
import sys

import torch
from torch import nn

sys.path.append(".")
from models.convlstm_cell.convlstm_cell import BaseConvLSTMCell
from models.self_attention_memory_convlstm.self_attention_memory_module import (
    SelfAttentionMemory,
)
from models.self_attention_convlstm.self_attention import SelfAttention
from common.constans import DEVICE, WeightsInitializer


class SAMConvLSTMCell(BaseConvLSTMCell):
    def __init__(
        self,
        attention_hidden_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        padding: Union[int, Tuple, str],
        activation: str,
        frame_size: Tuple,
        weights_initializer: Optional[str] = WeightsInitializer.Zeros.value,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding,
            activation,
            frame_size,
            weights_initializer,
        )

        self.attention_x = SelfAttention(in_channels, attention_hidden_dims)
        self.attention_memory = SelfAttentionMemory(out_channels, attention_hidden_dims)

    def forward(
        self,
        X: torch.Tensor,
        prev_h: torch.Tensor,
        prev_cell: torch.Tensor,
        prev_memory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        new_h, new_cell = self.convlstm_cell(X, prev_h, prev_cell)
        new_h, new_memory = self.attention_memory(new_h, prev_memory)
        return new_h.to(DEVICE), new_cell.to(DEVICE), new_memory.to(DEVICE)

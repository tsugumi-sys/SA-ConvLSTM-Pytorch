from typing import Tuple, Union, Optional
import sys

import torch
from torch import nn

sys.path.append(".")
from models.convlstm_cell.convlstm_cell import BaseConvLSTMCell
from models.self_attention_convlstm.self_attention import SelfAttention
from common.constans import DEVICE, WeightsInitializer


class SAConvLSTMCell(BaseConvLSTMCell):
    def __init__(
        self,
        attention_hidden_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        padding: Union[int, Tuple, str],
        activation: str,
        frame_size: Tuple,
        weights_initializer: Optional[str] = WeightsInitializer.Zeros,
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
        self.attention_h = SelfAttention(out_channels, attention_hidden_dims)

        self.attention_hidden_dims = attention_hidden_dims

    def forward(
        self, X: torch.Tensor, prev_h: torch.Tensor, prev_cell: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.attention_x(X)
        new_h, new_cell = self.convlstm_cell(X, prev_h, prev_cell)
        new_h = self.attention_h(new_h) + new_h
        return new_h, new_cell

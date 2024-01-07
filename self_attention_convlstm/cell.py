from typing import Tuple, Union

import torch

from core.constants import WeightsInitializer
from core.convlstm_cell import BaseConvLSTMCell
from self_attention_convlstm.self_attention import (
    SelfAttention,
)


class SAConvLSTMCell(BaseConvLSTMCell):
    """Base Self-Attention ConvLSTM cell implementation (Lin et al., 2020)."""

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

    def forward(
        self, X: torch.Tensor, prev_h: torch.Tensor, prev_cell: torch.Tensor
    ) -> Tuple:
        X, _ = self.attention_x(X)
        new_h, new_cell = self.convlstm_cell(X, prev_h, prev_cell)
        new_h, attention = self.attention_h(new_h)
        new_h += new_h
        return new_h, new_cell, attention

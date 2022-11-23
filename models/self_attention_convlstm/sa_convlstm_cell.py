from typing import Tuple, Union, Optional
import sys

import torch
from torch import nn

sys.path.apend(".")
from models.convlstm_cell.convlstm_cell import BaseConvLSTMCell
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
        self.attention_hidden_dims = attention_hidden_dims

    def forward(
        self, X: torch.Tensor, prev_h: torch.Tensor, prev_cell: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return

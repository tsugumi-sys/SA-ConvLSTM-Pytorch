from typing import Tuple, Union, Optional
import sys

import torch
from torch import nn

sys.path.append(".")
from models.self_attention_convlstm.sa_convlstm_cell import SAConvLSTMCell
from common.constans import DEVICE, WeightsInitializer


class SAConvLSTM(nn.Module):
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
        super(SAConvLSTM, self).__init__()

        self.sa_convlstm_cell = SAConvLSTMCell(
            attention_hidden_dims,
            in_channels,
            out_channels,
            kernel_size,
            padding,
            activation,
            frame_size,
            weights_initializer,
        )

        self.out_channels = out_channels

    def forward(
        self,
        X: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, _, seq_len, height, width = X.size()

        if h is None:
            h = torch.zeros(
                (batch_size, self.out_channels, height, width), device=DEVICE
            )

        if cell is None:
            cell = torch.zeros(
                (batch_size, self.out_channels, height, width), device=DEVICE
            )

        output = torch.zeros(
            (batch_size, self.out_channels, seq_len, height, width), device=DEVICE
        )

        for time_step in range(seq_len):
            h, cell = self.sa_convlstm_cell(X[:, :, time_step], h, cell)

            output[:, :, time_step] = h  # type: ignore

        return output

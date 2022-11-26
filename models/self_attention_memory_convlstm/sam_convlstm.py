from typing import Union, Tuple, Optional
import torch
from torch import nn

import sys


sys.path.append(".")
from models.self_attention_memory_convlstm.sam_convlstm_cell import SAMConvLSTMCell
from common.constans import WeightsInitializer, DEVICE


class SAMConvLSTM(nn.Module):
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
    ):
        super(SAMConvLSTM, self).__init__()
        self.sam_convlstm_cell = SAMConvLSTMCell(
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
        memory: Optional[torch.Tensor] = None,
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

        if memory is None:
            memory = torch.zeros(
                (batch_size, self.out_channels, height, width), device=DEVICE
            )

        output = torch.zeros(
            (batch_size, self.out_channels, seq_len, height, width), device=DEVICE
        )

        for time_step in range(seq_len):
            h, cell, memory = self.sam_convlstm_cell(
                X[:, :, time_step], h, cell, memory
            )

            output[:, :, time_step] = h  # type: ignore

        return output

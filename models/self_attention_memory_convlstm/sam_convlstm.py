import sys
from typing import Optional, Tuple, Union

import torch
from torch import nn

sys.path.append(".")
from common.constants import DEVICE, WeightsInitializer  # noqa: E402
from models.self_attention_memory_convlstm.sam_convlstm_cell import (  # noqa: E402
    SAMConvLSTMCell,
)


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

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.attention_scores = None

    def forward(
        self,
        X: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, _, seq_len, height, width = X.size()

        # NOTE: Cannot store all attention scores because of memory. So only store attention map of the center.
        # And the same attention score are applyed to each channels.
        self.attention_scores = torch.zeros(
            (batch_size, seq_len, height * width), device=DEVICE
        )

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
            h, cell, memory, attention_h = self.sam_convlstm_cell(
                X[:, :, time_step], h, cell, memory
            )

            output[:, :, time_step] = h  # type: ignore
            # Save attention maps of the center point because storing
            # the full `attention_h` is difficult because of the lot of memory usage.
            # `attention_h` shape is (batch_size, height*width, height*width)
            self.attention_scores[:, time_step] = attention_h[
                :, attention_h.size(0) // 2
            ]

        return output

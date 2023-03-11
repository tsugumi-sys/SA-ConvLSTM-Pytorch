import sys
from typing import Optional, Tuple, Union

import torch
from torch import nn

sys.path.append(".")
from common.constants import DEVICE, WeightsInitializer  # noqa: E402
from models.self_attention_convlstm.sa_convlstm_cell import SAConvLSTMCell  # noqa: E402


class SAConvLSTM(nn.Module):
    """Base Self-Attention ConvLSTM implementation (Lin et al., 2020)."""

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

        self.attention_scores = None
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(
        self,
        X: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
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

        output = torch.zeros(
            (batch_size, self.out_channels, seq_len, height, width), device=DEVICE
        )

        for time_step in range(seq_len):
            h, cell, attention = self.sa_convlstm_cell(X[:, :, time_step], h, cell)

            output[:, :, time_step] = h  # type: ignore
            self.attention_scores[:, time_step] = attention[
                :, attention.size(0) // 2
            ]  # attention shape is (batch_size, height*width, height*width)

        return output

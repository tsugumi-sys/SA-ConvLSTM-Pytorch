from typing import Optional, Tuple, TypedDict, Union

import torch
from torch import nn

from convlstm.model import ConvLSTMParams
from core.constants import DEVICE, WeightsInitializer
from self_attention_convlstm.cell import SAConvLSTMCell


class SAConvLSTMParams(TypedDict):
    attention_hidden_dims: int
    convlstm_params: ConvLSTMParams


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
        weights_initializer: WeightsInitializer = WeightsInitializer.Zeros,
    ) -> None:
        super().__init__()

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

        self.in_channels = in_channels
        self.out_channels = out_channels
        self._attention_scores: Optional[torch.Tensor] = None

    @property
    def attention_scores(self) -> Optional[torch.Tensor]:
        return self._attention_scores

    def forward(
        self,
        X: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, _, seq_len, height, width = X.size()

        # NOTE: Cannot store all attention scores because of memory. So only store attention map of the center.
        # And the same attention score are applied to each channels.
        self._attention_scores = torch.zeros(
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
            self._attention_scores[:, time_step] = attention[
                :, attention.size(0) // 2
            ]  # attention shape is (batch_size, height*width, height*width)

        return output

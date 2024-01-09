from typing import Optional, TypedDict

import torch
from torch import nn

from convlstm.model import ConvLSTMParams
from core.constants import DEVICE
from self_attention_convlstm.cell import SAConvLSTMCell


class SAConvLSTMParams(TypedDict):
    attention_hidden_dims: int
    convlstm_params: ConvLSTMParams


class SAConvLSTM(nn.Module):
    """Base Self-Attention ConvLSTM implementation (Lin et al., 2020)."""

    def __init__(
        self, attention_hidden_dims: int, convlstm_params: ConvLSTMParams
    ) -> None:
        super().__init__()
        self.attention_hidden_dims = attention_hidden_dims
        self.in_channels = convlstm_params["in_channels"]
        self.kernel_size = convlstm_params["kernel_size"]
        self.padding = convlstm_params["padding"]
        self.activation = convlstm_params["activation"]
        self.frame_size = convlstm_params["frame_size"]
        self.out_channels = convlstm_params["out_channels"]
        self.weights_initializer = convlstm_params["weights_initializer"]

        self.sa_convlstm_cell = SAConvLSTMCell(
            attention_hidden_dims=attention_hidden_dims, **convlstm_params
        )

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

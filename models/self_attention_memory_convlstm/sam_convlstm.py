from typing import Union, Tuple, Optional
import torch
from torch import nn

import sys

from train.src.models.self_attention_convlstm.self_attention_convlstm import (
    SelfAttentionWithConv2d,
)

sys.path.append("..")
from train.src.config import DEVICE, WeightsInitializer
from train.src.models.convlstm_cell.interface import ConvLSTMCellInterface
from train.src.models.self_attention_memory_convlstm.self_attention_meomry_module import (
    SelfAttentionMemoryWithConv2d,
)


class SelfAttentionMemoryConvLSTMCell(ConvLSTMCellInterface):
    def __init__(
        self,
        attention_layer_hidden_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        padding: Union[int, Tuple, str],
        activation: str,
        frame_size: Tuple,
        weights_initializer: Optional[str] = WeightsInitializer.Zeros.value,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding,
            activation,
            frame_size,
            weights_initializer,
        )
        self.self_attention_memory = SelfAttentionMemoryWithConv2d(
            out_channels, attention_layer_hidden_dims
        )
        self.self_attention = SelfAttentionWithConv2d(
            in_channels, attention_layer_hidden_dims
        )

    def forward(
        self,
        X: torch.Tensor,
        prev_h: torch.Tensor,
        prev_cell: torch.Tensor,
        prev_m: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X = self.self_attention(X)

        conv_output = self.conv(torch.cat([X, prev_h], dim=1))

        i_conv, f_conv, c_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * prev_cell)
        forget_gate = torch.sigmoid(f_conv + self.W_cf * prev_cell)

        new_cell = forget_gate * prev_cell + input_gate * self.activation(c_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * new_cell)

        new_h = output_gate * self.activation(new_cell)
        new_h, new_m = self.self_attention_memory(new_h, prev_m)
        return new_h.to(DEVICE), new_cell.to(DEVICE), new_m.to(DEVICE)


class SelfAttentionMemoryConvLSTM(nn.Module):
    def __init__(
        self,
        attention_layer_hidden_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        padding: Union[int, Tuple, str],
        activation: str,
        frame_size: Tuple,
        weights_initializer: Optional[str] = WeightsInitializer.Zeros.value,
    ):
        super(SelfAttentionMemoryConvLSTM, self).__init__()
        self.out_channels = out_channels
        self.sam_convlstm_cell = SelfAttentionMemoryConvLSTMCell(
            attention_layer_hidden_dims,
            in_channels,
            out_channels,
            kernel_size,
            padding,
            activation,
            frame_size,
            weights_initializer,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros(
            (batch_size, self.out_channels, seq_len, height, width)
        ).to(DEVICE)

        # Initialize hidden state
        H = torch.zeros((batch_size, self.out_channels, height, width)).to(DEVICE)

        # Initialize cell input
        C = torch.zeros((batch_size, self.out_channels, height, width)).to(DEVICE)

        # Initialize memory module
        M = torch.zeros((batch_size, self.out_channels, height, width)).to(DEVICE)

        for time_step in range(seq_len):
            H, C, M = self.sam_convlstm_cell(X[:, :, time_step], H, C, M)
            output[:, :, time_step] = H

        return output

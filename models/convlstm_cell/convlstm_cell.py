import sys
from typing import Optional, Tuple, Union

import torch
from torch import nn

sys.path.append(".")
from common.constants import DEVICE, WeightsInitializer  # noqa: E402


class BaseConvLSTMCell(nn.Module):
    """The ConvLSTM Cell implementation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        padding: Union[int, Tuple, str],
        activation: str,
        frame_size: Tuple,
        weights_initializer: Optional[str] = WeightsInitializer.Zeros,
    ) -> None:
        """

        Args:
            in_channels (int): Number of channels of input tensor.
            out_channels (int): Number of channels of output tensor
            kernel_size (Union[int, Tuple]): Size of the convolution kernel.
            padding (padding (Union[int, Tuple, str]): 'same', 'valid' or (int, int)
            activation (str): Name of activation function
            frame_size (Tuple): height and width
        """
        super(BaseConvLSTMCell, self).__init__()

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "leakyRelu":
            self.activation = torch.nn.LeakyReLU()
        elif activation == "sigmoid":
            self.activation = torch.sigmoid
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels,
            out_channels=4 * out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.W_ci = nn.parameter.Parameter(
            torch.zeros(out_channels, *frame_size, dtype=torch.float)
        ).to(DEVICE)
        self.W_co = nn.parameter.Parameter(
            torch.zeros(out_channels, *frame_size, dtype=torch.float)
        ).to(DEVICE)
        self.W_cf = nn.parameter.Parameter(
            torch.zeros(out_channels, *frame_size, dtype=torch.float)
        ).to(DEVICE)

        if weights_initializer == WeightsInitializer.Zeros:
            pass
        elif weights_initializer == WeightsInitializer.He:
            nn.init.kaiming_normal_(self.W_ci, mode="fan_in", nonlinearity="leaky_relu")
            nn.init.kaiming_normal_(self.W_co, mode="fan_in", nonlinearity="leaky_relu")
            nn.init.kaiming_normal_(self.W_cf, mode="fan_in", nonlinearity="leaky_relu")
        elif weights_initializer == WeightsInitializer.Xavier:
            nn.init.xavier_normal_(self.W_ci, gain=1.0)
            nn.init.xavier_normal_(self.W_co, gain=1.0)
            nn.init.xavier_normal_(self.W_cf, gain=1.0)
        else:
            raise ValueError(f"Invlaid weights Initializer: {weights_initializer}")

    def forward(
        self, X: torch.Tensor, prev_h: torch.Tensor, prev_cell: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        new_h, new_cell = self.convlstm_cell(X, prev_h, prev_cell)
        return new_h, new_cell

    def convlstm_cell(
        self, X: torch.Tensor, prev_h: torch.Tensor, prev_cell: torch.Tensor
    ):
        """ConvLSTM cell calclation.

        Args:
            X (torch.Tensor): input data.
            h_prev (torch.Tensor): previous hidden state.
            c_prev (torch.Tensor): previous cell state.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (current_hidden_state, current_cell_state)
        """
        conv_output = self.conv(torch.cat([X, prev_h], dim=1))

        i_conv, f_conv, c_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * prev_cell)
        forget_gate = torch.sigmoid(f_conv + self.W_cf * prev_cell)

        # Current cell output (state)
        C = forget_gate * prev_cell + input_gate * self.activation(c_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C)

        # Current hidden state
        H = output_gate * self.activation(C)

        return H.to(DEVICE), C.to(DEVICE)

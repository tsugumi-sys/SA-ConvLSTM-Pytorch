from typing import NotRequired, Optional, Tuple, TypedDict, Union

import torch
from torch import nn

from core.constants import DEVICE, WeightsInitializer
from core.convlstm_cell import BaseConvLSTMCell


class ConvLSTMParams(TypedDict):
    in_channels: int
    out_channels: int
    kernel_size: Union[int, Tuple]
    padding: Union[int, Tuple, str]
    activation: str
    frame_size: Tuple[int, int]
    weights_initializer: NotRequired[str]


class ConvLSTM(nn.Module):
    """The ConvLSTM implementation (Shi et al., 2015)."""

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
            in_channels (int): input channel.
            out_channels (int): output channel.
            kernel_size (Union[int, Tuple]): The size of convolution kernel.
            padding (Union[int, Tuple, str]): Should be in ['same', 'valid' or (int, int)]
            activation (str): Name of activation function.
            frame_size (Tuple): height and width.
            weights_initializer: Optional[str]: Weight initializers of ['zeros', 'he', 'xavier'].
        """
        super().__init__()

        self.ConvLSTMCell = BaseConvLSTMCell(
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
        """

        Args:
            X (torch.Tensor): tensor with the shape of (batch_size, num_channels, seq_len, height, width)

        Returns:
            torch.Tensor: tensor with the same shape of X
        """
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros(
            (batch_size, self.out_channels, seq_len, height, width)
        ).to(DEVICE)

        # Initialize hidden state
        h = torch.zeros((batch_size, self.out_channels, height, width)).to(DEVICE)

        # Initialize cell input
        cell = torch.zeros((batch_size, self.out_channels, height, width)).to(DEVICE)

        # Unroll over time steps
        for time_step in range(seq_len):
            h, cell = self.ConvLSTMCell(X[:, :, time_step], h, cell)

            output[:, :, time_step] = h  # type: ignore

        return output

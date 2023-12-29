from typing import Optional, Tuple, Union

import torch
from torch import nn

from core.constants import DEVICE, WeightsInitializer
from core.convlstm_cell import BaseConvLSTMCell


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


if __name__ == "__main__":
    input_X = torch.rand((5, 6, 3, 16, 16), dtype=torch.float)
    convlstm = ConvLSTM(
        in_channels=6,
        out_channels=15,
        kernel_size=3,
        padding=1,
        activation="relu",
        frame_size=(16, 16),
    )
    y = convlstm.forward(input_X)
    print(y.shape)

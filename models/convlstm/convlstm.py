from typing import Tuple, Union, Optional
import sys

import torch
from torch import nn

sys.path.append("..")
from train.src.convlstmcell import ConvLSTMCell
from train.src.config import DEVICE, WeightsInitializer


class ConvLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        padding: Union[int, Tuple, str],
        activation: str,
        frame_size: Tuple,
        weights_initializer: Optional[str] = WeightsInitializer.Zeros.value,
    ) -> None:
        """Initialize ConsLTSM

        Args:
            in_channels (int): [input channel]
            out_channels (int): [output channel]
            kernel_size (Union[int, Tuple]): [The size of convolution kernel.]
            padding (Union[int, Tuple, str]): ['same', 'valid' or (int, int)]): ['same', 'valid' or (int, int)]
            activation (str): [Name of activation function]
            frame_size (Tuple): [height and width]
        """
        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels
        self.ConvLSTMCell = ConvLSTMCell(in_channels, out_channels, kernel_size, padding, activation, frame_size, weights_initializer)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """forward calculation of ConvLSTM

        Args:
            X (torch.Tensor): [tensor with the shape of (batch_size, num_channels, seq_len, height, width)]

        Returns:
            torch.Tensor: [tensor with the same shape of X]
        """
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros((batch_size, self.out_channels, seq_len, height, width)).to(DEVICE)

        # Initialize hidden state
        H = torch.zeros((batch_size, self.out_channels, height, width)).to(DEVICE)

        # Initialize cell input
        C = torch.zeros((batch_size, self.out_channels, height, width)).to(DEVICE)

        # Unroll over time steps
        for time_step in range(seq_len):
            H, C = self.ConvLSTMCell(X[:, :, time_step], H, C)

            output[:, :, time_step] = H

        return output


if __name__ == "__main__":
    input_X = torch.rand((5, 6, 3, 16, 16), dtype=torch.float)
    convlstm = ConvLSTM(in_channels=6, out_channels=15, kernel_size=3, padding=1, activation="relu", frame_size=(16, 16))
    y = convlstm.forward(input_X)
    print(y.shape)

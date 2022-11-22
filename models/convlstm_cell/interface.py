from typing import Tuple, Union, Optional
import sys

import torch
from torch import nn

sys.path.append("..")
from train.src.config import DEVICE, WeightsInitializer


class ConvLSTMCellInterface(nn.Module):
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
        """[Initialize ConvLSTMCell]
        Args:
            in_channels (int): [Number of channels of input tensor.]
            out_channels (int): [Number of channels of output tensor]
            kernel_size (Union[int, Tuple]): [Size of the convolution kernel.]
            padding (padding (Union[int, Tuple, str]): ['same', 'valid' or (int, int)]): ['same', 'valid' or (int, int)]
            activation (str): [Name of activation function]
            frame_size (Tuple): [height and width]
        """
        super(ConvLSTMCellInterface, self).__init__()
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "sigmoid":
            self.activation = torch.sigmoid

        self.conv = nn.Conv2d(in_channels=in_channels + out_channels, out_channels=4 * out_channels, kernel_size=kernel_size, padding=padding,)

        # Initialize weights for Hadamard Products. It is equal to zeros initialize.
        # [NOTE] When dtype isn't set correctly, predict value comes to be all nan.
        self.W_ci = nn.parameter.Parameter(torch.zeros(out_channels, *frame_size, dtype=torch.float)).to(DEVICE)
        self.W_co = nn.parameter.Parameter(torch.zeros(out_channels, *frame_size, dtype=torch.float)).to(DEVICE)
        self.W_cf = nn.parameter.Parameter(torch.zeros(out_channels, *frame_size, dtype=torch.float)).to(DEVICE)

        if weights_initializer == WeightsInitializer.Zeros:
            pass
        elif weights_initializer == WeightsInitializer.He:
            nn.init.kaiming_normal_(self.W_ci, mode="fan_in", nonlinearity="relu")
            nn.init.kaiming_normal_(self.W_co, mode="fan_in", nonlinearity="relu")
            nn.init.kaiming_normal_(self.W_cf, mode="fan_in", nonlinearity="relu")

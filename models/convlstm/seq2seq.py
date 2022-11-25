from typing import Tuple, Union, Optional
import sys

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F

sys.path.append(".")
from models.convlstm.convlstm import ConvLSTM
from common.constans import WeightsInitializer


class Seq2Seq(nn.Module):
    def __init__(
        self,
        num_channels: int,
        kernel_size: Union[int, Tuple],
        num_kernels: int,
        padding: Union[int, Tuple, str],
        activation: str,
        frame_size: Tuple,
        num_layers: int,
        weights_initializer: Optional[str] = WeightsInitializer.Zeros.value,
        return_sequences: bool = False,
    ) -> None:
        """Initialize SeqtoSeq

        Args:
            num_channels (int): [Number of input channels]
            kernel_size (int): [kernel size]
            num_kernels (int): [Number of kernels]
            padding (Union[str, Tuple]): ['same', 'valid' or (int, int)]
            activation (str): [the name of activation function]
            frame_size (Tuple): [height and width]
            num_layers (int): [the number of layers]
        """
        super(Seq2Seq, self).__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.padding = padding
        self.activation = activation
        self.frame_size = frame_size
        self.num_layers = num_layers
        self.weights_initializer = weights_initializer
        self.return_sequences = return_sequences

        self.sequencial = nn.Sequential()

        # Add first layer (Different in_channels than the rest)
        self.sequencial.add_module(
            "convlstm1",
            ConvLSTM(
                in_channels=num_channels,
                out_channels=num_kernels,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation,
                frame_size=frame_size,
                weights_initializer=weights_initializer,
            ),
        )

        self.sequencial.add_module(
            "layernorm1", nn.LayerNorm([num_kernels, 3, *self.frame_size])
        )

        # Add the rest of the layers
        for layer_idx in range(2, num_layers + 1):
            self.sequencial.add_module(
                f"convlstm{layer_idx}",
                ConvLSTM(
                    in_channels=num_kernels,
                    out_channels=num_kernels,
                    kernel_size=kernel_size,
                    padding=padding,
                    activation=activation,
                    frame_size=frame_size,
                    weights_initializer=weights_initializer,
                ),
            )

            self.sequencial.add_module(
                f"layernorm{layer_idx}",
                nn.LayerNorm([num_kernels, 3, *self.frame_size]),
            )

        self.sequencial.add_module(
            "convlstm_last",
            ConvLSTM(
                in_channels=num_kernels,
                out_channels=num_channels,
                kernel_size=kernel_size,
                padding=padding,
                activation="sigmoid",
                frame_size=frame_size,
                weights_initializer=weights_initializer,
            ),
        )

    def forward(self, X: torch.Tensor):
        # Forward propagation through all the layers
        output = self.sequencial(X)

        if self.return_sequences is True:
            return output

        return output[:, :, -1:, ...]


if __name__ == "__main__":
    input_X = torch.rand((5, 6, 3, 16, 16), dtype=torch.float)
    convlstm = Seq2Seq(
        num_channels=6,
        kernel_size=3,
        num_kernels=4,
        padding="same",
        activation="relu",
        frame_size=(16, 16),
        num_layers=3,
    )
    y = convlstm.forward(input_X)
    print(y.shape)

from typing import NotRequired, Optional, Tuple, TypedDict, Union

import torch
from torch import nn

from convlstm.model import ConvLSTM, ConvLSTMParams
from core.constants import WeightsInitializer


class Seq2SeqParams(TypedDict):
    num_layers: int
    input_seq_length: int
    return_sequences: NotRequired[bool]
    convlstm_params: ConvLSTMParams


class Seq2Seq(nn.Module):
    """The sequence to sequence model implementation using ConvLSTM."""

    def __init__(
        self,
        num_channels: int,
        kernel_size: Union[int, Tuple],
        num_kernels: int,
        padding: Union[int, Tuple, str],
        activation: str,
        frame_size: Tuple,
        num_layers: int,
        input_seq_length: int,
        out_channels: Optional[int] = None,
        weights_initializer: WeightsInitializer = WeightsInitializer.Zeros,
        return_sequences: bool = False,
    ) -> None:
        """

        Args:
            num_channels (int): Number of input channels.
            kernel_size (int): kernel size.
            num_kernels (int): Number of kernels.
            padding (Union[str, Tuple]): 'same', 'valid' or (int, int)
            activation (str): the name of activation function.
            frame_size (Tuple): height and width.
            num_layers (int): the number of layers.
        """
        super().__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.padding = padding
        self.activation = activation
        self.frame_size = frame_size
        self.num_layers = num_layers
        self.input_seq_length = input_seq_length
        self.out_channels = out_channels if out_channels is not None else num_channels
        self.weights_initializer = weights_initializer
        self.return_sequences = return_sequences

        self.sequential = nn.Sequential()

        # Add first layer (Different in_channels than the rest)
        self.sequential.add_module(
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

        self.sequential.add_module(
            "layernorm1",
            nn.LayerNorm([num_kernels, self.input_seq_length, *self.frame_size]),
        )

        # Add the rest of the layers
        for layer_idx in range(2, num_layers + 1):
            self.sequential.add_module(
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

            self.sequential.add_module(
                f"layernorm{layer_idx}",
                nn.LayerNorm([num_kernels, self.input_seq_length, *self.frame_size]),
            )

        self.sequential.add_module(
            "conv3d",
            nn.Conv3d(
                in_channels=self.num_kernels,
                out_channels=self.out_channels,
                kernel_size=(3, 3, 3),
                padding="same",
            ),
        )

        self.sequential.add_module("sigmoid", nn.Sigmoid())

    def forward(self, X: torch.Tensor):
        # Forward propagation through all the layers
        output = self.sequential(X)

        if self.return_sequences is True:
            return output

        return output[:, :, -1:, ...]

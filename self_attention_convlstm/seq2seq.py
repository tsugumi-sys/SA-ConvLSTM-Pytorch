from typing import NotRequired, TypedDict

import torch
from torch import nn

from convlstm.model import ConvLSTMParams
from self_attention_convlstm.model import SAConvLSTM


class SASeq2SeqParams(TypedDict):
    attention_hidden_dims: int
    input_seq_length: int
    num_layers: int
    num_kernels: int
    return_sequences: NotRequired[bool]
    convlstm_params: ConvLSTMParams


class SASeq2Seq(nn.Module):
    """The sequence to sequence model implementation using Base Self-Attention ConvLSTM."""

    def __init__(
        self,
        attention_hidden_dims: int,
        input_seq_length: int,
        num_layers: int,
        num_kernels: int,
        convlstm_params: ConvLSTMParams,
        return_sequences: bool = False,
    ) -> None:
        """

        Args:
            attention_hidden_dims (int): Number of attention hidden layers.
            input_seq_length (int): Number of input frames.
            num_layers (int): Number of ConvLSTM layers.
            num_kernels (int): Number of kernels.
            return_sequences (int): If True, the model predict the next frames that is the same length of inputs. If False, the model predicts only one next frame.
            convlstm_params (ConvLSTMParams): Parameters for ConvLSTM module.
        """
        super().__init__()
        self.attention_hidden_dims = attention_hidden_dims
        self.input_seq_length = input_seq_length
        self.num_layers = num_layers
        self.num_kernels = num_kernels
        self.return_sequences = return_sequences
        self.in_channels = convlstm_params["in_channels"]
        self.kernel_size = convlstm_params["kernel_size"]
        self.padding = convlstm_params["padding"]
        self.activation = convlstm_params["activation"]
        self.frame_size = convlstm_params["frame_size"]
        self.out_channels = convlstm_params["out_channels"]
        self.weights_initializer = convlstm_params["weights_initializer"]
        self.sequential = nn.Sequential()

        # Add first layer (Different in_channels than the rest)
        self.sequential.add_module(
            "sa_convlstm1",
            SAConvLSTM(
                attention_hidden_dims=self.attention_hidden_dims,
                convlstm_params={
                    "in_channels": self.in_channels,
                    "out_channels": self.num_kernels,
                    "kernel_size": self.kernel_size,
                    "padding": self.padding,
                    "activation": self.activation,
                    "frame_size": self.frame_size,
                    "weights_initializer": self.weights_initializer,
                },
            ),
        )

        self.sequential.add_module(
            "layernorm1",
            nn.LayerNorm([self.num_kernels, self.input_seq_length, *self.frame_size]),
        )

        # Add the rest of the layers
        for layer_idx in range(2, self.num_layers + 1):
            self.sequential.add_module(
                f"sa_convlstm{layer_idx}",
                SAConvLSTM(
                    attention_hidden_dims=self.attention_hidden_dims,
                    convlstm_params={
                        "in_channels": self.num_kernels,
                        "out_channels": self.num_kernels,
                        "kernel_size": self.kernel_size,
                        "padding": self.padding,
                        "activation": self.activation,
                        "frame_size": self.frame_size,
                        "weights_initializer": self.weights_initializer,
                    },
                ),
            )

            self.sequential.add_module(
                f"layernorm{layer_idx}",
                nn.LayerNorm(
                    [self.num_kernels, self.input_seq_length, *self.frame_size]
                ),
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

    def get_attention_maps(self):
        # get all sa_convlstm module
        sa_convlstm_modules = [
            (name, module)
            for name, module in self.named_modules()
            if module.__class__.__name__ == "SAConvLSTM"
        ]
        return {
            name: module.attention_scores for name, module in sa_convlstm_modules
        }  # attention scores shape is (batch_size, seq_length, height * width)

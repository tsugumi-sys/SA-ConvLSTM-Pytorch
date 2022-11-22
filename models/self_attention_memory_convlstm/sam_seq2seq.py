from typing import Union, Tuple, Optional
import torch
from torch import nn

import sys

sys.path.append("..")
from train.src.config import DEVICE, WeightsInitializer
from train.src.models.convlstm_cell.interface import ConvLSTMCellInterface
from train.src.models.self_attention_memory_convlstm.sam_convlstm import SelfAttentionMemoryConvLSTM


class SAMSeq2Seq(nn.Module):
    def __init__(
        self,
        attention_layer_hidden_dims: int,
        num_channels: int,
        kernel_size: Union[int, Tuple],
        num_kernels: int,
        padding: Union[int, Tuple, str],
        activation: str,
        frame_size: Tuple,
        num_layers: int,
        input_seq_length: int,
        prediction_seq_length: int,
        out_channels: Optional[int] = None,
        weights_initializer: Optional[str] = WeightsInitializer.Zeros.value,
        return_sequences: bool = False,
    ):
        super(SAMSeq2Seq, self).__init__()
        self.attention_layer_hidden_dims = attention_layer_hidden_dims
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.padding = padding
        self.activation = activation
        self.frame_size = frame_size
        self.num_layers = num_layers
        self.input_seq_length = input_seq_length
        self.prediction_seq_length = prediction_seq_length
        self.out_channels = out_channels
        self.weights_initializer = weights_initializer
        self.return_sequences = return_sequences

        self.sequential = nn.Sequential()

        self.sequential.add_module(
            "sa-convlstm1",
            SelfAttentionMemoryConvLSTM(
                attention_layer_hidden_dims=self.attention_layer_hidden_dims,
                in_channels=self.num_channels,
                out_channels=self.num_kernels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                activation=self.activation,
                frame_size=self.frame_size,
                weights_initializer=self.weights_initializer,
            ),
        )

        self.sequential.add_module("batchnorm0", nn.BatchNorm3d(num_features=self.num_kernels))

        self.sequential.add_module(
            "convlstm2",
            SelfAttentionMemoryConvLSTM(
                attention_layer_hidden_dims=self.attention_layer_hidden_dims,
                in_channels=self.num_kernels,
                out_channels=self.num_channels if self.out_channels is None else self.out_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                activation="sigmoid",
                frame_size=self.frame_size,
                weights_initializer=self.weights_initializer,
            ),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output = self.sequential(X)

        if self.return_sequences is True:
            return output

        return output[:, :, -1:, :, :]

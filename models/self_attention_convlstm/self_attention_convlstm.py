from typing import Union, Tuple, Optional
import torch
from torch import nn

import sys

sys.path.append("..")
from train.src.config import DEVICE, WeightsInitializer
from train.src.models.convlstm_cell.interface import ConvLSTMCellInterface


class SelfAttentionWithConv2d(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(SelfAttentionWithConv2d, self).__init__()
        self.query_layer = nn.Conv2d(input_dim, hidden_dim, 1, device=DEVICE)
        self.key_layer = nn.Conv2d(input_dim, hidden_dim, 1, device=DEVICE)
        self.value_layer = nn.Conv2d(input_dim, input_dim, 1, device=DEVICE)
        self.z_layer = nn.Conv2d(input_dim, input_dim, 1, device=DEVICE)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, h):
        batch_size, _, H, W = h.shape
        key = self.key_layer(h)
        query = self.query_layer(h)
        value = self.value_layer(h)

        key = key.view(batch_size, self.hidden_dim, H * W)
        query = query.view(batch_size, self.hidden_dim, H * W).transpose(1, 2)
        value = value.view(batch_size, self.input_dim, H * W)

        attention = torch.softmax(torch.bmm(query, key), dim=-1)  # the shape is (batch_size, H*W, H*W)

        z = torch.matmul(attention, value.permute(0, 2, 1))
        z = z.transpose(1, 2).view(batch_size, self.input_dim, H, W)

        new_h = self.z_layer(z) + h

        return new_h


class SelfAttentionConvLSTMCell(ConvLSTMCellInterface):
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
        super().__init__(in_channels, out_channels, kernel_size, padding, activation, frame_size, weights_initializer)
        self.attention_h = SelfAttentionWithConv2d(out_channels, attention_layer_hidden_dims)
        self.attention_x = SelfAttentionWithConv2d(in_channels, attention_layer_hidden_dims)

    def forward(self, X: torch.Tensor, prev_h: torch.Tensor, prev_cell: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.attention_x(X)

        conv_output = self.conv(torch.cat([X, prev_h], dim=1))

        i_conv, f_conv, c_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * prev_cell)
        forget_gate = torch.sigmoid(f_conv + self.W_cf * prev_cell)

        new_cell = forget_gate * prev_cell + input_gate * self.activation(c_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * new_cell)

        new_h = output_gate * self.activation(new_cell)
        new_h = self.attention_h(new_h)

        return new_h.to(DEVICE), new_cell.to(DEVICE)


class SelfAttentionConvLSTM(nn.Module):
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
        super(SelfAttentionConvLSTM, self).__init__()
        self.out_channels = out_channels
        self.convlstm_cell = SelfAttentionConvLSTMCell(
            attention_layer_hidden_dims, in_channels, out_channels, kernel_size, padding, activation, frame_size, weights_initializer
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros((batch_size, self.out_channels, seq_len, height, width)).to(DEVICE)

        # Initialize hidden state
        H = torch.zeros((batch_size, self.out_channels, height, width)).to(DEVICE)

        # Initialize cell input
        C = torch.zeros((batch_size, self.out_channels, height, width)).to(DEVICE)

        for time_step in range(seq_len):
            H, C = self.convlstm_cell(X[:, :, time_step], H, C)
            output[:, :, time_step] = H

        return output


class SelfAttentionSeq2Seq(nn.Module):
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
        super(SelfAttentionSeq2Seq, self).__init__()
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
            SelfAttentionConvLSTM(
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
            SelfAttentionConvLSTM(
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


if __name__ == "__main__":
    input_X = torch.rand((5, 3, 6, 50, 50), dtype=torch.float).to(DEVICE)
    convlstm = (
        SelfAttentionSeq2Seq(
            attention_layer_hidden_dims=1,
            num_channels=3,
            kernel_size=3,
            num_kernels=32,
            padding="same",
            activation="relu",
            frame_size=(50, 50),
            num_layers=1,
            input_seq_length=6,
            prediction_seq_length=6,
            return_sequences=True,
        )
        .to(DEVICE)
        .to(dtype=torch.float)
    )
    # convlstm = (
    #     SelfAttentionConvLSTM(attention_layer_hidden_dims=1, in_channels=3, out_channels=3, kernel_size=3, padding=1, activation="relu", frame_size=(50, 50))
    #     .to(DEVICE)
    #     .to(torch.float)
    # )
    y = convlstm.forward(input_X)
    print(y.shape)

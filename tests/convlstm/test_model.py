import torch

from convlstm.model import ConvLSTM, ConvLSTMParams
from core.constants import DEVICE


def test_ConvLSTM():
    model_params: ConvLSTMParams = {
        "in_channels": 1,
        "out_channels": 1,
        "kernel_size": 3,
        "padding": 1,
        "activation": "relu",
        "frame_size": (8, 8),
    }
    model = ConvLSTM(**model_params)
    output = model(torch.rand((2, 1, 3, 8, 8), dtype=torch.float, device=DEVICE))
    assert output.size() == (2, 1, 3, 8, 8)

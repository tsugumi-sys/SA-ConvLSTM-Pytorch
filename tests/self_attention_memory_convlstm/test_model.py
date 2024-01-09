import torch

from core.constants import DEVICE, WeightsInitializer
from self_attention_memory_convlstm.model import SAMConvLSTM, SAMConvLSTMParams


def test_SAMConvLSTM():
    model_params: SAMConvLSTMParams = {
        "attention_hidden_dims": 1,
        "convlstm_params": {
            "in_channels": 1,
            "out_channels": 1,
            "kernel_size": 3,
            "padding": 1,
            "activation": "relu",
            "frame_size": (8, 8),
            "weights_initializer": WeightsInitializer.He,
        },
    }

    model = SAMConvLSTM(**model_params)
    output = model(torch.rand((2, 1, 3, 8, 8), dtype=torch.float, device=DEVICE))
    assert output.size() == (2, 1, 3, 8, 8)

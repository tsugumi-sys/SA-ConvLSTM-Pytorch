from typing import Tuple

import pytest
import torch

from convlstm.seq2seq import Seq2Seq, Seq2SeqParams
from core.constants import DEVICE, WeightsInitializer


@pytest.mark.parametrize(
    "return_sequences, expected_output_size",
    [(True, (2, 1, 2, 8, 8)), (False, (2, 1, 1, 8, 8))],
)
def test_seq2seq(return_sequences: bool, expected_output_size: Tuple):
    model_params: Seq2SeqParams = {
        "input_seq_length": 2,
        "num_layers": 2,
        "num_kernels": 4,
        "return_sequences": return_sequences,
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
    model = Seq2Seq(**model_params).to(DEVICE).to(torch.float)
    output = model(torch.rand((2, 1, 2, 8, 8), dtype=torch.float, device=DEVICE))
    assert output.size() == expected_output_size


def test_seq2seq_label_seq_length():
    # test if `label_seq_length` is less than the number of frames of the given datasaet.
    label_seq_length = 2
    model_params: Seq2SeqParams = {
        "input_seq_length": 4,
        "label_seq_length": label_seq_length,
        "num_layers": 2,
        "num_kernels": 4,
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
    model = Seq2Seq(**model_params).to(DEVICE).to(torch.float)
    output = model(torch.rand((2, 1, 4, 8, 8), dtype=torch.float, device=DEVICE))
    assert output.size() == (2, 1, label_seq_length, 8, 8)

    # test if `label_seq_length` is more than the number of frames of the given dataset.
    label_seq_length = 5
    model_params: Seq2SeqParams = {
        "input_seq_length": 4,
        "label_seq_length": label_seq_length,
        "num_layers": 2,
        "num_kernels": 4,
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
    model = Seq2Seq(**model_params).to(DEVICE).to(torch.float)
    output = model(torch.rand((2, 1, 4, 8, 8), dtype=torch.float, device=DEVICE))
    # the output has the same frames as the given dataset.
    assert output.size() == (2, 1, 4, 8, 8)

    # test if both `label_seq_length` and `return_sequences` are given.
    model_params: Seq2SeqParams = {
        "input_seq_length": 4,
        "label_seq_length": 3,
        "num_layers": 2,
        "num_kernels": 4,
        "return_sequences": True,
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
    model = Seq2Seq(**model_params).to(DEVICE).to(torch.float)
    output = model(torch.rand((2, 1, 4, 8, 8), dtype=torch.float, device=DEVICE))
    # the priority of `return_sequences` is higher than that of `label_seq_length`.
    assert output.size() == (2, 1, 4, 8, 8)

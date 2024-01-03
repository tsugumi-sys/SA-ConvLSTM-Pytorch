from typing import Tuple

import pytest
import torch

from core.constants import DEVICE
from self_attention_convlstm.seq2seq import SASeq2Seq


@pytest.mark.parametrize(
    "return_sequences, expected_output_size",
    [(True, (2, 1, 2, 8, 8)), (False, (2, 1, 1, 8, 8))],
)
def test_seq2seq(return_sequences: bool, expected_output_size: Tuple):
    model = (
        SASeq2Seq(
            attention_hidden_dims=1,
            num_channels=1,
            kernel_size=3,
            num_kernels=4,
            padding="same",
            activation="relu",
            frame_size=(8, 8),
            num_layers=2,
            input_seq_length=2,
            return_sequences=return_sequences,
        )
        .to(DEVICE)
        .to(torch.float)
    )
    output = model(torch.rand((2, 1, 2, 8, 8), dtype=torch.float, device=DEVICE))
    assert output.size() == expected_output_size

from typing import Tuple

import pytest
import torch

from core.constants import DEVICE
from tests.test_model.model import TestModel


@pytest.mark.parametrize(
    "return_sequences, expected_output_size",
    [(True, (5, 6, 3, 16, 16)), (False, (5, 6, 1, 16, 16))],
)
def test_TestModel_successfully_build(
    return_sequences: bool, expected_output_size: Tuple
):
    # return_sequence
    model = TestModel(return_sequences).to(DEVICE)
    output = model(torch.rand((5, 6, 3, 16, 16), dtype=torch.float, device=DEVICE))
    assert output.size() == expected_output_size

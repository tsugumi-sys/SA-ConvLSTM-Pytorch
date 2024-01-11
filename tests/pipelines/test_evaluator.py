import os
import tempfile
from unittest.mock import MagicMock

import torch

from pipelines.evaluator import Evaluator
from tests.test_model.model import TestModel
from tests.utils import mock_data_loader

batch_size = 1
dataset_length = 5
data_loader = mock_data_loader(batch_size, dataset_length)


def test_run_return_sequences():
    with tempfile.TemporaryDirectory() as tempdirpath:
        model = TestModel(return_sequences=True)
        evaluator = Evaluator(
            model,
            test_dataloader=data_loader,
            artifact_dir=tempdirpath,
        )
        evaluator.run()
        for i in range(dataset_length):
            assert os.path.exists(os.path.join(tempdirpath, f"test-case{i}.png"))


def test_run_return_sequences_false():
    with tempfile.TemporaryDirectory() as tempdirpath:
        model = TestModel(return_sequences=False)
        evaluator = Evaluator(
            model,
            test_dataloader=data_loader,
            artifact_dir=tempdirpath,
        )
        evaluator.run()
        for i in range(dataset_length):
            assert os.path.exists(os.path.join(tempdirpath, f"test-case{i}.png"))


def test_run_save_attention_maps():
    with tempfile.TemporaryDirectory() as tempdirpath:
        model = TestModel(return_sequences=True)

        # Mock attention maps
        input_frames, _ = next(iter(data_loader))
        _, _, num_frames, height, width = input_frames.size()
        attention_layer_name = "layer1"
        model.get_attention_maps = MagicMock(
            return_value={"layer1": torch.rand(1, num_frames, height * width)}
        )
        model.frame_size = (64, 64)

        evaluator = Evaluator(
            model,
            test_dataloader=data_loader,
            artifact_dir=tempdirpath,
            save_attention_maps=True,
        )
        evaluator.run()
        for i in range(dataset_length):
            assert os.path.exists(os.path.join(tempdirpath, f"test-case{i}.png"))
            assert os.path.exists(
                os.path.join(
                    tempdirpath,
                    "attention_maps",
                    f"test-case{i}",
                    attention_layer_name,
                    "attentionmaps.png",
                )
            )

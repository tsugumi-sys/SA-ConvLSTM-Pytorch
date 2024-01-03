import os
import tempfile
from unittest.mock import MagicMock

import pytest
import torch

from pipelines.evaluator import Evaluator
from tests.test_model.model import TestModel
from tests.utils import mock_data_loader

data_loader = mock_data_loader()


def test_run():
    with tempfile.TemporaryDirectory() as tempdirpath:
        model = TestModel(return_sequences=True)
        evaluator = Evaluator(
            model,
            test_dataloader=data_loader,
            save_dir_path=tempdirpath,
        )
        evaluator.run()


def test_visualize_predlabel_frames():
    with tempfile.TemporaryDirectory() as tempdirpath:
        model = TestModel(return_sequences=True)
        evaluator = Evaluator(
            model,
            test_dataloader=data_loader,
            save_dir_path=tempdirpath,
        )
        input_frames, label_frames = next(iter(data_loader))
        evaluator.visualize_predlabel_frames(0, label_frames, model(input_frames))
        assert os.path.exists(os.path.join(tempdirpath, "test-case0.png"))


def test_visualize_attention_maps():
    with tempfile.TemporaryDirectory() as tempdirpath:
        model = TestModel(return_sequences=True)
        evaluator = Evaluator(
            model,
            test_dataloader=data_loader,
            save_dir_path=tempdirpath,
        )

        with pytest.raises(ValueError):
            evaluator.visualize_attention_maps(batch_idx=0)

        input_frames, _ = next(iter(data_loader))
        _, _, num_frames, height, width = input_frames.size()
        model.get_attention_maps = MagicMock(
            return_value={"layer1": torch.rand(1, num_frames, height * width)}
        )
        model.frame_size = (64, 64)
        evaluator.visualize_attention_maps(batch_idx=0)
        assert os.path.exists(
            os.path.join(
                tempdirpath,
                "attention_maps",
                "test-case0",
                "layer1",
                "attentionmaps.png",
            )
        )

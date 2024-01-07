import os
import tempfile
from unittest.mock import patch

import torch
from torch import nn
from torch.optim import Adam

from pipelines.trainer import Trainer
from pipelines.utils.early_stopping import EarlyStopping
from tests.test_model.model import TestModel
from tests.utils import mock_data_loader


def mocked_save_model(model: nn.Module, save_path: str):
    torch.save({"model_state_dict": model.state_dict()}, save_path)


@patch("pipelines.utils.early_stopping.save_seq2seq_model")
def test_run(mocked_save_seq2seq_model):
    mocked_save_seq2seq_model.side_effect = mocked_save_model
    with tempfile.TemporaryDirectory() as tempdirpath:
        model = TestModel()
        epochs = 3
        trainer = Trainer(
            model=model,
            train_epochs=epochs,
            train_dataloader=mock_data_loader(),
            valid_dataloader=mock_data_loader(),
            loss_criterion=nn.MSELoss(),
            accuracy_criterion=nn.L1Loss(),
            optimizer=Adam(model.parameters(), lr=0.0005),
            early_stopping=EarlyStopping(
                patience=30,
                verbose=True,
                delta=0.0001,
                model_save_path=os.path.join(tempdirpath, "checkpoint.pt"),
            ),
            artifact_dir=tempdirpath,
            metrics_filename="example.csv",
        )
        trainer.run()

        assert os.path.exists(os.path.join(tempdirpath, "checkpoint.pt"))
        assert os.path.exists(os.path.join(tempdirpath, "example.csv"))
        for metrics in trainer.training_metrics.values():
            assert len(metrics) == epochs


@patch("pipelines.utils.early_stopping.save_seq2seq_model")
def test_run_early_stopping(mocked_save_seq2seq_model):
    mocked_save_seq2seq_model.side_effect = mocked_save_model
    with tempfile.TemporaryDirectory() as tempdirpath:
        model = TestModel()
        epochs = 3
        patience = 1
        trainer = Trainer(
            model=model,
            train_epochs=epochs,
            train_dataloader=mock_data_loader(),
            valid_dataloader=mock_data_loader(),
            loss_criterion=nn.MSELoss(),
            accuracy_criterion=nn.L1Loss(),
            optimizer=Adam(model.parameters(), lr=0.0005),
            early_stopping=EarlyStopping(
                patience=patience,
                verbose=True,
                delta=2,
                model_save_path=os.path.join(tempdirpath, "checkpoint.pt"),
            ),
            artifact_dir=tempdirpath,
            metrics_filename="example.csv",
        )
        trainer.run()

        assert os.path.exists(os.path.join(tempdirpath, "checkpoint.pt"))
        assert os.path.exists(os.path.join(tempdirpath, "example.csv"))
        for metrics in trainer.training_metrics.values():
            assert len(metrics) == epochs - patience

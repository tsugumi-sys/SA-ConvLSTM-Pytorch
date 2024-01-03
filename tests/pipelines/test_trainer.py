import os
import random
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
        trainer = Trainer(
            model=model,
            train_epochs=1,
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
            save_metrics_path=os.path.join(tempdirpath, "example.csv"),
        )
        trainer.run()
        print(os.listdir(tempdirpath))
        assert os.path.exists(os.path.join(tempdirpath, "checkpoint.pt"))


@patch("pipelines.utils.early_stopping.save_seq2seq_model")
def test_train(mocked_save_seq2seq_model):
    with tempfile.TemporaryDirectory() as tempdirpath:
        model = TestModel()
        trainer = Trainer(
            model=model,
            train_epochs=2,
            train_dataloader=mock_data_loader(),
            valid_dataloader=mock_data_loader(),
            loss_criterion=nn.MSELoss(),
            accuracy_criterion=nn.L1Loss(),
            optimizer=Adam(model.parameters(), lr=0.0005),
            early_stopping=EarlyStopping(
                patience=30,
                verbose=True,
                delta=0.0001,
            ),
            save_metrics_path=os.path.join(tempdirpath, "example.csv"),
        )
        res = trainer.train()
        assert len(res["train_loss"]) == 2
        assert len(res["validation_loss"]) == 2
        assert len(res["accuracy"]) == 2


@patch("pipelines.utils.early_stopping.save_seq2seq_model")
def test_validation(mocked_save_seq2seq_model):
    with tempfile.TemporaryDirectory() as tempdirpath:
        model = TestModel()
        trainer = Trainer(
            model=model,
            train_epochs=2,
            train_dataloader=mock_data_loader(),
            valid_dataloader=mock_data_loader(),
            loss_criterion=nn.MSELoss(),
            accuracy_criterion=nn.L1Loss(),
            optimizer=Adam(model.parameters(), lr=0.0005),
            early_stopping=EarlyStopping(
                patience=30,
                verbose=True,
                delta=0.0001,
            ),
            save_metrics_path=os.path.join(tempdirpath, "example.csv"),
        )
        valid_loss, acc_loss = trainer.validation()
        assert isinstance(valid_loss, float)
        assert isinstance(acc_loss, float)


@patch("pipelines.utils.early_stopping.save_seq2seq_model")
def test_save_metrics_to_csv(mocked_save_seq2seq_model):
    with tempfile.TemporaryDirectory() as tempdirpath:
        model = TestModel()
        save_metrics_path = os.path.join(tempdirpath, "example.csv")
        trainer = Trainer(
            model=model,
            train_epochs=2,
            train_dataloader=mock_data_loader(),
            valid_dataloader=mock_data_loader(),
            loss_criterion=nn.MSELoss(),
            accuracy_criterion=nn.L1Loss(),
            optimizer=Adam(model.parameters(), lr=0.0005),
            early_stopping=EarlyStopping(
                patience=30,
                verbose=True,
                delta=0.0001,
            ),
            save_metrics_path=save_metrics_path,
        )
        trainer.save_metrics_to_csv(
            {
                "train_loss": [random.random()],
                "validation_loss": [random.random()],
                "accuracy": [random.random()],
            },
            save_metrics_path,
        )
        assert os.path.exists(os.path.join(tempdirpath, "example.csv"))

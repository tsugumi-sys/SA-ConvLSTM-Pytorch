import os
import tempfile

from torch import nn
from torch.optim import Adam

from pipelines.trainer import Trainer
from pipelines.utils.early_stopping import EarlyStopping
from tests.test_model.model import TestModel
from tests.utils import mock_data_loader


def test_run():
    with tempfile.TemporaryDirectory() as tempdirpath:
        model = TestModel()
        epochs = 3
        trainer = Trainer(
            model=model,
            train_epochs=epochs,
            train_dataloader=mock_data_loader(),
            validation_dataloader=mock_data_loader(),
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
        assert os.path.exists(os.path.join(tempdirpath, "learning_curve.png"))
        for metrics in trainer.training_metrics.values():
            assert len(metrics) == epochs


def test_run_early_stopping():
    with tempfile.TemporaryDirectory() as tempdirpath:
        model = TestModel()
        epochs = 3
        patience = 1
        trainer = Trainer(
            model=model,
            train_epochs=epochs,
            train_dataloader=mock_data_loader(),
            validation_dataloader=mock_data_loader(),
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
        assert os.path.exists(os.path.join(tempdirpath, "learning_curve.png"))
        for metrics in trainer.training_metrics.values():
            assert len(metrics) == epochs - patience

import os
import tempfile

from torch import nn
from torch.optim import Adam

from pipelines.experimenter import Experimenter
from pipelines.trainer import TrainingParams
from pipelines.utils.early_stopping import EarlyStopping
from tests.test_model.model import TestModel
from tests.utils import MockMovingMNISTDataLoaders


def test_run():
    with tempfile.TemporaryDirectory() as tempdirpath:
        model = TestModel()
        training_params: TrainingParams = {
            "epochs": 1,
            "loss_criterion": nn.MSELoss(),
            "accuracy_criterion": nn.L1Loss(),
            "optimizer": Adam(model.parameters(), lr=0.0005),
            "early_stopping": EarlyStopping(
                patience=30,
                verbose=True,
                delta=0.0001,
                model_save_path=os.path.join(tempdirpath, "train", "model.pt"),
            ),
            "metrics_filename": "metrics.csv",
        }
        dataset_length = 3
        data_loaders = MockMovingMNISTDataLoaders(
            dataset_length=dataset_length, train_batch_size=1, split_ratio=10
        )
        experimenter = Experimenter(tempdirpath, data_loaders, model, training_params)
        experimenter.run()

        # testing trainer artifacts
        assert os.path.exists(os.path.join(tempdirpath, "train", "model.pt"))
        assert os.path.exists(os.path.join(tempdirpath, "train", "metrics.csv"))
        assert os.path.exists(os.path.join(tempdirpath, "train", "learning_curve.png"))
        # testing evaluator artifacts
        for i in range(dataset_length):
            assert os.path.exists(
                os.path.join(tempdirpath, "evaluation", f"test-case{i}.png")
            )

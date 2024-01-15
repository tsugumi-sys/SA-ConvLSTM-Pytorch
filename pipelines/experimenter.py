import os

from torch import nn

from core.constants import DEVICE
from data_loaders.base import BaseDataLoaders
from pipelines.base import BaseRunner
from pipelines.evaluator import Evaluator
from pipelines.trainer import Trainer, TrainingParams


class Experimenter(BaseRunner):
    def __init__(
        self,
        artifact_dir: str,
        data_loaders: BaseDataLoaders,
        model: nn.Module,
        training_params: TrainingParams,
    ):
        self._artifact_dir = artifact_dir
        self._model = model
        self._training_params = training_params
        self._data_loaders = data_loaders

    @property
    def artifact_dir(self) -> str:
        return self._artifact_dir

    @property
    def data_loaders(self) -> BaseDataLoaders:
        return self._data_loaders

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def training_params(self) -> TrainingParams:
        return self._training_params

    def run(self):
        self.__train()
        self.__evaluate()

    def __train(self):
        print(f"Training {self._model.__class__.__name__} ...")
        print(f" - Device: {DEVICE}")
        trainer = Trainer(
            model=self._model,
            train_epochs=self._training_params["epochs"],
            train_dataloader=self._data_loaders.train_dataloader,
            validation_dataloader=self._data_loaders.validation_dataloader,
            loss_criterion=self._training_params["loss_criterion"],
            accuracy_criterion=self._training_params["accuracy_criterion"],
            optimizer=self._training_params["optimizer"],
            early_stopping=self._training_params["early_stopping"],
            artifact_dir=os.path.join(self._artifact_dir, "train"),
            metrics_filename=self._training_params.get("metrics_filename")
            or "metrics.csv",
        )
        trainer.run()

    def __evaluate(self):
        print(f"Evaluating {self._model.__class__.__name__} ...")
        evaluator = Evaluator(
            model=self._model,
            test_dataloader=self._data_loaders.test_dataloader,
            artifact_dir=os.path.join(self._artifact_dir, "evaluation"),
        )
        evaluator.run()

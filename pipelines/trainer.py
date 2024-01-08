import os
from typing import Callable, Dict, List, Optional, TypedDict, cast

import pandas as pd
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from core.constants import DEVICE
from pipelines.base import BaseRunner
from pipelines.utils.early_stopping import EarlyStopping


class TrainingMetrics(TypedDict):
    train_loss: List[float]
    validation_loss: List[float]
    validation_accuracy: List[float]


class Trainer(BaseRunner):
    def __init__(
        self,
        model: nn.Module,
        train_epochs: int,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        loss_criterion: _Loss,
        accuracy_criterion: Callable,
        optimizer: Optimizer,
        early_stopping: EarlyStopping,
        artifact_dir: str,
        metrics_filename: str = "training_metrics.csv",
    ) -> None:
        self.model = model.to(DEVICE)
        self.train_epochs = train_epochs
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.loss_criterion = loss_criterion
        self.accuracy_criterion = accuracy_criterion
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        if not os.path.exists(artifact_dir):
            os.makedirs(artifact_dir)
        self.artifact_dir = artifact_dir
        if not metrics_filename.endswith(".csv"):
            raise ValueError("`save_metrics_filename` should be end with `.csv`")
        self.metrics_filename = metrics_filename
        self._training_metrics: TrainingMetrics = {
            "train_loss": [],
            "validation_loss": [],
            "validation_accuracy": [],
        }

    def run(self) -> None:
        for epoch in range(1, self.train_epochs + 1):
            self.__train()
            self.__validation()
            training_metric = self.__latest_training_metric()
            if epoch % 10 == 0:
                print(
                    f"Epoch: {epoch}, Training loss: "
                    "{.8f}, Validation loss: {.8f}, Validation Accuracy: {.8f}".format(
                        training_metric["train_loss"],
                        training_metric["validation_loss"],
                        training_metric["validation_accuracy"],
                    )
                )

            self.early_stopping(training_metric["validation_loss"], self.model)
            if self.early_stopping.early_stop is True:
                print(f"Early stopped at epoch {epoch}")
                break

        self.__save_metrics()

    @property
    def training_metrics(self) -> TrainingMetrics:
        return self._training_metrics

    def __train(self):
        train_loss = 0
        self.model.train()
        for _, (input, target) in enumerate(self.train_dataloader, start=1):
            input, target = input.to(DEVICE), target.to(DEVICE)

            output = self.model(input)
            loss = self.loss_criterion(output.flatten(), target.flatten())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        self.__log_metric(train_loss=train_loss / len(self.train_dataloader))

    def __validation(self):
        valid_loss, valid_acc = 0, 0
        self.model.eval()
        with torch.no_grad():
            for input, target in self.validation_dataloader:
                input, target = input.to(DEVICE), target.to(DEVICE)
                output = self.model(input)
                loss = self.loss_criterion(output.flatten(), target.flatten())
                acc = self.accuracy_criterion(output.flatten(), target.flatten())
                valid_loss += loss.item()
                valid_acc += acc.item()
        dataset_length = len(self.validation_dataloader)
        self.__log_metric(
            validation_loss=valid_loss / dataset_length,
            validation_accuracy=valid_acc / dataset_length,
        )

    def __log_metric(
        self,
        train_loss: Optional[float] = None,
        validation_loss: Optional[float] = None,
        validation_accuracy: Optional[float] = None,
    ):
        if train_loss is not None:
            self._training_metrics["train_loss"].append(train_loss)
        if validation_loss is not None:
            self._training_metrics["validation_loss"].append(validation_loss)
        if validation_accuracy is not None:
            self._training_metrics["validation_accuracy"].append(validation_accuracy)

    def __latest_training_metric(self) -> Dict[str, float]:
        return {k: cast(List[float], v)[-1] for k, v in self._training_metrics.items()}

    def __save_metrics(self) -> None:
        pd.DataFrame(self._training_metrics).to_csv(
            os.path.join(self.artifact_dir, self.metrics_filename)
        )

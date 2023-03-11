import sys
from typing import Callable, Dict, Tuple

import pandas as pd
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

sys.path.append(".")
from train.early_stopping import EarlyStopping  # noqa: E402


class Trainer:
    def __init__(
        self,
        save_model_path: str,
        model: nn.Module,
        train_epochs: int,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        loss_criterion: _Loss,
        accuracy_criterion: Callable,
        optimizer: Optimizer,
        early_stopping: EarlyStopping,
        save_metrics_path: str,
    ) -> None:
        self.save_model_path = save_model_path
        self.model = model
        self.train_epochs = train_epochs
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.loss_criterion = loss_criterion
        self.accuracy_criterion = accuracy_criterion
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.save_metrics_path = save_metrics_path

    def run(self) -> None:
        results = self.train()
        self.save_metrics_to_csv(results, self.save_metrics_path)

    def train(self) -> Dict:
        results = {"train_loss": [], "validation_loss": [], "accuracy": []}
        for epoch in range(1, self.train_epochs + 1):
            train_loss = 0
            self.model.train()
            for _, (input, target) in enumerate(self.train_dataloader, start=1):
                output = self.model(input)

                loss = self.loss_criterion(output.flatten(), target.flatten())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss = loss.item()

            train_loss /= len(self.train_dataloader)
            valid_loss, valid_acc = self.validation()

            results["train_loss"].append(train_loss)
            results["validation_loss"].append(valid_loss)
            results["accuracy"].append(valid_acc)

            if epoch % 10 == 0:
                print(
                    f"Epoch: {epoch}, Training loss: {train_loss:.8f}, Validation loss: {valid_loss:.8f}, Validation Accuracy: {valid_acc:.8f}"
                )

            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop is True:
                print(f"Early stopped at epoch {epoch}")
                break
        return results

    def validation(self) -> Tuple[float, float]:
        self.model.eval()
        with torch.no_grad():
            input, target = next(iter(self.valid_dataloader))
            output = self.model(input)
            loss = self.loss_criterion(output.flatten(), target.flatten())
            acc = self.accuracy_criterion(output.flatten(), target.flatten())

        return loss.item(), acc.item()

    def save_metrics_to_csv(self, results: Dict, save_metrics_path: str) -> None:
        if not save_metrics_path.endswith(".csv"):
            raise ValueError("save_metrics_path should be end with `.csv`")

        pd.DataFrame(results).to_csv(save_metrics_path)

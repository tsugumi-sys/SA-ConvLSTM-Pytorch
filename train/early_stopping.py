from typing import Callable
import sys

import numpy as np
import torch
from torch import nn

sys.path.append(".")
from train.src.seq_to_seq import Seq2Seq  # noqa: E402
from train.src.obpoint_seq_to_seq import OBPointSeq2Seq  # noqa: E402
from train.src.model_for_test import TestModel  # noqa: E402
from train.src.models.self_attention_convlstm.self_attention_convlstm import (
    SelfAttentionSeq2Seq,
)


class EarlyStopping:
    def __init__(
        self,
        patience: int = 7,
        verbose: bool = False,
        delta: float = 0.0,
        path: str = "checkpoint.pt",
        trace_func: Callable = print,
    ) -> None:
        """Early stops the training if validation loss doesn't improve after a given patience

        Args:
            patience (int, optional): How long to wait after last time validation loss improved. Defaults to 7.
            verbose (bool, optional): If True, prints a message for each validation loss improvement. Defaults to False.
            delta (float, optional): Minumum change in the monitored quantity to qualify as an improvement. Defaults to 0.0.
            path (str, optional): Path for the checkpoint to be saved to. Defaults to "checkpoint.pt".
            trace_func (Callable, optional): trace print function. Defaults to print.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.state_dict = None

    def __call__(self, val_loss: float, model: nn.Module):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module):
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
            )

        if isinstance(model, OBPointSeq2Seq):
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_channels": model.num_channels,
                    "ob_point_count": model.ob_point_count,
                    "kernel_size": model.kernel_size,
                    "num_kernels": model.num_kernels,
                    "padding": model.padding,
                    "activation": model.activation,
                    "frame_size": model.frame_size,
                    "num_layers": model.num_layers,
                    "input_seq_length": model.input_seq_length,
                    "prediction_seq_length": model.prediction_seq_length,
                    "out_channels": model.out_channels,
                    "weights_initializer": model.weights_initializer,
                },
                self.path,
            )
        elif isinstance(model, Seq2Seq):
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_channels": model.num_channels,
                    "kernel_size": model.kernel_size,
                    "num_kernels": model.num_kernels,
                    "padding": model.padding,
                    "activation": model.activation,
                    "frame_size": model.frame_size,
                    "num_layers": model.num_layers,
                    "weights_initializer": model.weights_initializer,
                },
                self.path,
            )
        elif isinstance(model, SelfAttentionSeq2Seq):
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "attention_layer_hidden_dims": model.attention_layer_hidden_dims,
                    "num_channels": model.num_channels,
                    "kernel_size": model.kernel_size,
                    "num_kernels": model.num_kernels,
                    "padding": model.padding,
                    "activation": model.activation,
                    "frame_size": model.frame_size,
                    "num_layers": model.num_layers,
                    "input_seq_length": model.input_seq_length,
                    "prediction_seq_length": model.prediction_seq_length,
                    "out_channels": model.out_channels,
                    "weights_initializer": model.weights_initializer,
                },
                self.path,
            )
        elif isinstance(model, TestModel):
            torch.save({"model_state_dict": model.state_dict()}, self.path)

        self.val_loss_min = val_loss

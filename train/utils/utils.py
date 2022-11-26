from typing import List, Optional
import os
import sys

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

sys.path.append(".")
from models.convlstm.seq2seq import Seq2Seq
from models.self_attention_convlstm.sa_seq2seq import SASeq2Seq
from models.self_attention_memory_convlstm.sam_seq2seq import SAMSeq2Seq


def save_learning_curve_plot(
    save_dir_path: str,
    model_name: str,
    training_losses: List,
    validation_losses: List,
) -> None:
    """create and save leanring curve plot

    Args:
        save_dir_path (str): Directory path to save figure
        training_losses (List): training_losses from train(...)
        validation_losses (List): validation_losses from train(...)
        validation_accuracy (Optional[List], optional): validation_accuracy from train(...). Defaults to None.

    Raises:
        ValueError: training_losses and validation_losses must be the same length
        ValueError: training_losses and validation_accuracy must be the same length

    Returns:
        str: Saved figure path.
    """
    if len(training_losses) != len(validation_losses):
        raise ValueError("train_losses and validation_losses must be the same length.")

    _, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("Epochs")
    x = [i for i in range(len(training_losses))]
    sns.lineplot(
        x=x, y=training_losses, label="Training Loss", ax=ax, color="tab:orange"
    )
    sns.lineplot(
        x=x, y=validation_losses, label="Validation Loss", ax=ax, color="tab:blue"
    )
    ax.set_ylabel("Training & Validation Loss")

    ax.legend(loc="upper center")
    plt.tight_layout()
    save_path = os.path.join(save_dir_path, f"{model_name}_training_results.png")
    plt.savefig(save_path)
    plt.close()


def save_metrics(
    save_dir_path: str, model_name: str, training_losses: List, validation_losses: List
) -> None:
    df = pd.DataFrame()
    df["TrainLoss"] = training_losses
    df["ValidationLoss"] = validation_losses
    df.to_csv(os.path.join(save_dir_path, f"{model_name}-metrics.csv"))


def save_seq2seq_model(model: nn.Module, save_path: str) -> None:
    if isinstance(model, Seq2Seq):
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
            save_path,
        )
    elif isinstance(model, SASeq2Seq) or isinstance(model, SAMSeq2Seq):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "attention_hidden_dims": model.attention_hidden_dims,
                "num_channels": model.num_channels,
                "kernel_size": model.kernel_size,
                "num_kernels": model.num_kernels,
                "padding": model.padding,
                "activation": model.activation,
                "frame_size": model.frame_size,
                "num_layers": model.num_layers,
                "weights_initializer": model.weights_initializer,
            },
            save_path,
        )
    else:
        raise ValueError(f"Unknown model {model}")
    return

from typing import List, Optional
import os

import seaborn as sns
import matplotlib.pyplot as plt


def learning_curve_plot(
    save_dir_path: str,
    model_name: str,
    training_losses: List,
    validation_losses: List,
    validation_accuracy: Optional[List] = None,
) -> str:
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
    if validation_accuracy is not None:
        if len(training_losses) != len(validation_accuracy):
            raise ValueError(
                "train_losses and validation_accuracy must be the same length."
            )

        ax2 = ax.twinx()
        sns.lineplot(
            x=x,
            y=validation_accuracy,
            label="Validation Accuracy",
            ax=ax2,
            color="tab:green",
        )

        ax2.set_ylabel("Validation Accuracy")
        ax2.legend(loc="upper right")

    ax.legend(loc="upper center")
    plt.tight_layout()
    save_path = os.path.join(save_dir_path, f"{model_name}_training_results.png")
    plt.savefig(save_path)
    plt.close()

    return save_path

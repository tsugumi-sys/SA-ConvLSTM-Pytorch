import os
from typing import List

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib import colormaps
from torch import nn


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


def save_pred_vs_label_images(
    image_path: str, label_frames: torch.Tensor, pred_frames: torch.Tensor
):
    """
    frames: (BatchSize=1, Channels, Frames, H, W)
    """
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    num_frames = label_frames.size(2)
    fig = plt.figure(figsize=(16, 6))
    subfigs = fig.subfigures(nrows=2, ncols=1)
    for rowidx, subfig in enumerate(subfigs):
        title = "prediction" if rowidx == 1 else "label"
        subfig.suptitle(title)
        axs = subfig.subplots(1, num_frames)
        frame_data = pred_frames if rowidx == 1 else label_frames
        frame_data = frame_data.cpu().detach()
        for fidx in range(num_frames):
            ax = axs[fidx]
            with torch.no_grad():
                ax.imshow(frame_data[0, 0, fidx], cmap="gray")
            ax.set_xlabel(f"frame{fidx}")
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.savefig(image_path)


def save_attention_maps(save_dir_path: str, model: nn.Module) -> None:
    os.makedirs(os.path.dirname(save_dir_path), exist_ok=True)
    get_attention_maps = getattr(model, "get_attention_maps", None)
    if not callable(get_attention_maps):
        raise ValueError(
            f"{model.__class__.__name__} does not have `get_attention_maps` method."
        )

    # NOTE: attention maps shape is (batch_size=1, frames, height * width)
    # Extracted only center attention map because of memory usage.
    for layer_name, attention_maps in get_attention_maps().items():
        layer_name = layer_name.split(".")[-1]
        save_dir = os.path.join(save_dir_path, layer_name)
        os.makedirs(save_dir, exist_ok=True)
        num_frames = attention_maps.size(1)
        fig, axes = plt.subplots(1, num_frames, figsize=(16, 3))
        for fidx in range(num_frames):
            # save only attention maps of center
            target_maps = attention_maps[0, fidx]
            target_maps = (
                torch.reshape(target_maps, model.frame_size)
                .cpu()
                .detach()
                .numpy()
                .copy()
            )

            # Save as npy file.
            with open(
                os.path.join(save_dir, f"attentionMap-frame{fidx}.npy"), "wb"
            ) as fp:
                np.save(fp, target_maps)

            xi, yi = np.meshgrid(
                np.linspace(0, model.frame_size[0], 5),
                np.linspace(0, model.frame_size[1], 5),
            )
            ax = axes[fidx]
            clevels = np.linspace(0, target_maps.max(), 25)
            cmap = colormaps.get_cmap("bwr")
            ax.imshow(
                target_maps,
                cmap=cmap,
                norm=mcolors.BoundaryNorm(clevels, cmap.N),
            )

            x_center, y_center = (
                model.frame_size[0] / 2,
                model.frame_size[1] / 2,
            )
            ax.plot(x_center, y_center, color="black", marker="+", markersize=8)
            ax.set_xlabel(f"frame{fidx}")
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.savefig(os.path.join(save_dir, "attentionmaps.png"))

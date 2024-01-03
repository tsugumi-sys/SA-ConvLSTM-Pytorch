import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from torch import nn
from torch.utils.data import DataLoader

from core.constants import DEVICE


class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        test_dataloader: DataLoader,
        save_dir_path: str,
        save_attention_maps: bool = False,
    ):
        self.model = model
        self.test_dataloader = test_dataloader

        os.makedirs(save_dir_path, exist_ok=True)
        self.save_dir_path = save_dir_path
        self.save_attention_maps = save_attention_maps

    def run(self):
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(self.test_dataloader):
                input, label = input.to(DEVICE), label.to(DEVICE)
                pred_frames = self.model(input)
                self.visualize_predlabel_frames(batch_idx, label, pred_frames)
                if self.save_attention_maps:
                    self.visualize_attention_maps(batch_idx)

    def visualize_predlabel_frames(
        self, batch_idx: int, label_frames: torch.Tensor, pred_frames: torch.Tensor
    ):
        """
        frames: (BatchSize=1, Channels, Frames, H, W)
        """
        num_frames = label_frames.size(2)
        fig = plt.figure(figsize=(16, 6))
        subfigs = fig.subfigures(nrows=2, ncols=1)
        for rowidx, subfig in enumerate(subfigs):
            title = "prediction" if rowidx == 1 else "label"
            subfig.suptitle(title)
            axs = subfig.subplots(1, num_frames)
            frame_data = pred_frames if rowidx == 1 else label_frames
            for fidx in range(num_frames):
                ax = axs[fidx]
                with torch.no_grad():
                    print(frame_data.shape)
                    ax.imshow(frame_data[0, 0, fidx], cmap="gray")
                ax.set_xlabel(f"frame{fidx}")
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.savefig(os.path.join(self.save_dir_path, f"test-case{batch_idx}.png"))

    def visualize_attention_maps(self, batch_idx: int) -> None:
        get_attention_maps = getattr(self.model, "get_attention_maps", None)
        if not callable(get_attention_maps):
            raise ValueError(
                f"{self.model.__class__.__name__} does not have `get_attention_maps` method."
            )

        # NOTE: attention maps shape is (batch_size=1, frames, height * width)
        # Extracted only center attention map because of memory usage.
        for layer_name, attention_maps in get_attention_maps().items():
            layer_name = layer_name.split(".")[-1]
            save_dir = os.path.join(
                self.save_dir_path,
                "attention_maps",
                f"test-case{batch_idx}",
                layer_name,
            )
            os.makedirs(save_dir, exist_ok=True)
            num_frames = attention_maps.size(1)
            fig, axes = plt.subplots(1, num_frames, figsize=(16, 3))
            for fidx in range(num_frames):
                # save only attention maps of center
                target_maps = attention_maps[0, fidx]
                target_maps = (
                    torch.reshape(target_maps, self.model.frame_size)
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
                    np.linspace(0, self.model.frame_size[0], 5),
                    np.linspace(0, self.model.frame_size[1], 5),
                )
                ax = axes[fidx]
                clevels = np.linspace(0, target_maps.max(), 25)
                cmap = cm.bwr
                ax.imshow(
                    target_maps,
                    cmap=cmap,
                    norm=mcolors.BoundaryNorm(clevels, cmap.N),
                )

                x_center, y_center = (
                    self.model.frame_size[0] / 2,
                    self.model.frame_size[1] / 2,
                )
                ax.plot(x_center, y_center, color="black", marker="+", markersize=8)
                ax.set_xlabel(f"frame{fidx}")
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            plt.savefig(os.path.join(save_dir, "attentionmaps.png"))

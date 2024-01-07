import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from core.constants import DEVICE
from pipelines.base import BaseRunner
from pipelines.utils.visualize_utils import (
    save_attention_maps,
    save_pred_vs_label_images,
)


class Evaluator(BaseRunner):
    def __init__(
        self,
        model: nn.Module,
        test_dataloader: DataLoader,
        artifact_dir: str,
        save_attention_maps: bool = False,
    ):
        self.model = model
        self.test_dataloader = test_dataloader

        os.makedirs(artifact_dir, exist_ok=True)
        self.artifact_dir = artifact_dir
        self.save_attention_maps = save_attention_maps

    def run(self):
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(self.test_dataloader):
                input, label = input.to(DEVICE), label.to(DEVICE)
                pred_frames = self.model(input)
                save_pred_vs_label_images(
                    os.path.join(self.artifact_dir, f"test-case{batch_idx}.png"),
                    label,
                    pred_frames,
                )
                if self.save_attention_maps:
                    save_attention_maps(
                        os.path.join(
                            self.artifact_dir, "attention_maps", f"test-case{batch_idx}"
                        ),
                        self.model,
                    )

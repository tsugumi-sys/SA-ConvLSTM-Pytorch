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
                pred_frames = self.__predict_frames(input, label)
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

    def __predict_frames(
        self, input: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        input, label = input.to(DEVICE), label.to(DEVICE)
        if self.model.return_sequences:
            return self.model(input)

        # Generate prediction frames with updating input data sequentially.
        pred_frames = torch.zeros(label.size(), dtype=torch.float, device=DEVICE)
        for frame_idx in range(input.size(2)):
            if frame_idx == 0:
                pred_frames[:, :, frame_idx] = self.model(input)
            else:
                pred_frames[:, :, frame_idx] = self.model(
                    torch.cat(
                        (input[:, :, frame_idx:], label[:, :, :frame_idx]),
                        2,
                    )
                )
        return pred_frames

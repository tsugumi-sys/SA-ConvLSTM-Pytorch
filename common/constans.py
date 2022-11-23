from enum import Enum

import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class WeightsInitializer(str, Enum):
    Zeros = "zeros"
    He = "he"
    Xavier = "xavier"

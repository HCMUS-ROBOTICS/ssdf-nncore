import torch
from .typing import *
import numpy as np


def multi_class_prediction(output: Tensor) -> Tensor:
    return torch.argmax(output, dim=1)


def binary_prediction(output: Tensor, thresh: float = 0.5) -> Tensor:
    return (output.squeeze(1) > thresh).long()


class OneHotEncoding:
    r"""Convert (H,W) to (N_CLASSES, H, W)
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        ncols = self.n_classes

        inputs = inputs.astype(int)
        out = np.zeros((inputs.size, ncols), dtype=np.uint8)
        out[np.arange(inputs.size), inputs.ravel()] = 1
        out.shape = inputs.shape + (ncols,)
        return out


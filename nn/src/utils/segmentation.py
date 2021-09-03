import torch
from .typing import *


def multi_class_prediction(output: Tensor) -> Tensor:
    return torch.argmax(output, dim=1)


def binary_prediction(output: Tensor, thresh: float = 0.0) -> Tensor:
    return (output.squeeze(1) > thresh).long()

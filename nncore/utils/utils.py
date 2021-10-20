import copy
from typing import Any, List

import matplotlib.pyplot as plt
import torch
from torch.nn import Module


def copy_model(model: Module) -> Module:
    return copy.deepcopy(model)


def tensor2plt(obj: torch.Tensor, title: List[Any]):
    fig = plt.figure()
    plt.imshow(obj.permute(1, 2, 0))
    plt.title(title)
    return fig


def inverse_normalize(
    tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
):
    """
    Inverse normalization for one image.
    Input : C x H x W
    Output : C x H x W
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)

    return tensor


def inverse_normalize_batch(
    tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
):
    """
    Inverse normalization for a batch of RGB images (3 channels only).
    Input : B x C x H x W
    Output : B x C x H x W
    """
    x = tensor.new(*tensor.size())
    x[:, 0, :, :] = tensor[:, 0, :, :] * std[0] + mean[0]
    x[:, 1, :, :] = tensor[:, 1, :, :] * std[1] + mean[1]
    x[:, 2, :, :] = tensor[:, 2, :, :] * std[2] + mean[2]
    return x

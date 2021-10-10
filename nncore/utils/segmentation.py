import torch
import numpy as np
from .color import color_map


def multi_class_prediction(output: torch.Tensor) -> torch.Tensor:
    return torch.argmax(output, dim=1)


def binary_prediction(output: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    return (output.squeeze(1) > thresh).long()


def np2cmap(np_image):
    """
    Using Dot product to compute the color batch 

    1. Create a new color axis from H x W x B to H x W x B x 1 
    2. Multiply the color axis by the image tensor H x W x B x 1 and 1 x C (3 with RGB)
    3. Sum the color axis to get the final color map H x W x B x C 


    input: numpy batch H x W x B 
    output: color map batch  H x W x B x C
    """

    cmap = color_map()[:, np.newaxis, :]
    np_image = np_image[..., np.newaxis]
    new_im = np.dot(np_image == 0, cmap[0])
    for i in range(1, cmap.shape[0]):
        new_im += np.dot(np_image == i, cmap[i])
    return new_im


def tensor2cmap(tensor):
    """
    input: Tensor batch B x H x W
    output: color map batch  B x C x H x W
    """
    np_image = tensor.permute(1, 2, 0).cpu().numpy()
    return torch.tensor(np2cmap(np_image)).permute(2, 3, 0, 1)

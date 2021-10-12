""" 
Python implementation of the color map function for the PASCAL VOC data set. 
Official Matlab version can be found in the PASCAL VOC devkit 
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
"""
import numpy as np
import torch


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = "float32" if normalized else "uint8"
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


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

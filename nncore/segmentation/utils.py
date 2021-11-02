"""
Python implementation of the color map function for the PASCAL VOC data set.
Official Matlab version can be found in the PASCAL VOC devkit
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
"""
import numpy as np
import torch


def color_map_cityscape(n=256):
    # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.pyconstruction
    from collections import namedtuple

    Label = namedtuple(
        "Label",
        [
            "name",
            "id",
            "trainId",
            "category",
            "categoryId",
            "hasInstances",
            "ignoreInEval",
            "color",
        ],
    )

    labels = [
        # name, id, trainId, category, catId, hasInstances, ignoreInEval, color
        Label("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
        Label("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)),
        Label("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)),
        Label("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)),
        Label("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
        Label("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)),
        Label("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
        Label("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
        Label("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
        Label("parking", 9, 255, "flat", 1, False, True, (250, 170, 160)),
        Label("rail track", 10, 255, "flat", 1, False, True, (230, 150, 140)),
        Label("building", 11, 2, "construction", 2, False, False, (70, 70, 70)),
        Label("wall", 12, 3, "construction", 2, False, False, (102, 102, 156)),
        Label("fence", 13, 4, "construction", 2, False, False, (190, 153, 153)),
        Label("guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180)),
        Label("bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)),
        Label("tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)),
        Label("pole", 17, 5, "object", 3, False, False, (153, 153, 153)),
        Label("polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)),
        Label("traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)),
        Label("traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)),
        Label("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)),
        Label("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)),
        Label("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
        Label("person", 24, 11, "human", 6, True, False, (220, 20, 60)),
        Label("rider", 25, 12, "human", 6, True, False, (255, 0, 0)),
        Label("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
        Label("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
        Label("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
        Label("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)),
        Label("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)),
        Label("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)),
        Label("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)),
        Label("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
        Label("license plate", -1, -1, "vehicle", 7, False, True, (0, 0, 142)),
    ]
    dtype = "uint8"
    cmap = np.zeros((n, 3), dtype=dtype)
    for i in range(len(labels)):
        r, g, b = labels[i].color
        cmap[i] = np.array([r, g, b])
    return cmap


def color_map_voc(n=256, normalized=False):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = "float32" if normalized else "uint8"
    cmap = np.zeros((n, 3), dtype=dtype)
    for i in range(n):
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


def color_map(label_format: str):
    if label_format == "voc":
        return color_map_voc()
    elif label_format == "cityscape":
        return color_map_cityscape()
    else:
        raise ValueError("Unknown label format: {}".format(label_format))


def binary_prediction(output: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    return (output.squeeze(1) > thresh).long()


def np2cmap(np_image, label_format: str = "voc"):
    """
    Using Dot product to compute the color batch

    1. Create a new color axis from H x W x B to H x W x B x 1
    2. Multiply the color axis by the image tensor H x W x B x 1 and 1 x C (3 with RGB)
    3. Sum the color axis to get the final color map H x W x B x C


    input: numpy batch H x W x B
    output: color map batch  H x W x B x C
    """
    cmap = color_map(label_format=label_format)[:, np.newaxis, :]
    np_image = np_image[..., np.newaxis]
    new_im = np.dot(np_image == 0, cmap[0])
    for i in range(1, cmap.shape[0]):
        new_im += np.dot(np_image == i, cmap[i])
    return new_im


def tensor2cmap(tensor, label_format: str = "voc"):
    """
    input: Tensor batch B x H x W
    output: color map batch  B x C x H x W
    """
    np_image = tensor.permute(1, 2, 0).cpu().numpy()
    return torch.tensor(np2cmap(np_image, label_format)).permute(2, 3, 0, 1)

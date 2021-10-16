import xml.etree.ElementTree as xmlET

import cv2
import numpy as np


class ColorEncoding():
    r"""Convert (H,W,3) to (H,W) where each pixel is a scalar of
    class index defined by given palette

    Args:
        palette: a dict of { class_idx: [color1, color2, ...] }

    Returns:
        a tensor of shape (N_CLASS, H, W), each pixel is an one-hot-vector
        represent for its class
    """

    def __init__(self, pallete):
        self.pallete = pallete
        self.n_classes = len(pallete)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        encoding = np.zeros(
            (image.shape[0], image.shape[1]), dtype=np.int32)  # (H, W)
        for class_idx, colors in self.pallete.items():
            for color in colors:
                class_mask = np.all(image == color, axis=-1)  # H, W
                encoding[class_mask] = class_idx - 1
        return encoding


class OneHotEncoding():
    r"""Convert (H,W) to (N_CLASSES, H, W)
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        ncols = self.n_classes
        out = np.zeros((inputs.size, ncols), dtype=np.uint8)
        out[np.arange(inputs.size), inputs.ravel()] = 1
        out.shape = inputs.shape + (ncols,)
        return out


class Resize():
    def __init__(self,
                 size=(256, 512),  # H, W
                 crop_to_preserve_aspect_ratio=True,
                 interpolation=cv2.INTER_CUBIC):
        self.size = size
        self.crop_to_preserve_aspect_ratio = crop_to_preserve_aspect_ratio
        self.interpolation = interpolation

    def apply(self, image):
        if not self.crop_to_preserve_aspect_ratio:
            return cv2.resize(image, (self.size[1], self.size[0]),  # (W, H)
                              interpolation=cv2.INTER_NEAREST)

        src_h, src_w = image.shape[:2]
        dst_h, dst_w = self.size

        # first crop to match target aspect ratio
        fx = dst_w / src_w
        fy = dst_h / src_h
        if fx < fy:
            y = 0
            w = src_h * dst_w / dst_h
            x = (src_w - w) // 2
            h = src_h
        else:
            w = int(src_w * dst_h / dst_w)
            y = (src_h - w) // 2
            x = 0
            h = src_w

        # then resize to target shape
        img = image[y:y+h, x:x+w]
        img = cv2.resize(
            img, (self.size[1], self.size[0]), interpolation=self.interpolation)
        return img

    def __call__(self, image, mask=None):
        image = self.apply(image)
        if mask is not None:
            mask = self.apply(mask)
        return image, mask


def parse_convert_xml(conversion_file_path):
    defRoot = xmlET.parse(conversion_file_path).getroot()
    one_hot_palette = {}
    for idx, defElement in enumerate(defRoot.findall("SLabel")):
        color = np.fromstring(defElement.get("fromColour"), dtype=int, sep=" ")
        class_idx = int(defElement.get("toValue"))
        d = one_hot_palette.get(class_idx, [])
        d.append(color)
        one_hot_palette[class_idx] = d

    return one_hot_palette


def parse_convert_xml_v2(conversion_file_path):
    defRoot = xmlET.parse(conversion_file_path).getroot()
    color_pallete = []
    for idx, defElement in enumerate(defRoot.findall("SLabel")):
        color = np.fromstring(defElement.get("fromColour"), dtype=int, sep=" ")
        class_idx = int(defElement.get("toValue"))
        color_pallete.append([class_idx, *color])

    color_pallete = np.array(color_pallete)
    return color_pallete

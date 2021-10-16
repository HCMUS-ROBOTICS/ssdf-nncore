import sys
sys.path.insert(0, "../../")

import cv2
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision import transforms as tf
from PIL import Image
from nncore.utils.segmentation import multi_class_prediction
from nncore.utils.getter import get_instance
from nncore.utils.device import detach, move_to
from nncore.utils import get_device, image_batch_show, load_model, load_yaml
import torch.nn as nn
import torch
from typing import Optional
import matplotlib.pyplot as plt
from pathlib import Path
import torchvision
import xml.etree.ElementTree as xmlET


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

    def __call__(self, image, mask):
        image = self.apply(image)
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


class SegmentationModel(nn.Module):
    """Some Information about SegmentationModel"""

    def __init__(
        self,
        model: nn.Module,
    ):
        super(SegmentationModel, self).__init__()
        self.model = model
        self.onehot_encode = OneHotEncoding(10)
        self.color_encode = ColorEncoding(parse_convert_xml('convert_10.xml'))

    @torch.inference_mode()
    def predict(
        self,
        image_list,
        device,
        image_size=(224, 224),
        batch_size=1,
        verbose=True,
        return_inp=False,
    ):
        progress_bar = tqdm(image_list) if verbose else image_list

        res = []
        inp = []
        for image_path in progress_bar:
            image = np.array(Image.open(image_path))
            inputs = self.onehot_encode(self.color_encode(image))
            inputs = torch.tensor(inputs).unsqueeze_(0).permute(0, 3, 1, 2)  # 1, C, H, W
            inputs = inputs.float().cuda()
            preds = model(inputs)['out']
            outputs = detach(preds)
            inp.append(image)
            res.append(outputs)

        return inp, res

if __name__ == "__main__":
    checkpoint_folder = Path("./runs/default/")
    cfg = load_yaml(checkpoint_folder / "config.yaml")
    model = get_instance(cfg["pipeline"]["model"])
    load_model(model, checkpoint_folder / "best_loss.pth")
    device = get_device()
    inference_model = SegmentationModel(model).to(device).eval()
    rgbs, pred = inference_model.predict(
        [
            "/media/vinhloiit/Data/DiRA/bev-cityscape-format/segmentation-front-ipm/00001.png",
            "/media/vinhloiit/Data/DiRA/bev-cityscape-format/segmentation-front-ipm/00002.png",
        ],
        device=device,
        batch_size=2,
        return_inp=True,
    )
    print(rgbs[0].shape)
    print(pred[0].shape)
    # print(rgbs.shape)
    # print(pred.shape)

    save_dir = Path("./demo")
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.imsave('rgbs.png', rgbs[0])

    # pred[0] # B,C,H,W
    plt.imsave('pred.png', pred[0].squeeze(0).argmax(0).cpu().numpy())

    # pred = multi_class_prediction(pred).unsqueeze(1)
    # outs = image_batch_show(pred)
    # rgbs = image_batch_show(rgbs)

    # save_image(rgbs, str(save_dir / "pred.png"), normalize=True)
    # save_image(outs, str(save_dir / "rbgs.png"))

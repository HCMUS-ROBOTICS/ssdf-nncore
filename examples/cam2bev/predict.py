import sys
from typing import List

sys.path.insert(0, "../../")

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from nncore.segmentation.models import MODEL_REGISTRY
from nncore.utils.device import detach, get_device
from nncore.utils.getter import get_instance
from nncore.utils.loading import load_yaml
from nncore.utils.utils import load_model
from PIL import Image
from tqdm import tqdm

from dataset import Cam2BEVDataset  # noqa
from learner import Cam2BEVLearner  # noqa
from model import build_deeplabv3_cam2bev  # noqa
from transform import ColorEncoding, OneHotEncoding, Resize, parse_convert_xml


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
        self.resize = Resize()

    @torch.inference_mode()
    def predict(
        self,
        image_list: List[str],
        device,
        verbose=True,
    ):
        progress_bar = tqdm(image_list) if verbose else image_list

        for image_path in progress_bar:
            image = np.array(Image.open(image_path))
            inputs, _ = self.resize(image)
            inputs = self.color_encode(inputs)
            inputs = self.onehot_encode(inputs)
            inputs = torch.tensor(inputs).unsqueeze_(0).permute(0, 3, 1, 2)  # 1, C, H, W
            inputs = inputs.float().to(device)
            preds = model(inputs)['out']
            outputs = detach(preds)
            yield inputs, outputs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('checkpoint_dir', type=str)
    parser.add_argument('input_image_paths', type=str, nargs='+')
    parser.add_argument('--output_dir', type=str, default='outputs')

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    cfg = load_yaml(checkpoint_dir / "config.yaml")
    model = get_instance(cfg["pipeline"]["model"], registry=MODEL_REGISTRY)
    load_model(model, checkpoint_dir / "best_loss.pth")
    device = get_device()
    print('Run on', device)
    inference_model = SegmentationModel(model).to(device).eval()
    predicts = inference_model.predict(
        args.input_image_paths,
        device=device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    for input_path, (input_, output_) in zip(args.input_image_paths, predicts):
        input_path = Path(input_path)
        input_ = input_.squeeze(0).argmax(0).cpu().numpy()
        # import pdb; pdb.set_trace()
        input_path_ = output_dir / f'{input_path.stem}_input.png'
        plt.imsave(input_path_, input_)
        output_numpy = output_.squeeze(0).argmax(0).cpu().numpy()  # H, W
        output_path = output_dir / f'{input_path.stem}_output.png'
        plt.imsave(output_path, output_numpy)

from pathlib import Path

import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

from nncore.core.datasets import TestImageDataset
from nncore.core.transforms.albumentation import TRANSFORM_REGISTRY
from nncore.segmentation.models import MODEL_REGISTRY
from nncore.segmentation.utils import tensor2cmap
from nncore.utils.device import get_device
from nncore.utils.getter import get_instance, get_instance_recursively
from nncore.utils.loading import load_model, load_yaml
from nncore.utils.utils import inverse_normalize_batch, tensor2plt

if __name__ == "__main__":
    checkpoint_folder = Path("./runs/default_2021_11_02-23_37_18/checkpoints")
    cfg = load_yaml(checkpoint_folder / "config.yaml")
    model = get_instance(cfg["pipeline"]["model"], registry=MODEL_REGISTRY)
    tf = get_instance_recursively(cfg["transform"]["val"], registry=TRANSFORM_REGISTRY)
    load_model(model, checkpoint_folder / "best_loss.pth")
    device = get_device()
    print(device)
    datasets = TestImageDataset(img_ls=[
        "../../../Lyft/CameraRGB/F61-1.png",
        "../../../Lyft/CameraRGB/F61-1.png",
        "../../../Lyft/CameraRGB/F61-1.png",
        "../../../Lyft/CameraRGB/F61-1.png",
    ], alb_transform=tf)
    model.to(device)
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=2, shuffle=False, num_workers=0)
    save_dir = Path("./")
    for data in dataloader:
        im = data["input"]

        with torch.no_grad():
            output = model(im.to(device))
        output = output['out']
        output = torch.argmax(output, dim=1)  # From B x N_CLS x W x H -> B x W x H

        im = inverse_normalize_batch(im)
        im = make_grid(im, nrow=2, normalize=True).float().cpu()  # B,C,H,W
        im = tensor2plt(im, title="inputs")

        output = tensor2cmap(output, label_format="cityscape")
        output = make_grid(output, nrow=2, normalize=False).float().cpu()
        output = tensor2plt(output.long(), title="labels")
        plt.show()
        break

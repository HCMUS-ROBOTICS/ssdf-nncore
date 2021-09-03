from re import I
from utils import move_to, AverageValueMeter, detach, load_model, get_device
from models import MobileUnet
from datasets import SDataset
from metrics import PixelAccuracy

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from pathlib import Path
from torchvision import transforms as tf

from typing import Dict


@torch.no_grad()
def evaluate(model, dataloader, metric, device, criterion, verbose=True):
    running_loss = AverageValueMeter()
    for m in metric.values():
        m.reset()
    model.eval()
    progress_bar = tqdm(dataloader) if verbose else dataloader
    for i, (inp, lbl) in enumerate(progress_bar):
        # 1: Load inputs and labels
        inp = move_to(inp, device)
        lbl = move_to(lbl, device)

        # 2: Get network outputs
        outs = model(inp)
        # pdb.set_trace()
        # 3: Calculate the loss
        loss = criterion(outs, lbl)
        # 4: Update loss
        running_loss.add(loss.item())
        # 5: Update metric
        outs = detach(outs)
        lbl = detach(lbl)
        for m in metric.values():
            m.update(outs, lbl)

    avg_loss = running_loss.value()[0]
    return avg_loss, metric


def test(model_path: Path):

    device = get_device()

    transform = [
        tf.ToTensor(),
        tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    dataset = SDataset.from_folder(
        root="/mnt/c/Users/nhoxs/workspace/ssdf/devtools/nn/data",
        train=True,
        transform=transform,
        mask_folder_name="mask",
        image_folder_name="images",
        extension="png",
        image_size=(224, 224),
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = MobileUnet().to(device)
    criterion = CrossEntropyLoss().to(device)
    load_model(model, model_path, map_location=device)
    metric = {"pixel accuracty": PixelAccuracy(nclasses=2)}
    result = evaluate(model, dataloader, metric, device, criterion, verbose=True)

    print("+ Evaluation result")
    avg_loss = result[0].value()[0]
    print("Loss:", avg_loss)

    for k in result[1].keys():
        result[1][k].summary()


if __name__ == "__main__":
    test(Path("../checkpoints/baseline.pth"))

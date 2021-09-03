from utils import move_to, AverageValueMeter, detach, load_model, get_device
from models import MobileUnet
from datasets import SDataset
from metrics import PixelAccuracy
from utils.typing import *

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from torchvision import transforms as tf


@torch.no_grad()
def evaluate(
    model: Module,
    dataloader: DataLoader,
    metric: Metric,
    device: torch.device,
    criterion: Module,
    verbose: bool = True,
):
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


def test():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--img-folder-name", type=str, required=True)
    parser.add_argument("--msk-folder-name", type=str, required=True)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--extension", default="png", type=str)

    args = parser.parse_args()

    device = get_device()

    transform = [
        tf.ToTensor(),
        tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    dataset = SDataset.from_folder(
        root=args.data_path,
        train=args.train,
        mask_folder_name=args.msk_folder_name,
        image_folder_name=args.img_folder_name,
        extension=args.extension,
        transform=transform,
    )

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = MobileUnet().to(device)
    criterion = CrossEntropyLoss().to(device)
    load_model(model, args.model_path, map_location=device)
    metric = {"pixel accuracty": PixelAccuracy(nclasses=2)}
    result = evaluate(model, dataloader, metric, device, criterion, verbose=True)

    print("+ Evaluation result")
    avg_loss = result[0]
    print("Loss:", avg_loss)

    for k in result[1].keys():
        result[1][k].summary()


if __name__ == "__main__":
    # test command
    # python test.py --model /mnt/c/Users/nhoxs/workspace/ssdf/devtools/nn/src/tmp/best_loss.pth --data /mnt/c/Users/nhoxs/workspace/ssdf/devtools/nn/data --img images --msk mask --train
    test()

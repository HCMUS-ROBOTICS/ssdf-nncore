from utils import move_to, AverageValueMeter, detach, load_model, get_device
from models import MobileUnet, ModelWithLoss
from models.loss import CE
from datasets import SDataset
from metrics import PixelAccuracy
from utils.typing import *

import torch
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
    for i, batch in enumerate(progress_bar):
        # 1: Load inputs and labels
        batch = move_to(batch, device)

        # 2: Get network outputs
        # 3: Calculate the loss

        outs, loss, loss_dict = model(batch)
        # 4: Update loss
        running_loss.add(loss.item())
        # 5: Update metric
        outs = detach(outs)
        batch = detach(batch)
        for m in metric.values():
            m.update(outs, batch)

    avg_loss = running_loss.value()[0]
    return avg_loss, metric


def test():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, required=True, help="path to pretrained model"
    )
    parser.add_argument(
        "--data-path", type=str, required=True, help="path to data root folder"
    )
    parser.add_argument(
        "--img-folder-name", type=str, required=True, help="image folder name"
    )
    parser.add_argument(
        "--msk-folder-name", type=str, required=True, help="mask / label folder name"
    )
    parser.add_argument(
        "--test", action="store_true", default=False, help="inference flag ",
    )

    parser.add_argument(
        "--extension",
        default="png",
        type=str,
        help="image extenstion (example: png, jpg) ",
    )

    args = parser.parse_args()

    device = get_device()

    transform = [
        tf.ToTensor(),
        tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    dataset = SDataset.from_folder(
        root=args.data_path,
        test=args.test,
        mask_folder_name=args.msk_folder_name,
        image_folder_name=args.img_folder_name,
        extension=args.extension,
        transform=transform,
    )

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = MobileUnet().to(device)
    criterion = CE().to(device)
    modelwl = ModelWithLoss(model, criterion).to(device)

    load_model(model, args.model_path, map_location=device)
    metric = {"pixel accuracty": PixelAccuracy(nclasses=2)}
    result = evaluate(modelwl, dataloader, metric, device, criterion, verbose=True)

    print("+ Evaluation result")
    avg_loss = result[0]
    print("Loss:", avg_loss)

    for k in result[1].keys():
        result[1][k].summary()


if __name__ == "__main__":
    test()

from utils import move_to, AverageValueMeter, detach, load_model, get_device
from models import MobileUnet
from datasets import SDataset
from metrics import PixelAccuracy
from learner import Learner
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from torch.optim import SGD, Adam, RMSprop

from torchvision import transforms as tf


def train():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
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

    model = globals()[args.model]().to(device)
    criterion = CrossEntropyLoss().to(device)
    metric = {"pixel accuracty": PixelAccuracy(nclasses=2)}

    optimizer = Adam(params=model.parameters())

    model_learner = Learner(
        train_data=dataloader,
        val_data=dataloader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        save_dir="./tmp",
    )
    model_learner.fit(n_epochs=3, metrics=metric)


if __name__ == "__main__":
    # test command
    # python train.py --model MobileUnet --data /mnt/c/Users/nhoxs/workspace/ssdf/devtools/nn/data --img images --msk mask --train

    train()

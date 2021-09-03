from utils import move_to, AverageValueMeter, detach, load_model, get_device
from models import MobileUnet
from datasets import SDataset
from metrics import PixelAccuracy
from learner import Learner
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch.optim import SGD, Adam, RMSprop

from pathlib import Path
from torchvision import transforms as tf

from typing import Dict


def train():

    device = get_device()

    transform = [
        tf.ToTensor(),
        tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    print("aaaaa")

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
    print("aaaaa")
    model_learner.fit(n_epochs=3, metrics=metric)


if __name__ == "__main__":
    print("aaaaa")

    train()

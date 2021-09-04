from utils import get_device
from models import MobileUnet, ModelWithLoss
from models.loss import CE
from datasets import SDataset
from metrics import PixelAccuracy
from learner import SupervisedLearner
import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim import SGD, Adam, RMSprop

from torchvision import transforms as tf


def train():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="MobileUnet",
        help="model class (only support MobileUnet now)",
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
        "--test",
        action="store_true",
        default=False,
        help="test flag, not use in training mode",
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

    model = globals()[args.model]().to(device)
    criterion = CE().to(device)
    modelwithloss = ModelWithLoss(model, criterion).to(device)

    metric = {"pixel accuracty": PixelAccuracy(nclasses=2)}

    optimizer = Adam(params=model.parameters())

    model_learner = SupervisedLearner(
        train_data=dataloader,
        val_data=dataloader,
        model=modelwithloss,
        optimizer=optimizer,
        save_dir="./tmp",
    )

    model_learner.fit(n_epochs=3, metrics=metric)


if __name__ == "__main__":

    train()

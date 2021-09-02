import os
import sys
from pathlib import Path
import torch
from glob import glob
import numpy as np
from PIL import Image
from torchvision import transforms as tf

__all__ = ["SDataset"]


class DatasetTemplate(torch.utils.data.Dataset):
    """Some Information about DatasetTemplate"""

    def __init__(self):
        super(DatasetTemplate, self).__init__()

    def __getitem__(self, idx):
        return

    def __len__(self):
        return


class SDataset(torch.utils.data.Dataset):
    """Some Information about SDataset"""

    def __init__(
        self,
        root: str,
        train=True,
        transform=None,
        mask_folder_name="labels",
        image_folder_name="images",
        extension="png",
        image_size=(224, 224),
    ):
        super(SDataset, self).__init__()

        self.train = train
        self.data_root = Path(root)
        self.img_folder = self.data_root / Path(image_folder_name)
        self.lbl_folder = self.data_root / Path(mask_folder_name)
        self.image_size = image_size
        self.extension = extension
        self.transform = (
            transform
            if transform is not None
            else tf.Compose([tf.Resize(self.image_size), tf.ToTensor(),])
        )

        self.list_rgb = self.get_images_list(self.img_folder, self.extension)
        self.list_mask = self.get_images_list(self.lbl_folder, self.extension)
        # self.list_depth = get_images_list(self.img_folder, self.extension)

        assert len(self.list_rgb) == len(
            self.list_mask
        ), f"Image list and mask list should be the same number of images, but are {len(self.list_rgb)} and {len(self.list_mask)}"

    def __getitem__(self, idx):

        im, mask = Image.open(self.list_rgb[idx]), Image.open(self.list_mask[idx])

        assert (
            im.size == mask.size
        ), f"Image and mask {idx} should be the same size, but are {im.size} and {mask.size}"

        im = self.transform(im)
        mask = self.transform(mask)
        mask[mask > 0] = 1
        return im.long(), mask.long()

    def __len__(self):
        return len(self.list_rgb)

    @staticmethod
    def get_images_list(folder_path: Path, extension: str):
        folder_path = str(folder_path)
        print(folder_path)
        return glob(f"{folder_path}/*.{extension}")


if __name__ == "__main__":
    dataset = SDataset(
        root="/mnt/c/Users/nhoxs/workspace/ssdf/devtools/nn/data",
        train=True,
        mask_folder_name="mask",
        image_folder_name="images",
        extension="png",
        image_size=(224, 224),
    )

    print(len(dataset))

    for im, label in dataset:
        print(label)
        break

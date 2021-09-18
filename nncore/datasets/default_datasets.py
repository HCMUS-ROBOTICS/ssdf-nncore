from torchvision import transforms
from ..utils.typing import *
from pathlib import Path

from PIL import Image

import torch


def available(path: str):
    return Path(path).exists()


class TestImageDataset(torch.utils.data.Dataset):
    """Some Information about TestDataset"""

    def __init__(self, transform: transforms, img_ls: List[str]):
        super(TestImageDataset, self).__init__()
        self.ls = []
        self.tf = transform
        for item in img_ls:
            if available(item):
                self.ls.append(item)
            else:
                print(f"{item} is not exist")

    def __getitem__(self, idx):
        im = Image.open(self.ls[idx])
        return {"input": self.tf(im)}

    def __len__(self):
        return len(self.ls)


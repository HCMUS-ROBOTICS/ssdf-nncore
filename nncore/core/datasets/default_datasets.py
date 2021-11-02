from pathlib import Path
from typing import List

import cv2
import torch
from PIL import Image


def available(path: str):
    return Path(path).exists()


class TestImageDataset(torch.utils.data.Dataset):
    """Some Information about TestDataset"""

    def __init__(self, img_ls: List[str], alb_transform=None, tvf_transform=None):
        super(TestImageDataset, self).__init__()
        self.ls = []
        if tvf_transform is None:
            self.tf = alb_transform
            self.tf_type = 'albumentation'
        else:
            self.tf = tvf_transform
            self.tf_type = 'torchvision'
        for item in img_ls:
            if available(item):
                self.ls.append(item)
            else:
                print(f"{item} is not exist")

    def __getitem__(self, idx):
        if self.tf_type == 'albumentation':
            im = cv2.imread(self.ls[idx])[:, :, :3]
            im = self.tf(image=im)['image']
        else:
            im = Image.open(self.ls[idx])
            im = self.tf(im)
        return {"input": im}

    def __len__(self):
        return len(self.ls)

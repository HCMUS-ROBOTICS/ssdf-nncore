from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from nncore.core.datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class LyftDataset(torch.utils.data.Dataset):
    r"""LyftDataset multi-classes segmentation dataset


    Attributes:
        from_list(**args): Create dataset from list
        from_folder(**args): Create dataset from folder path

    Examples:

        dataset = LyftDataset.from_folder(
            root='./data',
            mask_folder_name='masks',
            image_folder_name='images',
            test=False,
        )

        print(len(dataset))
        print(dataset[0]['input'].shape)
        print(dataset[0]['mask'].shape)

    """

    def __init__(
        self,
        rgb_path_ls: List[str],
        mask_path_ls: List[str],
        transform: Optional[List] = None,
        test: bool = False,
        sample: bool = False,
    ):
        super(LyftDataset, self).__init__()
        rgb_path_ls.sort()
        mask_path_ls.sort()
        self.list_rgb = rgb_path_ls[:4] if sample else rgb_path_ls
        self.list_mask = mask_path_ls[:4] if sample else mask_path_ls
        self.train = not (test)
        self.transform = transform
        assert (
            len(self.list_rgb) > 0
        ), f"Data length must be greater than 0, got {len(self.list_rgb)}"
        assert len(self.list_rgb) == len(self.list_mask), (
            f"Image list and mask list should be the same number of images,"
            f"but are {len(self.list_rgb)} and {len(self.list_mask)}"
        )
        # self.list_depth = get_images_list(self.img_folder, self.extension)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:

        im, mask = (
            plt.imread(self.list_rgb[idx])[:, :, :3],
            plt.imread(self.list_mask[idx])[:, :, 0],
        )
        mask = (mask * 255).astype(int)  # convert label to 0 - 1 (W, H)
        # mask = self.label_encoder(mask)  # convert label to one-hot encoding (W, H, 14)
        # mask = mask.T

        transformed = self.transform(image=im, mask=mask)
        im = transformed["image"]
        mask = transformed["mask"]
        # convert (H, W, 14) to (14, W, H)
        item = {"input": im, "mask": mask.long()}
        return item

    def __len__(self) -> int:
        return len(self.list_rgb)

    @staticmethod
    def get_images_list(folder_path: Path, extension: str) -> List[str]:
        """Return file list with specify type from folder

        Args:
            folder_path (Path): folder path
            extension: file extension

        Returns:
            List[str]: full path list of items in folder
        """
        folder_path = str(folder_path)
        return glob(f"{folder_path}/*.{extension}")

    @classmethod
    def from_list(
        cls,
        rgb_path_ls: Optional[List[str]] = None,
        mask_path_ls: Optional[List[str]] = None,
        transform: Optional[List] = None,
        test: bool = False,
        sample: bool = False,
    ):
        """From list method

        Args:
            rgb_path_ls: full image paths list. Defaults to None.
            mask_path_ls: full label paths list. Defaults to None.
            test: Option using for inference mode. If True, `__getitem__`
                does not return label. Defaults to False.
            transform: rgb transform. Defaults to None.
        Returns:
            LyftDataset: dataset class
        """
        return cls(
            rgb_path_ls=rgb_path_ls,
            mask_path_ls=mask_path_ls,
            test=test,
            transform=transform,
            sample=sample,
        )

    @classmethod
    def from_folder(
        cls,
        root: str,
        image_folder_name: str,
        mask_folder_name: str,
        extension="png",
        test: bool = False,
        transform: Optional[List] = None,
        sample: bool = False,
    ):
        r"""From folder method

        Args:
            root: folder root
            image_folder_name: image folder name
            mask_folder_name: label folder name
            extension: image file type extenstion. Defaults to "png".
            test: Option using for inference mode. if True, __getitem__ does not return label.
                Defaults to False.
            transform: rgb transform. Defaults to None.

        Returns:
            LyftDataset: dataset class
        """

        data_root = Path(root)
        img_folder = data_root / Path(image_folder_name)
        lbl_folder = data_root / Path(mask_folder_name)

        list_rgb = LyftDataset.get_images_list(img_folder, extension)
        list_mask = LyftDataset.get_images_list(lbl_folder, extension)

        return cls(list_rgb, list_mask, test=test, transform=transform, sample=sample,)


DATASET_REGISTRY._do_register("LyftDataset.from_folder", LyftDataset.from_folder)

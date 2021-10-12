from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms as tf

__all__ = ["SDataset"]


class DatasetTemplate(torch.utils.data.Dataset):
    """Some Information about DatasetTemplate"""

    def __init__(self):
        super(DatasetTemplate, self).__init__()

    def __getitem__(self, idx):
        NotImplemented

    def __len__(self):
        NotImplemented


class SDataset(torch.utils.data.Dataset):
    r"""SDataset Binary segmentation dataset


    Attributes:
        from_list(**args): Create dataset from list
        from_folder(**args): Create dataset from folder path
    
    Examples: 
    
        dataset = SDataset.from_folder(
            root='./data',
            mask_folder_name='masks',
            image_folder_name='images',
            test=False,
        )

        print(len(dataset))
        print(dataset[0][0].shape)
        print(dataset[0][1].shape)

    """

    def __init__(
        self,
        rgb_path_ls: List[str],
        mask_path_ls: List[str],
        transform: Optional[List] = None,
        m_transform: Optional[List] = None,
        image_size: Tuple[int, int] = (224, 224),
        test: bool = False,
        sample: bool = False,
    ):
        super(SDataset, self).__init__()

        self.list_rgb = rgb_path_ls[:50] if sample else rgb_path_ls
        self.list_mask = mask_path_ls[:50] if sample else mask_path_ls
        self.train = not (test)
        self.image_size = image_size
        self.img_transform = (
            tf.Compose([tf.Resize(self.image_size)] + transform)
            if transform is not None
            else tf.Compose([tf.Resize(self.image_size), tf.ToTensor(),])
        )
        self.msk_transform = (
            tf.Compose([tf.Resize(self.image_size)] + m_transform)
            if m_transform is not None
            else tf.Compose([tf.Resize(self.image_size), tf.ToTensor(),])
        )

        assert len(self.list_rgb) == len(
            self.list_mask
        ), f"Image list and mask list should be the same number of images, but are {len(self.list_rgb)} and {len(self.list_mask)}"
        # self.list_depth = get_images_list(self.img_folder, self.extension)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        im, mask = Image.open(self.list_rgb[idx]), Image.open(self.list_mask[idx])

        assert (
            im.size == mask.size
        ), f"Image and mask {idx} should be the same size, but are {im.size} and {mask.size}"

        im = self.img_transform(im)
        mask = self.msk_transform(mask)
        mask = mask[0, :]
        mask[mask > 0] = 1
        item = {"input": im, "mask": mask.long()}
        return item

    def __len__(self) -> int:
        return len(self.list_rgb)

    @staticmethod
    def get_images_list(folder_path: Path, extension: str) -> List[str]:
        """Return file list with specify type from folder

        Args:
            folder_path (Path): folder path
            extension (str): file extension

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
        m_transform: Optional[List] = None,
        image_size: Tuple[int, int] = (224, 224),
        test: bool = False,
        sample: bool = False,
    ):
        """From list method

        Args:
            rgb_path_ls (Optional[List[str]], optional): full image paths list. Defaults to None.
            mask_path_ls (Optional[List[str]], optional): full label paths list. Defaults to None.
            test (bool, optional): Option using for inference mode. if True, __get_item__ does not return label. Defaults to False.
            transform (Optional[List], optional): rgb transform. Defaults to None.
            m_transform (Optional[List], optional): label transform. Defaults to None.
            image_size (Tuple[int, int]): image size (width, height). Defaults to (224, 224)..
        Returns:
            SDataset: dataset class
        """
        return cls(
            rgb_path_ls=rgb_path_ls,
            mask_path_ls=mask_path_ls,
            test=test,
            transform=transform,
            m_transform=m_transform,
            image_size=image_size,
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
        m_transform: Optional[List] = None,
        image_size: Tuple[int, int] = (224, 224),
        sample: bool = False,
    ):
        r"""From folder method

        Args:
            root (str): folder root
            image_folder_name (str): image folder name
            mask_folder_name (str): label folder name
            extension (str, optional): image file type extenstion. Defaults to "png".
            test (bool, optional): Option using for inference mode. if True, __get_item__ does not return label. Defaults to False.
            transform (Optional[List], optional): rgb transform. Defaults to None.
            m_transform (Optional[List], optional): label transform. Defaults to None.
            image_size (Tuple[int, int]): image size (width, height). Defaults to (224, 224).

        Returns:
            SDataset: dataset class
        """

        data_root = Path(root)
        img_folder = data_root / Path(image_folder_name)
        lbl_folder = data_root / Path(mask_folder_name)

        list_rgb = SDataset.get_images_list(img_folder, extension)
        list_mask = SDataset.get_images_list(lbl_folder, extension)

        return cls(
            list_rgb,
            list_mask,
            test=test,
            transform=transform,
            m_transform=m_transform,
            image_size=image_size,
            sample=sample,
        )


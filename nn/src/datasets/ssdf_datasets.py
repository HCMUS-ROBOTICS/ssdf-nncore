from pathlib import Path
import torch
from glob import glob
from PIL import Image
from torchvision import transforms as tf
from utils.typing import *

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
    """Some Information about SDataset"""

    def __init__(
        self,
        rgb_path_ls: List[str],
        mask_path_ls: List[str],
        train: bool = True,
        transform: Optional[List] = None,
        m_transform: Optional[List] = None,
        image_size: Tuple[int, int] = (224, 224),
    ):
        super(SDataset, self).__init__()

        self.train = train
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
        self.list_rgb = rgb_path_ls
        self.list_mask = mask_path_ls

        assert len(self.list_rgb) == len(
            self.list_mask
        ), f"Image list and mask list should be the same number of images, but are {len(self.list_rgb)} and {len(self.list_mask)}"
        # self.list_depth = get_images_list(self.img_folder, self.extension)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:

        im, mask = Image.open(self.list_rgb[idx]), Image.open(self.list_mask[idx])

        assert (
            im.size == mask.size
        ), f"Image and mask {idx} should be the same size, but are {im.size} and {mask.size}"

        im = self.img_transform(im)
        mask = self.msk_transform(mask)
        mask = mask[0, :]
        mask[mask > 0] = 1
        return im, mask.long()

    def __len__(self) -> int:

        return len(self.list_rgb)

    @staticmethod
    def get_images_list(folder_path: Path, extension: str) -> List[str]:
        folder_path = str(folder_path)
        print(folder_path)
        return glob(f"{folder_path}/*.{extension}")

    @classmethod
    def from_folder(
        cls: Any,
        root: str,
        image_folder_name: str,
        mask_folder_name: str,
        extension="png",
        train: bool = True,
        transform: Optional[List] = None,
        m_transform: Optional[List] = None,
        image_size: tuple = (224, 224),
    ) -> Any:

        data_root = Path(root)
        img_folder = data_root / Path(image_folder_name)
        lbl_folder = data_root / Path(mask_folder_name)

        list_rgb = SDataset.get_images_list(img_folder, extension)
        list_mask = SDataset.get_images_list(lbl_folder, extension)

        obj = cls(
            list_rgb,
            list_mask,
            train=train,
            transform=transform,
            m_transform=m_transform,
            image_size=image_size,
        )
        return obj


if __name__ == "__main__":
    # test command
    # python ssdf_dataset.py --data /mnt/c/Users/nhoxs/workspace/ssdf/devtools/nn/data --img images --msk mask --train

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--img-folder-name", type=str, required=True)
    parser.add_argument("--msk-folder-name", type=str, required=True)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--extension", default="png", type=str)

    args = parser.parse_args()

    dataset = SDataset.from_folder(
        root=args.data_path,
        train=args.train,
        mask_folder_name=args.msk_folder_name,
        image_folder_name=args.img_folder_name,
        extension=args.extension,
        image_size=(224, 224),
    )

    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)


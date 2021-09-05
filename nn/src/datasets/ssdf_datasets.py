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
        self, from_folder=False, **kwargs,
    ):
        super(SDataset, self).__init__()

        if from_folder:
            self.from_folder(**kwargs)
        else:
            self.from_list(**kwargs)

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
        item = {"input": im, "mask": mask.long()}
        return item

    def __len__(self) -> int:
        return len(self.list_rgb)

    @staticmethod
    def get_images_list(folder_path: Path, extension: str) -> List[str]:
        folder_path = str(folder_path)
        return glob(f"{folder_path}/*.{extension}")

    def from_list(
        self,
        rgb_path_ls: Optional[List[str]] = None,
        mask_path_ls: Optional[List[str]] = None,
        transform: Optional[List] = None,
        m_transform: Optional[List] = None,
        image_size: Tuple[int, int] = (224, 224),
        test: bool = False,
        **kwargs,
    ):
        self.list_rgb = rgb_path_ls
        self.list_mask = mask_path_ls
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

    def from_folder(
        self,
        root: str,
        image_folder_name: str,
        mask_folder_name: str,
        extension="png",
        test: bool = False,
        transform: Optional[List] = None,
        m_transform: Optional[List] = None,
        image_size: tuple = (224, 224),
        **kwargs,
    ) -> Any:

        data_root = Path(root)
        img_folder = data_root / Path(image_folder_name)
        lbl_folder = data_root / Path(mask_folder_name)

        list_rgb = SDataset.get_images_list(img_folder, extension)
        list_mask = SDataset.get_images_list(lbl_folder, extension)

        self.from_list(
            list_rgb,
            list_mask,
            test=test,
            transform=transform,
            m_transform=m_transform,
            image_size=image_size,
            **kwargs,
        )


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


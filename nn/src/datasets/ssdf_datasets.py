from pathlib import Path
import torch
from glob import glob
from PIL import Image
from torchvision import transforms as tf
from utils.segmentation import binary_prediction

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
        rgb_path_ls: list,
        mask_path_ls: list,
        train: bool = True,
        transform: tf = None,
        image_size: tuple = (224, 224),
    ):
        super(SDataset, self).__init__()

        self.train = train
        self.image_size = image_size
        self.transform = (
            transform
            if transform is not None
            else tf.Compose([tf.Resize(self.image_size), tf.ToTensor(),])
        )

        self.list_rgb = rgb_path_ls
        self.list_mask = mask_path_ls
        # self.list_depth = get_images_list(self.img_folder, self.extension)

    def __getitem__(self, idx):

        im, mask = Image.open(self.list_rgb[idx]), Image.open(self.list_mask[idx])

        assert (
            im.size == mask.size
        ), f"Image and mask {idx} should be the same size, but are {im.size} and {mask.size}"

        im = self.transform(im)
        mask = self.transform(mask)
        mask = mask[0, :]
        mask[mask > 0] = 1
        return im, mask.long()

    def __len__(self):
        assert len(self.list_rgb) == len(
            self.list_mask
        ), f"Image list and mask list should be the same number of images, but are {len(self.list_rgb)} and {len(self.list_mask)}"
        return len(self.list_rgb)

    @staticmethod
    def get_images_list(folder_path: Path, extension: str):
        folder_path = str(folder_path)
        print(folder_path)
        return glob(f"{folder_path}/*.{extension}")

    @classmethod
    def from_folder(
        cls,
        root: str,
        image_folder_name: str,
        mask_folder_name: str,
        extension="png",
        train: bool = True,
        transform: tf = None,
        image_size: tuple = (224, 224),
    ):

        data_root = Path(root)
        img_folder = data_root / Path(image_folder_name)
        lbl_folder = data_root / Path(mask_folder_name)

        list_rgb = SDataset.get_images_list(img_folder, extension)
        list_mask = SDataset.get_images_list(lbl_folder, extension)

        obj = cls(
            list_rgb, list_mask, train=train, transform=transform, image_size=image_size
        )
        return obj


if __name__ == "__main__":
    dataset = SDataset.from_folder(
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

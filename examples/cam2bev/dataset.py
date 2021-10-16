from pathlib import Path
import numpy as np
from PIL import Image
import torch
from examples.cam2bev.transform import Resize, parse_convert_xml, parse_convert_xml_v2
from nncore.segmentation.datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Cam2BEVDataset(torch.utils.data.Dataset):
    """Cam2BEV Dataset

    Ref: https://github.com/ika-rwth-aachen/Cam2BEV
    """

    def __init__(self, data_dir: str, num_classes: int = 10, raw: bool = False):
        super().__init__()
        self.data_paths = sorted(list(Path(data_dir).glob('*.npz')))
        # self.data_dir = Path(data_dir)
        self.num_classes = num_classes
        self.raw = raw
        # self.homography = sorted(list((self.data_dir / "homography").glob('*.png')))
        # self.bev_occl = sorted(list((self.data_dir / "bev+occlusion").glob('*.png')))
        # self.input_pallete = parse_convert_xml_v2('convert_10.xml')
        # print(self.input_pallete)
        # assert len(self.homography) == len(self.bev_occl)

    def __getitem__(self, idx):
        with np.load(self.data_paths[idx]) as data:
            image = data['image']  # 1, H, W
            mask = data['mask']    # 1, H, W
        # print(self.homography[idx])
        # print(self.bev_occl[idx])
        # homo = Image.open(self.homography[idx])        
        # bev = Image.open(self.bev_occl[idx])

        # # process here
        # image = np.array(homo)
        # mask = np.array(bev)  # H, W, 3

        # image = np.where(image == self.input_pallete[:, 1:])


        if self.raw:
            return (image, mask)

        image = self._to_one_hot(torch.tensor(image, dtype=torch.long))   # 1, H, W, C
        image = image.squeeze(0).permute(2, 0, 1)       # C, H, W
        mask = torch.tensor(mask).squeeze(0)            # H, W

        item = {"input": image.float(), "mask": mask.long()}
        return item

    def __len__(self):
        # return len(self.bev_occl)
        return len(self.data_paths)

    def _to_one_hot(self, x: torch.Tensor):
        scatter_dim = len(x.size())
        x_tensor = x.view(*x.size(), -1)
        zeros = torch.zeros(*x.size(), self.num_classes, dtype=x.dtype)
        return zeros.scatter(scatter_dim, x_tensor, 1)

def main():
    dataset = Cam2BEVDataset('./preprocess_np/val', num_classes=10, raw=True)
    print(len(dataset))
    # raw_image_dir = Path('/media/vinhloiit/Data/DiRA/cam2bev-data_2_F/cam2bev-data/2_F/val/homography')
    # raw_images = sorted(list(raw_image_dir.glob('*.png')))

    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    index = 0
    # raw_image = plt.imread(raw_images[index])
    item = dataset[index]
    inputs = item[0].squeeze(0).astype(np.uint8)
    mask = item[1].squeeze(0).astype(np.uint8)
    plt.figure(figsize=(20,10))
    plt.subplot(131)
    plt.title('RAW')
    # plt.imshow(raw_image)
    plt.subplot(132)
    plt.title('Input')
    plt.imshow(inputs)
    plt.subplot(133)
    plt.title('Mask')
    plt.imshow(mask)
    plt.savefig(f'item_{index}.png')


if __name__ == '__main__':
    main()

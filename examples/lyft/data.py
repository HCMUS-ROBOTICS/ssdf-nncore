from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from nncore.segmentation.datasets import DATASET_REGISTRY
from nncore.segmentation.utils import tensor2cmap
from nncore.utils.utils import inverse_normalize_batch

if __name__ == "__main__":
    dataset = DATASET_REGISTRY.get('LyftDataset.from_folder')(
        root=".",
        mask_folder_name="CameraSeg",
        image_folder_name="CameraRGB",
        test=False,
    )

    im = dataset[0]["input"]
    mask = dataset[0]["mask"]
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4)
    for i, data in enumerate(dataloader):
        im = data["input"]
        target = data["mask"]

        im = inverse_normalize_batch(im)
        target = tensor2cmap(target)

        im = make_grid(im, nrow=1, normalize=True)
        target = make_grid(target, nrow=1, normalize=False)
        save_dir = Path("./")
        save_image(im, str(save_dir / "last_batch_inputs.png"))
        save_image(target.float(), str(save_dir / "last_batch_labels.png"))

        if i == 0:
            break

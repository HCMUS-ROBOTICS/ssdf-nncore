from pathlib import Path
import torch
from nncore.datasets import LyftDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from nncore.utils.utils import inverse_normalize_batch
from nncore.utils.utils import tensor2plt
import matplotlib.pyplot as plt
import numpy as np
from nncore.utils.logger import image_batch_show
from nncore.utils.segmentation import tensor2cmap
from torchvision.utils import save_image

from nncore.utils.registry import DATASET_REGISTRY

# from
# create a pytorch transform
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
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
        im = data["input"]  # .permute(1, 2, 0).numpy()
        target = data["mask"]

        im = inverse_normalize_batch(im)
        target = tensor2cmap(target)

        im = image_batch_show(im, normalize=True)
        target = image_batch_show(target, normalize=False)
        save_dir = Path("./")
        save_image(im, str(save_dir / "last_batch_inputs.png"))
        save_image(target, str(save_dir / "last_batch_labels.png"))

        # im = tensor2plt(im, title="input")
        # target = tensor2plt(target.long(), title="input")
        plt.show()
        if i == 0:
            break


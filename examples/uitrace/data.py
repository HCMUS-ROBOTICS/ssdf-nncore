from pathlib import Path

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from nncore.core.transforms.albumentation import TRANSFORM_REGISTRY
from nncore.segmentation.datasets import DATASET_REGISTRY
from nncore.segmentation.utils import tensor2cmap
from nncore.utils.getter import get_instance_recursively
from nncore.utils.loading import load_yaml
from nncore.utils.utils import inverse_normalize_batch, tensor2plt

if __name__ == "__main__":
    dataset = DATASET_REGISTRY.get("LyftDataset.from_folder")(
        root="../../../unitydatasets/FPT_Dataset_v2/",
        mask_folder_name="SemanticSegmentation_Front",
        image_folder_name="RGB_Front",
        test=False,
    )
    transform_cfg = load_yaml("transform.yml")
    transform = get_instance_recursively(transform_cfg, registry=TRANSFORM_REGISTRY)
    dataset.transform = transform["train"]
    im = dataset[0]["input"]
    mask = dataset[0]["mask"]
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    for data in dataloader:
        im = data["input"]
        target = data["mask"]

        im = inverse_normalize_batch(im)
        target = tensor2cmap(target, label_format="cityscape")
        im = make_grid(im, nrow=2, normalize=True)
        target = make_grid(target, nrow=2, normalize=False)
        save_dir = Path("./")

        im = tensor2plt(im, title="inputs")
        target = tensor2plt(target.long(), title="labels")
        plt.show()
        break

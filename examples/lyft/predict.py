import sys
from importlib import import_module
from re import I

sys.path.insert(0, "../../")

from pathlib import Path

from nncore.models.wrapper import SegmentationModel
from torchvision.utils import save_image

from nncore.utils import get_device, image_batch_show, load_model, load_yaml
from nncore.utils.getter import get_instance
from nncore.utils.segmentation import multi_class_prediction

if __name__ == "__main__":
    checkpoint_folder = Path("./runs/default_2021_09_18-20_28_41/checkpoints")
    cfg = load_yaml(checkpoint_folder / "config.yaml")
    model = get_instance(cfg["pipeline"]["model"])
    load_model(model, checkpoint_folder / "best_loss.pth")
    device = get_device()
    inference_model = SegmentationModel(model, None)
    rgbs, pred = inference_model.predict(
        [
            "./data/images/1.png",
            "./data/images/1.png",
            "./data/images/1.png",
            "./data/images/1.png",
        ],
        device=device,
        batch_size=2,
        return_inp=True,
    )
    print(rgbs.shape)
    print(pred.shape)

    save_dir = Path("./demo")
    save_dir.mkdir(parents=True, exist_ok=True)
    pred = multi_class_prediction(pred).unsqueeze(1)
    outs = image_batch_show(pred)
    rgbs = image_batch_show(rgbs)

    save_image(rgbs, str(save_dir / "pred.png"), normalize=True)
    save_image(outs, str(save_dir / "rbgs.png"))

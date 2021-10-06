from numpy import tile
from ..datasets import *
from ..utils.typing import *
from ..utils import *
from ..schedulers import *
from ..utils.utils import inverse_normalize_batch

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn import Module
from torch import device

# from ..utils.cityscapes import cityscapes_map
from .supervisedlearner import SupervisedLearner
from ..metrics import Metric

from torchvision.utils import save_image


class SegmentationLearner(SupervisedLearner):
    r"""Segmentation learner class 

    Support training and evaluate strategy for supervise learning 
    
    Args:
        cfg (Any): [description]
        save_dir (str): save folder directory path
        train_data (DataLoader): train dataloader
        val_data (DataLoader): validation dataloader
        device (torch.device): training device
        model (Module): model to optimize
        scheduler (lr_scheduler): learning rate scheduler  
        optimizer (torch.optim.Optimizer): optimizer 
        metrics (Dict[str, Metric]): evaluate metrics
    """

    def __init__(
        self,
        cfg: Any,
        train_data: DataLoader,
        val_data: DataLoader,
        metrics: Dict[str, Metric],
        model: Module,
        scheduler: lr_scheduler,
        optimizer: Optimizer,
        device: device = get_device(),
    ):
        super().__init__(
            cfg=cfg,
            train_data=train_data,
            val_data=val_data,
            metrics=metrics,
            model=model,
            scheduler=scheduler,
            optimizer=optimizer,
            device=device,
        )

    def save_result(self, pred, batch, stage: str):
        input_key = "input"
        label_key = "mask"
        save_dir = self.save_dir / "samples"
        pred = pred["out"] if isinstance(pred, Dict) else pred  # B x N_CLS x W x H

        images = batch[input_key]  # B x C x W x H
        mask = batch[label_key]  # B x W x H
        pred = multi_class_prediction(pred)  # From B x N_CLS x W x H -> B x W x H

        images = inverse_normalize_batch(images)

        mask = tensor2cmap(mask)
        pred = tensor2cmap(pred)

        rgbs = image_batch_show(images, normalize=True)
        lbls = image_batch_show(mask, normalize=False)
        outs = image_batch_show(pred, normalize=False)

        save_image(rgbs, str(save_dir / "last_batch_inputs.png"))
        save_image(lbls, str(save_dir / "last_batch_labels.png"))
        save_image(outs, str(save_dir / "last_batch_preds.png"))

        rbgs_plt = tensor2plt(rgbs, title="inputs")
        lbls_plt = tensor2plt(lbls.long(), title="labels")
        outs_plt = tensor2plt(outs.long(), title="predictions")

        self.tsboard.update_figure(
            f"{stage}/samples/last_batch ",
            [rbgs_plt, lbls_plt, outs_plt],
            step=self.epoch,
        )


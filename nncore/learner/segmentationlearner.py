from ..datasets import *
from ..utils.typing import *
from ..utils import *
from ..schedulers import *

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn import Module
from torch import device


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

    def save_result(self, pred, batch, input_key="input", label_key="mask"):
        save_dir = self.save_dir / "samples"

        images = batch[input_key]
        mask = batch[label_key].unsqueeze(1)
        pred = multi_class_prediction(pred).unsqueeze(1)
        outs = image_batch_show(pred)
        rgbs = image_batch_show(images)
        lbls = image_batch_show(mask)
        save_image(rgbs, str(save_dir / "last_batch_inputs.png"), normalize=True)
        save_image(lbls, str(save_dir / "last_batch_labels.png"))
        save_image(outs, str(save_dir / "last_batch_preds.png"))
        self.logger.update_figure(
            f"samples/{self.epoch}_{self.iters}_{idx}", fig_cam, step=self.epoch
        )


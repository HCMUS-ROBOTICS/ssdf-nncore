from typing import Any, Dict, Optional

import numpy as np
import torch
import torchvision
from torch import device
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from nncore.core.learner.supervisedlearner import SupervisedLearner
from nncore.core.metrics.metric_template import Metric
from nncore.segmentation.learner import LEARNER_REGISTRY
from nncore.segmentation.utils import color_map
from nncore.utils.device import get_device
from nncore.utils.utils import inverse_normalize_batch, tensor2plt


@LEARNER_REGISTRY.register()
class Cam2BEVLearner(SupervisedLearner):
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

    def save_result(self, pred, batch, stage: str):
        images = batch['input']     # B,n_class,H,W
        images = images.argmax(dim=1).float()
        mask = batch['mask']        # B,H,W
        save_dir = self.save_dir / "samples"
        pred = pred["out"] if isinstance(pred, Dict) else pred  # B x N_CLS x W x H
        pred = torch.argmax(pred, dim=1)  # From B x N_CLS x W x H -> B x W x H

        images = self._tensor2cmap(images)
        mask = self._tensor2cmap(mask)
        pred = self._tensor2cmap(pred)

        rgbs = self._image_batch_show(images, normalize=False)
        lbls = self._image_batch_show(mask, normalize=False)
        outs = self._image_batch_show(pred, normalize=False)

        save_image(rgbs, str(save_dir / "last_batch_inputs.png"))
        save_image(lbls, str(save_dir / "last_batch_labels.png"))
        save_image(outs, str(save_dir / "last_batch_preds.png"))

        rbgs_plt = tensor2plt(rgbs, title="inputs")
        lbls_plt = tensor2plt(lbls.long(), title="labels")
        outs_plt = tensor2plt(outs.long(), title="predictions")

        self.tsboard.update_figure(
            f"{stage}/samples/last_batch",
            [rbgs_plt, lbls_plt, outs_plt],
            step=self.epoch,
        )

    def _image_batch_show(self, batch, ncol=5, fig_size=(30, 10), normalize=False):
        grid_img = torchvision.utils.make_grid(batch, nrow=ncol, normalize=normalize)
        return grid_img.float().cpu()

    def _tensor2cmap(self, tensor):
        """
        input: Tensor batch B x H x W
        output: color map batch  B x C x H x W
        """
        np_image = tensor.permute(1, 2, 0).cpu().numpy()
        return torch.tensor(self._np2cmap(np_image)).permute(2, 3, 0, 1)

    def _np2cmap(self, np_image):
        """
        Using Dot product to compute the color batch
        1. Create a new color axis from H x W x B to H x W x B x 1
        2. Multiply the color axis by the image tensor H x W x B x 1 and 1 x C (3 with RGB)
        3. Sum the color axis to get the final color map H x W x B x C
        input: numpy batch H x W x B
        output: color map batch  H x W x B x C
        """
        cmap = color_map()[:, np.newaxis, :]
        np_image = np_image[..., np.newaxis]
        new_im = np.dot(np_image == 0, cmap[0])
        for i in range(1, cmap.shape[0]):
            new_im += np.dot(np_image == i, cmap[i])
        return new_im

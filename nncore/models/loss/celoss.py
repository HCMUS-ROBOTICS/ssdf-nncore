from typing import Dict
from torch.functional import Tensor
from torch.nn import functional as F

from torch import nn
import torch


class CEwithstat(nn.Module):
    r"""CEwithstat is warper of cross-entropy loss"""

    def __init__(self):
        super(CEwithstat, self).__init__()

    def forward(self, pred, batch):
        pred = pred["out"] if isinstance(pred, Dict) else pred
        # in torchvision models, pred is a dict[key=out, value=Tensor]
        target = batch["mask"] if isinstance(batch, Dict) else batch
        # custom label is storaged in batch["mask"]
        print("CEwithstat: pred:", pred.shape, "target:", target.shape)
        loss = F.cross_entropy(pred, target)
        loss_dict = {"loss": loss}
        return loss, loss_dict

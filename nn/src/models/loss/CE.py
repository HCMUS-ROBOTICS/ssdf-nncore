from torch.nn import functional as F

from torch import nn
import torch


class CE(nn.Module):
    """Some Information about CE"""

    def __init__(self):
        super(CE, self).__init__()

    def forward(self, pred, batch):
        loss = F.cross_entropy(pred, batch["mask"])
        loss_dict = {"loss": loss}
        return loss, loss_dict

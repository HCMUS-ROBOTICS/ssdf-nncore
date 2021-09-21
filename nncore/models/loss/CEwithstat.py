from torch.nn import functional as F

from torch import nn
import torch


class CEwithstat(nn.Module):
    r"""CEwithstat is warper of cross-entropy loss"""

    def __init__(self):
        super(CEwithstat, self).__init__()

    def forward(self, pred, batch):
        loss = F.cross_entropy(pred, batch["mask"])
        loss_dict = {"loss": loss}
        return loss, loss_dict

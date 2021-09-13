import torch
from torch.nn import Module


class ModelWithLoss(torch.nn.Module):
    """Model with loss wrapper 

    Args:
        model (Module): Base Model without loss
        loss (Module): Base loss function with stat

    Example: 

        from nncore.model.segmentation import MobileUnet
        from nncore.model.loss import CEwithstat 
        from nncore.model import ModelWithLoss

        model = MobileUnet()
        loss = CEwithstat(nclasses = 2)

        modelwithloss = ModelWithLoss(model = model, loss = loss) 
    """

    def __init__(self, model: Module, loss: Module):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        outputs = self.model(batch["input"])
        loss, loss_dict = self.loss(outputs, batch)
        return outputs, loss, loss_dict


from typing import Optional
import torch
from torch import nn
from torch.nn import Module
from torchvision.transforms import transforms as tf

from ..datasets.default_datasets import TestImageDataset
from ..utils import *

from tqdm import tqdm


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


class SegmentationModel(nn.Module):
    """Some Information about SegmentationModel"""

    def __init__(
        self,
        model: Module,
        loss: Optional[Module] = None,
        transform: Optional[Module] = None,
    ):
        super(SegmentationModel, self).__init__()
        self.model = model
        self.loss = loss
        self.transform = transform

    def forward(self, batch):
        outputs = self.model(batch["input"])
        loss, loss_dict = self.loss(outputs, batch)
        return outputs, loss, loss_dict

    def predict(
        self,
        image_list,
        device,
        image_size=(224, 224),
        batch_size=1,
        verbose=True,
        return_inp=False,
    ):
        self.transform = (
            tf.Compose(
                [
                    tf.Resize(image_size),
                    tf.ToTensor(),
                    tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            if self.transform is None
            else self.transform
        )
        dataset = TestImageDataset(img_ls=image_list, transform=self.transform)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        progress_bar = tqdm(dataloader) if verbose else dataloader
        res = []
        inp = []
        for batch in progress_bar:
            batch = move_to(batch, device)
            outputs = self.model(batch["input"])
            outputs = detach(outputs)
            batch = detach(batch)
            res += [outputs]
            inp += [batch["input"]]
        if return_inp:
            return torch.cat(inp), torch.cat(res)
        return torch.cat(res)


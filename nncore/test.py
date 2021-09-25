from .utils import move_to, AverageValueMeter, detach
from .utils.typing import *

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm as tqdm

from torchvision import transforms as tf


@torch.no_grad()
def evaluate(
    model: Module,
    dataloader: DataLoader,
    metric: Metric,
    device: torch.device,
    verbose: bool = True,
):
    running_loss = AverageValueMeter()
    for m in metric.values():
        m.reset()
    model.eval()
    progress_bar = tqdm(dataloader) if verbose else dataloader
    for i, batch in enumerate(progress_bar):
        # 1: Load inputs and labels
        batch = move_to(batch, device)

        # 2: Get network outputs
        # 3: Calculate the loss
        outs, loss, loss_dict = model(batch)
        # 4: Update loss
        running_loss.add(loss.item())
        # 5: Update metric
        outs = detach(outs)
        batch = detach(batch)
        for m in metric.values():
            m.update(outs, batch)

    avg_loss = running_loss.value()[0]
    return avg_loss, metric


from .utils import move_to, AverageValueMeter, detach
from .utils.typing import *

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm


from .metrics.metric_template import Metric


@torch.no_grad()
def evaluate(
    model: Module,
    dataloader: DataLoader,
    metric: Metric,
    device: torch.device,
    verbose: bool = True,
    return_last_batch: bool = False,
):
    running_loss = AverageValueMeter()
    for m in metric.values():
        m.reset()
    model.eval()
    progress_bar = tqdm(dataloader) if verbose else dataloader
    for i, batch in enumerate(progress_bar):
        # 1: Load inputs and labels
        batch = move_to(batch, device)

        # 2: Calculate the loss
        outs, loss, loss_dict = model(batch)
        # 3: Update loss
        running_loss.add(loss.item())
        # 4: detach from gpu
        outs = detach(outs)
        batch = detach(batch)
        # 5: Update metric
        for m in metric.values():
            m.update(outs, batch)
    avg_loss = running_loss.value()[0]
    if return_last_batch:
        last_batch_pred = outs, batch
        return last_batch_pred, avg_loss, metric
    return avg_loss, metric

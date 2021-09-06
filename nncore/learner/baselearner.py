import torch
from ..utils.typing import *
from ..utils.logger import TensorboardLogger
from ..utils import vprint, AverageValueMeter, detach, move_to
from ..metrics import Metric

from pathlib import Path
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader


class BaseLearner:
    def __init__(
        self,
        cfg: Any,
        save_dir: str,
        train_data: DataLoader,
        val_data: DataLoader,
        device: torch.device,
        model: Module,
        scheduler: lr_scheduler,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, Metric],
        criterion: Optional[Module] = None,
        verbose: bool = True,
    ):
        self.train_data, self.val_data = train_data, val_data
        self.model, self.criterion, self.optimizer = model, criterion, optimizer
        self.save_dir = Path(save_dir)
        self.tsboard = TensorboardLogger(path=self.save_dir)
        self.device = device
        self.scaler = GradScaler(enabled=False)
        self.max_grad_norm = 1.0
        self.verbose = verbose
        self.metric = metrics
        self.best_metric = {k: 0.0 for k in self.metric.keys()}
        self.best_loss = np.inf
        self.scheduler = scheduler
        self.cfg = cfg

    def fit():
        raise NotImplementedError

    def train_epoch(self, epoch: int, dataloader: DataLoader) -> float:
        running_loss = AverageValueMeter()
        total_loss = AverageValueMeter()
        for m in self.metric.values():
            m.reset()
        self.model.train()
        self.print("Training........")
        progress_bar = tqdm(dataloader) if self.verbose else dataloader
        for i, batch in enumerate(progress_bar):
            # 1: Load img_inputs and labels
            batch = move_to(batch, self.device)

            # 2: Clear gradients from previous iteration
            self.optimizer.zero_grad()
            with autocast(enabled=self.cfg.fp16):
                # 3: Get network outputs
                # 4: Calculate the loss
                outs, loss, loss_dict = self.model(batch)
            # 5: Calculate gradients
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # 6: Performing backpropagation
            with torch.no_grad():
                # 7: Update loss
                running_loss.add(loss.item())
                total_loss.add(loss.item())

                if (i + 1) % self.cfg.log_step == 0 or (i + 1) == len(dataloader):
                    self.tsboard.update_loss(
                        "train", running_loss.value()[0], epoch * len(dataloader) + i
                    )
                    running_loss.reset()

                # 8: Update metric
                outs = detach(outs)
                batch = detach(batch)
                for m in self.metric.values():
                    m.update(outs, batch)

        avg_loss = total_loss.value()[0]
        return avg_loss

    def save_checkpoints():
        raise NotImplementedError

    def save_result():
        raise NotImplementedError

    def print(self, obj):
        vprint(obj, self.verbose)

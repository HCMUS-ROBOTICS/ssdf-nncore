import torch
from datasets import *
from utils.typing import *
from metrics import Metric
from utils import (
    vprint,
    get_device,
    move_to,
    save_model,
    load_model,
    detach,
    AverageValueMeter,
)
from test import evaluate
from utils.logger import TensorboardLogger
from scheduler import *

import numpy as np
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn import Module
from torch import device
from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm

from pathlib import Path


class BaseLearner:
    def __init__(
        self,
        train_data: DataLoader,
        val_data: DataLoader,
        model: Module,
        criterion: Module,
        optimizer: Optimizer,
        save_dir: str,
    ):
        self.train_data, self.val_data = train_data, val_data
        self.model, self.criterion, self.optimizer = model, criterion, optimizer
        self.save_dir = Path(save_dir)


class SupervisedLearner(BaseLearner):
    def __init__(
        self,
        train_data: DataLoader,
        val_data: DataLoader,
        model: Module,
        criterion: Module,
        optimizer: Optimizer,
        device: device = get_device(),
        load_path: Optional[str] = None,
        save_dir: str = "./",
    ):
        super().__init__(train_data, val_data, model, criterion, optimizer, save_dir)
        self.device = device

        if load_path is not None:
            load_model(self.model, load_path)
        self.tsboard = TensorboardLogger(path=self.save_dir)
        self.verbose = True
        self.max_grad_norm = 1.0

    def fit(
        self,
        n_epochs: int,
        metrics: Dict[str, Metric] = None,
        log_step=1,
        val_step=1,
        fp16: bool = False,
        verbose: bool = True,
    ) -> float:
        self.fp16 = fp16
        self.debug = False
        self.verbose = verbose
        self.metric = metrics
        self.best_loss = np.inf
        self.best_metric = {k: 0.0 for k in self.metric.keys()}
        self.log_step = log_step
        self.val_step = val_step
        self.scaler = GradScaler(enabled=self.fp16)
        self.scheduler = StepLR(self.optimizer, step_size=3, gamma=0.1)
        for epoch in range(n_epochs):
            self.print(f"\nEpoch {epoch:>3d}")
            self.print("-----------------------------------")

            # 1: Training phase
            # 1.1 train
            avg_loss = self.train_epoch(epoch=epoch, dataloader=self.train_data)

            # 1.2 log result
            self.print("+ Training result")
            self.print(f"Loss: {avg_loss}")
            for m in self.metric.values():
                m.summary()

            # 2: Evalutation phase
            if (epoch + 1) % self.val_step == 0:
                with autocast(enabled=self.fp16):
                    # 2: Evaluating model
                    avg_loss = self.evaluate(epoch, dataloader=self.val_data)

                    self.print("+ Evaluation result")
                    self.print(f"Loss: {avg_loss}")
                    for m in self.metric.values():
                        m.summary()

                    # 3: Learning rate scheduling
                    self.scheduler.step(avg_loss)

                    # 4: Saving checkpoints
                    if not self.debug:
                        # Get latest val loss here
                        val_metric = {k: m.value() for k, m in self.metric.items()}
                        self.save_checkpoint(epoch, avg_loss, val_metric)
            self.print("-----------------------------------")

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
            with autocast(enabled=self.fp16):
                # 3: Get network outputs
                outs, loss, loss_dict = self.model(batch)
                # 4: Calculate the loss
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

                if (i + 1) % self.log_step == 0 or (i + 1) == len(dataloader):
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

    def save_checkpoint(self, epoch, val_loss, val_metric):
        data = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        if val_loss < self.best_loss:
            self.print(
                f"Loss is improved from {self.best_loss: .6f} to {val_loss: .6f}. Saving weights...",
            )
            save_model(data, self.save_dir / Path("best_loss.pth"))
            # Update best_loss
            self.best_loss = val_loss
        else:
            self.print(f"Loss is not improved from {self.best_loss:.6f}.")

        for k in self.metric.keys():
            if val_metric[k] > self.best_metric[k]:
                self.print(
                    f"{k} is improved from {self.best_metric[k]: .6f} to {val_metric[k]: .6f}. Saving weights...",
                )
                save_model(data, self.save_dir / Path(f"best_metric_{k}.pth"))
                self.best_metric[k] = val_metric[k]
            else:
                self.print(f"{k} is not improved from {self.best_metric[k]:.6f}.")

    @torch.no_grad()
    def evaluate(self, epoch, dataloader):
        avg_loss, metric = evaluate(
            model=self.model,
            dataloader=dataloader,
            metric=self.metric,
            device=self.device,
            criterion=self.criterion,
            verbose=self.verbose,
        )
        self.metric = metric
        return avg_loss

    def print(self, obj):
        vprint(obj, self.verbose)

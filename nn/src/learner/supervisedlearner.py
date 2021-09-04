import torch
from datasets import *
from utils.typing import *
from metrics import Metric
from utils import (
    vprint,
    get_device,
    save_model,
    load_model,
)
from test import evaluate
from scheduler import *

import numpy as np
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn import Module
from torch import device
from torch.cuda.amp import autocast


from pathlib import Path
from .baselearner import BaseLearner


class SupervisedLearner(BaseLearner):
    def __init__(
        self,
        train_data: DataLoader,
        val_data: DataLoader,
        model: Module,
        optimizer: Optimizer,
        device: device = get_device(),
        load_path: Optional[str] = None,
        save_dir: str = "./",
    ):
        super().__init__(
            exp_id="default",
            opt=None,
            save_dir=save_dir,
            train_data=train_data,
            val_data=val_data,
            device=device,
            model=model,
            optimizer=optimizer,
            criterion=None,
        )
        if load_path is not None:
            load_model(self.model, load_path)
        self.verbose = True

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
        self.metric = metrics
        self.best_loss = np.inf
        self.best_metric = {k: 0.0 for k in self.metric.keys()}
        self.log_step = log_step
        self.val_step = val_step
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


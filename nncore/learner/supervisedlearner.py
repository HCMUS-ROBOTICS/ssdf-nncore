import torch
from ..datasets import *
from ..utils.typing import *
from ..utils import (
    get_device,
    save_model,
    load_checkpoint,
)
from ..test import evaluate
from ..schedulers import *

import numpy as np
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn import Module
from torch import device


from pathlib import Path
from torch.cuda.amp import GradScaler, autocast

from .baselearner import BaseLearner
from ..metrics import Metric


class SupervisedLearner(BaseLearner):
    r"""SupervisedLearner class 

    Support training and evaluate strategy for supervise learning 
    
    Args:
        cfg (Any): [description]
        save_dir (str): save folder directory path
        train_data (DataLoader): train dataloader
        val_data (DataLoader): validation dataloader
        device (torch.device): training device
        model (Module): model to optimize
        scheduler (lr_scheduler): learning rate scheduler  
        optimizer (torch.optim.Optimizer): optimizer 
        metrics (Dict[str, Metric]): evaluate metrics
    """

    def __init__(
        self,
        cfg: Any,
        train_data: DataLoader,
        val_data: DataLoader,
        metrics: Dict[str, Metric],
        model: Module,
        scheduler: lr_scheduler,
        optimizer: Optimizer,
        device: device = get_device(),
    ):
        super().__init__(
            save_dir=cfg.save_dir,
            train_data=train_data,
            val_data=val_data,
            device=device,
            model=model,
            metrics=metrics,
            scheduler=scheduler,
            optimizer=optimizer,
            criterion=None,
            cfg=cfg,
        )
        if cfg.pretrained is not None:
            cp = load_checkpoint(cfg.pretrained)
            self.model.model.load_state_dict(cp["model_state_dict"])
            if cfg.resume:
                self.optimizer.load_state_dict(["optimizer_state_dict"])
        self.verbose = cfg.verbose
        self.scaler = GradScaler(enabled=cfg.fp16)
        self.cfg = cfg

    def fit(self):
        for epoch in range(self.cfg.nepochs):
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
            if (epoch + 1) % self.cfg.val_step == 0:
                with autocast(enabled=self.cfg.fp16):
                    # 2: Evaluating model
                    avg_loss = self.evaluate(epoch, dataloader=self.val_data)

                    self.print("+ Evaluation result")
                    self.print(f"Loss: {avg_loss}")
                    for m in self.metric.values():
                        m.summary()

                    # 3: Learning rate scheduling
                    self.scheduler.step(avg_loss)

                    # 4: Saving checkpoints
                    if not self.cfg.debug:
                        # Get latest val loss here
                        val_metric = {k: m.value() for k, m in self.metric.items()}
                        self.save_checkpoint(epoch, avg_loss, val_metric)
            self.print("-----------------------------------")

    def save_checkpoint(
        self, epoch: int, val_loss: float, val_metric: Dict[str, float]
    ) -> None:
        r"""Save checkpoint method

        Saving 
        -   model state dict
        -   optimizer state dict
        Args:
            epoch (int): current epoch
            val_loss (float): validation loss
            val_metric (Dict[str, float]): validation metrics result
        """
        data = {
            "epoch": epoch,
            "model_state_dict": self.model.model.state_dict(),
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
            verbose=self.verbose,
        )
        self.metric = metric
        return avg_loss

    def save_result(self, pred, batch):
        save_dir = self.save_dir / Path("samples")

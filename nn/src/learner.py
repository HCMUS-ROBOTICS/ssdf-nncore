import torch
from torch.serialization import save
from .datasets import *
from .utils import get_device, load_model
from .metrics import Metric

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn import Module
from torch import Tensor, device
from typing import Dict

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


class Learner(BaseLearner):
    def __init__(
        self,
        train_data: DataLoader,
        val_data: DataLoader,
        model: Module,
        criterion: Module,
        optimizer: Optimizer,
        device: device = get_device(),
        load_path: str = None,
    ):
        super().__init__(train_data, val_data, model, criterion, optimizer)
        self._device = device

        if load_path is not None:
            load_model(self._model, load_path)

    def fit(self, n_epoch: int, metrics: Dict[str, Metric] = None,) -> float:
        raise NotImplementedError

    def save_checkpoint(self, epoch, val_loss, val_metric):
        data = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }

        if val_loss < self.best_loss:
            print(
                f"Loss is improved from {self.best_loss: .6f} to {val_loss: .6f}. Saving weights...",
            )
            torch.save(data, self.save_dir / Path("best_loss.pth"))
            # Update best_loss
            self.best_loss = val_loss
        else:
            print(f"Loss is not improved from {self.best_loss:.6f}.")

        for k in self.metric.keys():
            if val_metric[k] > self.best_metric[k]:
                print(
                    f"{k} is improved from {self.best_metric[k]: .6f} to {val_metric[k]: .6f}. Saving weights...",
                )
                torch.save(data, self.save_dir / Path(f"best_metric_{k}.pth"))
                self.best_metric[k] = val_metric[k]
            else:
                print(f"{k} is not improved from {self.best_metric[k]:.6f}.")

    def learn_one_iter(self, inputs: Tensor, labels: Tensor):
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self) -> float:
        raise NotImplementedError

    def compute_outputs(self, inputs: Tensor, train: bool) -> Tensor:
        raise NotImplementedError

    def compute_loss(self, inputs: Tensor, labels: Tensor, train: bool) -> Tensor:
        raise NotImplementedError


class DistillationLearner(Learner):
    """
    Distilling Knowledge from a big teacher network to a smaller model (UNTESTED)
    References:
        Geoffrey Hinton, Oriol Vinyals, Jeff Dean. "Distilling the Knowledge in a Neural Network."
        https://arxiv.org/abs/1503.02531
        TTIC Distinguished Lecture Series - Geoffrey Hinton.
        https://www.youtube.com/watch?v=EK61htlw8hY
    """

    def __init__(
        self,
        train_data: DataLoader,
        val_data: DataLoader,
        model: Module,
        teacher: Module,
        criterion: Module,
        optimizer: Optimizer,
        temperature: float,
        teacher_weight: float,
        hard_label_weight: float,
        device=get_device(),
    ):
        raise NotImplementedError

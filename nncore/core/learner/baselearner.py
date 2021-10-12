from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from nncore.core.logger import TensorboardLogger
from nncore.core.metrics import Metric
from nncore.utils.device import detach, move_to
from nncore.utils.meter import AverageValueMeter
from torch.cuda.amp import GradScaler, autocast
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm.auto import tqdm as tqdm


class BaseLearner:
    r"""BaseLearner class 

    Abstract learner class, support training and evaluate strategy.

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
        criterion (Optional[Module], optional): Loss function. Defaults to None.
        verbose (bool, optional): if verbose is False, model does not log anything during training process. Defaults to True.
    """

    def __init__(
        self,
        cfg: Any,
        save_dir: str,
        train_data: DataLoader,
        val_data: DataLoader,
        device: torch.device,
        model: Module,
        scheduler,
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
        (self.save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.save_dir / "samples").mkdir(parents=True, exist_ok=True)

    def fit():
        raise NotImplementedError

    def train_epoch(self, epoch: int, dataloader: DataLoader) -> float:
        """Training epoch

        Args:
            epoch (int): [description]
            dataloader (DataLoader): [description]

        Returns:
            float: [description]
        """
        running_loss = AverageValueMeter()
        total_loss = AverageValueMeter()
        for m in self.metric.values():
            m.reset()
        self.model.train()
        print("Training........")
        progress_bar = tqdm(dataloader) if self.verbose else dataloader
        for i, batch in enumerate(progress_bar):
            # 1: Load img_inputs and labels
            batch = move_to(batch, self.device)

            # 2: Clear gradients from previous iteration
            self.optimizer.zero_grad()
            with autocast(enabled=self.cfg.fp16):
                # 3: Get network outputs
                # 4: Calculate the loss
                out_dict = self.model(batch)
            # 5: Calculate gradients
            self.scaler.scale(out_dict['loss']).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # 6: Performing backpropagation
            with torch.no_grad():
                # 7: Update loss
                running_loss.add(out_dict['loss'].item())
                total_loss.add(out_dict['loss'].item())

                if (i + 1) % self.cfg.log_step == 0 or (i + 1) == len(dataloader):
                    self.tsboard.update_loss(
                        "train", running_loss.value(
                        )[0], epoch * len(dataloader) + i
                    )
                    running_loss.reset()

                # 8: Update metric
                outs = detach(out_dict)
                batch = detach(batch)
                for m in self.metric.values():
                    m.update(outs['out'], batch)
        self.save_result(outs, batch, stage="train")
        avg_loss = total_loss.value()[0]
        return avg_loss

    def save_checkpoints():
        raise NotImplementedError

    def save_result(self, pred, batch, stage: str, **kwargs):
        NotImplemented

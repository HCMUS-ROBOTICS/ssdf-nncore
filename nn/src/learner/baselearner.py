import torch
from utils.typing import *
from utils.logger import TensorboardLogger
from pathlib import Path
from tqdm import tqdm
from utils import vprint, AverageValueMeter, detach, move_to
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader


class BaseLearner:
    def __init__(
        self,
        exp_id: str,
        opt: Any,
        save_dir: str,
        train_data: DataLoader,
        val_data: DataLoader,
        device: torch.device,
        model: Module,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[Module] = None,
        verbose: bool = True,
    ):
        self.train_data, self.val_data = train_data, val_data
        self.model, self.criterion, self.optimizer = model, criterion, optimizer
        self.save_dir = Path(save_dir) / exp_id
        self.tsboard = TensorboardLogger(path=self.save_dir)
        self.opt = opt
        self.device = device
        self.scaler = GradScaler(enabled=False)
        self.max_grad_norm = 1.0
        self.verbose = verbose

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
            with autocast(enabled=self.fp16):
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

    def save_checkpoints():
        raise NotImplementedError

    def save_result():
        raise NotImplementedError

    def print(self, obj):
        vprint(obj, self.verbose)

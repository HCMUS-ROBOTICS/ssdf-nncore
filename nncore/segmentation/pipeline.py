import logging
from typing import Optional

import yaml
from nncore.core.models.wrapper import ModelWithLoss
from nncore.core.opt import opts
from nncore.core.test import evaluate
from nncore.segmentation.criterion import CRITERION_REGISTRY
from nncore.segmentation.datasets import DATASET_REGISTRY
from nncore.segmentation.learner import LEARNER_REGISTRY
from nncore.segmentation.metrics import METRIC_REGISTRY
from nncore.segmentation.models import MODEL_REGISTRY
from nncore.utils.getter import (get_dataloader, get_dataset_size,
                                 get_instance, get_single_data)
from nncore.utils.loading import load_yaml
from torch.utils.data.dataset import random_split
from torchvision.transforms import transforms as tf


class Pipeline(object):
    """docstring for Pipeline."""

    def __init__(self, opt: opts, cfg_path: Optional[str] = None):
        super(Pipeline, self).__init__()
        self.opt = opt
        assert (cfg_path is not None) or (
            opt.cfg_pipeline is not None
        ), "learner params is none, \ please create config file follow default format. \n You could find an example in nn/configs/default/learner.yaml"
        self.cfg = (
            load_yaml(cfg_path) if cfg_path is not None else load_yaml(opt.cfg_pipeline)
        )

        self.device = get_instance(self.cfg["device"])
        print(self.device)

        self.train_dataloader, self.val_dataloader = self.get_data(
            self.cfg["data"], return_dataset=False
        )

        model = get_instance(self.cfg["model"], registry=MODEL_REGISTRY).to(self.device)
        criterion = get_instance(self.cfg["criterion"], registry=CRITERION_REGISTRY).to(self.device)
        self.model = ModelWithLoss(model, criterion)

        self.metric = {mcfg["name"]: get_instance(
            mcfg, registry=METRIC_REGISTRY) for mcfg in self.cfg["metric"]}

        self.optimizer = get_instance(
            self.cfg["optimizer"], params=self.model.parameters()
        )

        self.scheduler = get_instance(
            self.cfg["scheduler"], optimizer=self.optimizer)

        self.learner = get_instance(
            self.cfg["learner"],
            cfg=self.opt,
            train_data=self.train_dataloader,
            val_data=self.val_dataloader,
            scheduler=self.scheduler,
            model=self.model,
            metrics=self.metric,
            optimizer=self.optimizer,
            registry=LEARNER_REGISTRY,
        )

        save_cfg = {}
        save_cfg["opt"] = vars(opt)
        save_cfg["pipeline"] = self.cfg
        save_cfg["opt"]["save_dir"] = str(save_cfg["opt"]["save_dir"])
        with open(
            self.learner.save_dir / "checkpoints" / "config.yaml", "w"
        ) as outfile:
            yaml.dump(save_cfg, outfile, default_flow_style=False)
        self.logger = logging.getLogger()

    def sanitycheck(self):
        self.logger.info("Sanity checking before training")
        self.evaluate()

    def fit(self):
        self.sanitycheck()
        self.learner.fit()

    def evaluate(self):
        avg_loss, metric = evaluate(
            model=self.model,
            dataloader=self.val_dataloader,
            metric=self.metric,
            device=self.device,
            verbose=self.opt.verbose,
        )
        print("Evaluate result")
        print(f"Loss: {avg_loss}")
        for m in metric.values():
            m.summary()

    def get_data(self, cfg, return_dataset=False):
        if cfg.get("train", False) and cfg.get("val", False):
            train_dataloader, train_dataset = get_single_data(
                cfg["train"], return_dataset=True
            )
            val_dataloader, val_dataset = get_single_data(cfg["val"], return_dataset=True)
        elif cfg.get("trainval", False):
            trainval_cfg = cfg["trainval"]
            # Split dataset train:val = ratio:(1-ratio)
            ratio = trainval_cfg["test_ratio"]
            dataset = get_instance(trainval_cfg["dataset"], registry=DATASET_REGISTRY)
            train_sz, val_sz = get_dataset_size(ratio=ratio, dataset_sz=len(dataset))
            train_dataset, val_dataset = random_split(dataset, [train_sz, val_sz])
            # Get dataloader
            train_dataloader = get_dataloader(
                trainval_cfg["loader"]["train"], train_dataset
            )
            val_dataloader = get_dataloader(trainval_cfg["loader"]["val"], val_dataset)
        else:
            raise Exception("Dataset config is not correctly formatted.")
        return (
            (train_dataloader, val_dataloader, train_dataset, val_dataset)
            if return_dataset
            else (train_dataloader, val_dataloader)
        )

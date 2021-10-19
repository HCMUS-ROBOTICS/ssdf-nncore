import logging
from typing import Callable, Dict, Optional

import yaml
from torch.utils.data.dataset import random_split

from nncore.core.models.wrapper import ModelWithLoss
from nncore.core.opt import Opts
from nncore.core.test import evaluate
from nncore.core.transforms.albumentation import TRANSFORM_REGISTRY
from nncore.segmentation.criterion import CRITERION_REGISTRY
from nncore.segmentation.datasets import DATASET_REGISTRY
from nncore.segmentation.learner import LEARNER_REGISTRY
from nncore.segmentation.metrics import METRIC_REGISTRY
from nncore.segmentation.models import MODEL_REGISTRY
from nncore.utils.getter import (get_dataloader, get_instance,
                                 get_instance_recursively)
from nncore.utils.loading import load_yaml


class Pipeline(object):
    """docstring for Pipeline."""

    def __init__(self,
                 opt: Opts,
                 cfg_path: Optional[str] = None,
                 transform_cfg_path: Optional[str] = None):
        super(Pipeline, self).__init__()
        self.opt = opt
        assert (cfg_path is not None) or (
            opt.cfg_pipeline is not None
        ), ("learner params is none, \n please create config file follow default format.")
        self.cfg = (
            load_yaml(cfg_path) if cfg_path is not None else load_yaml(opt.cfg_pipeline)
        )

        assert (transform_cfg_path is not None) or (
            opt.cfg_transform is not None
        ), "learner params is none, \n please create config file follow default format."
        if transform_cfg_path is not None:
            self.transform_cfg = load_yaml(transform_cfg_path)
        else:
            self.transform_cfg = load_yaml(opt.cfg_transform)

        self.device = get_instance(self.cfg["device"])
        print(self.device)

        self.transform = None
        if self.transform_cfg is not None:
            self.transform = get_instance_recursively(self.transform_cfg,
                                                      registry=TRANSFORM_REGISTRY)

        data = self.get_data(self.cfg["data"], self.transform, return_dataset=False)
        self.train_dataloader, self.val_dataloader, self.train_dataset, self.val_dataset = data

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

    def get_data(self, cfg, transform: Optional[Dict[str, Callable]] = None, return_dataset=False):
        def get_single_data(cfg, transform, stage: str = 'train'):
            assert stage in cfg['dataset'].keys(), f"{stage} is not in dataset config"
            assert stage in cfg['loader'].keys(), f"{stage} is not in loader config"

            if transform is None:
                dataset = get_instance(cfg['dataset'][stage], registry=DATASET_REGISTRY)
            else:
                dataset = get_instance(cfg['dataset'][stage],
                                       registry=DATASET_REGISTRY,
                                       transform=transform[stage])
            dataloader = get_dataloader(cfg['loader'][stage], dataset)
            return dataloader, dataset

        train_dataloader, train_dataset = None, None
        if ('train' in cfg['dataset']) and ('train' in cfg['loader']):
            train_dataloader, train_dataset = get_single_data(cfg, transform, 'train')

        val_dataloader, val_dataset = None, None
        if ('val' in cfg['dataset']) and ('val' in cfg['loader']):
            val_dataloader, val_dataset = get_single_data(cfg, transform, 'val')

        if train_dataloader is None and val_dataloader is None:
            dataset = get_instance(cfg['dataset'], registry=DATASET_REGISTRY, transform=None)
            train_sz, val_sz = cfg['splits']['train'], cfg['splits']['val']
            train_sz = int(len(dataset) * train_sz)
            val_sz = len(dataset) - train_sz
            assert val_sz > 0, f'validation size must be greater than 0. val_sz = {val_sz}'
            train_dataset, val_dataset = random_split(dataset, [train_sz, val_sz])
            if transform is not None:
                train_dataset.dataset.transform = transform['train']
                val_dataset.dataset.transform = transform['val']
            train_dataloader = get_dataloader(cfg['loader']['train'], train_dataset)
            val_dataloader = get_dataloader(cfg['loader']['val'], val_dataset)

        return (train_dataloader, val_dataloader, train_dataset, val_dataset)

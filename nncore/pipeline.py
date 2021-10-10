import logging
from typing import Optional

import yaml
from torchvision.transforms import transforms as tf

from nncore.models.wrapper import ModelMixin
from nncore.utils.registry import MODEL_REGISTRY

from .opt import opts
from .test import evaluate
from .utils import load_yaml
from .utils.getter import get_data, get_instance


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
        self.transform = [
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        self.train_dataloader, self.val_dataloader = get_data(
            self.cfg["data"], return_dataset=False
        )
        # model: core-model + loss function
        ## train: loss + predicts
        ## eval: loss + predicts
        # detectron2: model: build_backbone (registry) + build_heads + build_loss

        # self.model = ModelMixin(model, criterion)

        # model = get_instance(self.cfg["model"]).to(self.device)
        # criterion = get_instance(self.cfg["criterion"]).to(self.device)
        self.model = get_instance(self.cfg["model"], getter=get_instance).to(self.device)

        self.metric = {mcfg["name"]: get_instance(mcfg) for mcfg in self.cfg["metric"]}

        self.optimizer = get_instance(
            self.cfg["optimizer"], params=self.model.model.parameters()
        )

        self.scheduler = get_instance(self.cfg["scheduler"], optimizer=self.optimizer)

        self.learner = get_instance(
            self.cfg["learner"],
            cfg=self.opt,
            train_data=self.train_dataloader,
            val_data=self.val_dataloader,
            scheduler=self.scheduler,
            model=self.model,
            metrics=self.metric,
            optimizer=self.optimizer,
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

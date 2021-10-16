from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, random_split
from nncore.core.models.wrapper import ModelWithLoss
from nncore.core.datasets import DATASET_REGISTRY
from nncore.core.metrics import METRIC_REGISTRY
from nncore.utils.device import get_device


def get_instance(config, registry=None, **kwargs):
    # ref https://github.com/vltanh/torchan/blob/master/torchan/utils/getter.py
    assert "name" in config
    config.setdefault("args", {})
    if config.get("args", None) is None:
        config["args"] = {}

    if registry:
        return registry.get(config['name'])(**config['args'], **kwargs)

    return globals()[config["name"]](**config["args"], **kwargs)


def get_function(name):
    return globals()[name]


def get_dataloader(cfg, dataset):
    collate_fn = None
    if cfg.get("collate_fn", False):
        collate_fn = get_function(cfg["collate_fn"])

    dataloader = get_instance(cfg, dataset=dataset, collate_fn=collate_fn)
    return dataloader


def get_single_data(cfg, return_dataset=True):
    dataset = get_instance(cfg, registry=DATASET_REGISTRY)
    dataloader = get_dataloader(cfg["loader"], dataset)
    return dataloader, dataset if return_dataset else dataloader


def get_dataset_size(ratio: float, dataset_sz: int):
    val_sz = max(1, int(ratio * dataset_sz))
    train_sz = dataset_sz - val_sz
    return train_sz, val_sz


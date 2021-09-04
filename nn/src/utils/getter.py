from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from datasets import *
from models import *
from metrics import *
from externals import *


def get_instance(config, **kwargs):
    # ref https://github.com/vltanh/torchan/blob/master/torchan/utils/getter.py
    assert "name" in config
    config.setdefault("args", {})
    if config["args"] is None:
        config["args"] = {}
    return globals()[config["name"]](**config["args"], **kwargs)

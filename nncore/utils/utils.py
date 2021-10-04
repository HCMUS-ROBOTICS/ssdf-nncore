import torch
import copy
from .typing import *
import yaml
import matplotlib.pyplot as plt


def copy_model(model: Module) -> Module:
    return copy.deepcopy(model)


def save_model(data: Dict[str, Any], path: str, verbose: bool = True):
    torch.save(data, path)
    if verbose:
        print("Model saved")


def load_checkpoint(path: str, map_location="cpu"):
    return torch.load(path, map_location=map_location)


def load_model(
    model: Module,
    path: str,
    verbose: bool = True,
    map_location="cpu",
    key="model_state_dict",
):
    model.load_state_dict(torch.load(path, map_location=map_location)[key])
    if verbose:
        print("Model loaded")


def load_yaml(cfg_path: str):
    return yaml.load(open(cfg_path, "r"), Loader=yaml.Loader)


def vprint(obj: str, verbose: bool):
    if verbose:
        print(obj)



def tensor2plt(obj: Tensor, title: List[Any]):
    fig = plt.figure()
    plt.imshow(obj.permute(1, 2, 0))
    plt.title(title)
    return fig

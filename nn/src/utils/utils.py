import torch
import copy
from .typing import *


def copy_model(model: Module) -> Module:
    return copy.deepcopy(model)


def save_model(data: Dict[str, Any], path: str, verbose: bool = True):
    torch.save(data, path)
    if verbose:
        print("Model saved")


def load_model(model: Module, path: str, verbose: bool = True, map_location="cpu"):
    model.load_state_dict(
        torch.load(path, map_location=map_location)["model_state_dict"]
    )
    if verbose:
        print("Model loaded")


def vprint(obj: str, verbose: bool):
    if verbose:
        print(obj)

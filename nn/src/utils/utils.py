import torch
import copy
from torch.nn import Module


def copy_model(model: Module) -> Module:
    return copy.deepcopy(model)


def save_model(model: Module, path: str, verbose: bool = True):
    torch.save(model.state_dict(), path)
    if verbose:
        print("Model saved")


def load_model(model: Module, path: str, verbose: bool = True, map_location="cpu"):
    model.load_state_dict(
        torch.load(path, map_location=map_location)  # ["model_state_dict"]
    )
    if verbose:
        print("Model loaded")


def vprint(obj, vb):
    if vb:
        print(obj)

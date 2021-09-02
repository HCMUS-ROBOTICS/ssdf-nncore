import torch
import copy
from torch.nn import Module


def copy_model(model: Module) -> Module:
    return copy.deepcopy(model)


def save_model(model: Module, path: str, verbose: bool = True):
    torch.save(model.state_dict(), path)
    if verbose:
        print("Model saved")


def load_model(model: Module, path: str, verbose: bool = True):
    model.load_state_dict(torch.load(path)["model_state_dict"])
    if verbose:
        print("Model loaded")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


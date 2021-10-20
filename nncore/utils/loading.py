from typing import Any, Dict

import torch
import torch.nn as nn
import yaml


def load_yaml(path):
    with open(path, 'rt') as f:
        return yaml.safe_load(f)


def load_checkpoint(path: str, map_location="cpu"):
    return torch.load(path, map_location=map_location)


def load_model(
    model: nn.Module,
    path: str,
    map_location="cpu",
    key="model_state_dict",
):
    model.load_state_dict(torch.load(path, map_location=map_location)[key])


def save_model(data: Dict[str, Any], path: str):
    torch.save(data, path)

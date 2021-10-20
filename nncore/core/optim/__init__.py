from torch.optim import SGD, Adam

from nncore.core.registry import Registry

from . import lr_scheduler

OPTIM_REGISTRY = Registry('OPTIMIZER')
OPTIM_REGISTRY.register(Adam)
OPTIM_REGISTRY.register(SGD)

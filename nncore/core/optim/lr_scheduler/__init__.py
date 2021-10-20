from nncore.core.registry import Registry

SCHEDULER_REGISTRY = Registry('LR_SCHEDULER')

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

SCHEDULER_REGISTRY.register(StepLR)
SCHEDULER_REGISTRY.register(ReduceLROnPlateau)

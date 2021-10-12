import torch.nn as nn
from nncore.utils.registry import MODEL_REGISTRY
from torchvision.models.segmentation import deeplabv3_resnet50

MODEL_REGISTRY.register(deeplabv3_resnet50)
from torchvision.models.segmentation import deeplabv3_resnet50

from nncore.core.models import MODEL_REGISTRY

MODEL_REGISTRY.register(deeplabv3_resnet50)

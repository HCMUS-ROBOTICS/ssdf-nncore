import torch
from torchvision import models

from nncore.segmentation.models import MODEL_REGISTRY


@MODEL_REGISTRY.register()
def build_deeplabv3_cam2bev(in_channels, pretrained: False, num_classes: 10):
    model = models.segmentation.deeplabv3_mobilenet_v3_large(
        pretrained=pretrained,
        num_classes=num_classes,
    )
    old_conv = model.backbone['0'][0]
    new_conv = torch.nn.Conv2d(in_channels,
                               old_conv.out_channels,
                               old_conv.kernel_size,
                               old_conv.stride,
                               old_conv.padding,
                               old_conv.dilation,
                               old_conv.groups)
    model.backbone['0'][0] = new_conv
    return model

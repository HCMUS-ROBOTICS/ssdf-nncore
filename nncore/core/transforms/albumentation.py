from . import TRANSFORM_REGISTRY

from albumentations import (
    SmallestMaxSize,
    ShiftScaleRotate,
    RandomCrop,
    RGBShift,
    RandomBrightnessContrast,
    Normalize,
    Resize,
    Compose,
)

from albumentations.pytorch.transforms import ToTensorV2


TRANSFORM_REGISTRY.register(RandomCrop, prefix='Alb')
TRANSFORM_REGISTRY.register(RGBShift, prefix='Alb')
TRANSFORM_REGISTRY.register(Normalize, prefix='Alb')
TRANSFORM_REGISTRY.register(Resize, prefix='Alb')
TRANSFORM_REGISTRY.register(Compose, prefix='Alb')
TRANSFORM_REGISTRY.register(RandomBrightnessContrast, prefix='Alb')
TRANSFORM_REGISTRY.register(ShiftScaleRotate, prefix='Alb')
TRANSFORM_REGISTRY.register(SmallestMaxSize, prefix='Alb')
TRANSFORM_REGISTRY.register(ToTensorV2, prefix='Alb')

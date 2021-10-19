from nncore.core.registry import Registry

DATASET_REGISTRY = Registry('DATASET')

from .default_datasets import TestImageDataset

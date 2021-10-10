from nncore.utils.registry import Registry
DATASET_REGISTRY = Registry('DATASET')

from .default_datasets import *
from .lyft_dataset import *
from .ssdf_datasets import *

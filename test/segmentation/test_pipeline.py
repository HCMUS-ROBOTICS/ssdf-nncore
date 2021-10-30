import pytest
from seg_utils import _create_dummy_image_mask
from torch.utils.data.dataset import Dataset

from nncore.segmentation.datasets import DATASET_REGISTRY
from nncore.segmentation.pipeline import Pipeline


class DummySegDataset(Dataset):
    def __init__(self, batch_size: int = 100, num_classes: int = 10, transform=None) -> None:
        self.images, self.masks = _create_dummy_image_mask(batch_size, 3, 64, 64,
                                                           num_classes=num_classes)
        assert len(self.images) == len(self.masks)

    def __getitem__(self, index):
        image, mask = self.images[index], self.masks[index]
        return {
            'image': image,
            'mask': mask,
        }

    def __len__(self):
        return len(self.images)


@pytest.fixture
def segmentation_dataset_registry():
    DATASET_REGISTRY._do_register('dummy_seg_dataset', DummySegDataset)
    yield DATASET_REGISTRY
    DATASET_REGISTRY._obj_map.pop('dummy_seg_dataset')


def test_get_data_pipeline(segmentation_dataset_registry):
    cfg = {
        'dataset': {
            "train": {
                'name': 'dummy_seg_dataset',
                'args': {
                    'batch_size': 9,
                },
            },
            'val': {
                'name': 'dummy_seg_dataset',
                'args': {
                    'batch_size': 9,
                }
            },
        },
        'loader': {
            'train': {
                'name': 'DataLoader',
                'args': {
                    'batch_size': 2,
                    'shuffle': True,
                    'drop_last': True,
                }
            },
            'val': {
                'name': 'DataLoader',
                'args': {
                    'batch_size': 2,
                    'shuffle': False,
                    'drop_last': False,
                }
            }
        }
    }
    transform = None
    train_loader, val_loader, train_data, val_data = Pipeline.get_data(None, cfg, transform)
    assert len(train_data) == 9
    assert len(val_data) == 9
    assert len(train_loader) == 4
    assert len(val_loader) == 5


def test_get_data_pipeline_split(segmentation_dataset_registry):
    cfg = {
        'splits': {
            'train': 0.7,
            'val': 0.3,
        },
        'dataset': {
            'name': 'dummy_seg_dataset',
            'args': {
                'batch_size': 10,
            },
        },
        'loader': {
            'train': {
                'name': 'DataLoader',
                'args': {
                    'batch_size': 2,
                    'shuffle': True,
                    'drop_last': True,
                }
            },
            'val': {
                'name': 'DataLoader',
                'args': {
                    'batch_size': 2,
                    'shuffle': False,
                    'drop_last': False,
                }
            }
        }
    }
    transform = None
    train_loader, val_loader, train_data, val_data = Pipeline.get_data(None, cfg, transform)
    assert len(train_data) == 7
    assert len(val_data) == 3
    assert len(train_loader) == 3
    assert len(val_loader) == 2

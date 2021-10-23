import pytest
import torch
from seg_utils import _create_dummy_image_mask

from nncore.segmentation.criterion import CEwithstat


@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('num_classes', [10])
def test_cross_entropy_forward(batch_size, num_classes):
    image, mask = _create_dummy_image_mask(batch_size, channels=num_classes,
                                           num_classes=num_classes)
    criterion = CEwithstat()
    predicts = {'out': image}
    batch = {'mask': mask}
    loss, loss_dict = criterion.forward(predicts, batch)
    assert 'loss' in loss_dict.keys()
    assert loss_dict['loss'].shape == torch.Size([])
    assert loss == loss_dict['loss']

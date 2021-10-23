import pytest
import torch
from seg_utils import _create_dummy_image_mask, _get_device

from nncore.segmentation.models import MODEL_REGISTRY
from nncore.utils.getter import get_instance


@pytest.mark.parametrize('batch_size', [2])
@pytest.mark.parametrize('num_classes', [1, 14])
@pytest.mark.parametrize('device', _get_device())
def test_model_deeplabv3_resnet50(batch_size, num_classes, device):
    cfg = {
        'name': 'deeplabv3_resnet50',
        'args': {
            'pretrained': False,
            'num_classes': num_classes,
        }
    }

    model = get_instance(cfg, MODEL_REGISTRY).to(device)
    dummy_image, _ = _create_dummy_image_mask(batch_size, height=64, width=64,
                                              num_classes=num_classes, device=device)
    with torch.no_grad():
        out = model.forward(dummy_image)
    assert 'out' in out.keys()
    assert out['out'].shape == torch.Size([batch_size, num_classes, 64, 64])

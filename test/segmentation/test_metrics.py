import pytest
import torch
from seg_utils import _isclose

from nncore.segmentation.metrics import PixelAccuracy


@pytest.mark.parametrize('ignore_index', [None, 0])
def test_pixel_accuracy(ignore_index):

    target = torch.tensor([[1, 0, 3],
                           [2, 0, 1],
                           [1, 1, 0]], dtype=torch.long)
    target = target.unsqueeze_(0)                           # B, H, W

    pred = torch.tensor([[1, 2, 3],
                         [2, 3, 1],
                         [1, 2, 0]], dtype=torch.long)
    pred = torch.nn.functional.one_hot(pred)  # H, W, C
    pred = pred.permute(2, 0, 1).unsqueeze(0).contiguous()  # B, N_CLS, H, W

    metric = PixelAccuracy(ignore_index)
    output = {'out': pred}
    batch = {'mask': target}
    metric.update(output, batch)
    if ignore_index is None:
        assert _isclose(metric.value(), (6 / 9))
    else:
        assert _isclose(metric.value(), (5 / 6))

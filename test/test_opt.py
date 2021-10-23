import pytest
import torch
import yaml

from nncore.core.opt import Opts


@pytest.fixture
def minimal_cfg(tmp_path):
    cfg = {
        'opts': {
            'id': 'default',
            'gpus': 0,
            'save_dir': str(tmp_path / 'runs'),
        }
    }
    return cfg


def _save_cfg(tmp_path, cfg):
    with open(tmp_path, 'w+') as f:
        yaml.safe_dump(cfg, f)


@pytest.mark.parametrize('gpus_flag', [None, 'delete', 'not_available'])
def test_opts_device_cpu(tmp_path, minimal_cfg, gpus_flag):
    def _fake_test():
        bak_func = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        _normal_test()
        torch.cuda.is_available = bak_func

    def _normal_test():
        _save_cfg(cfg_path, minimal_cfg)
        opts = Opts.parse(cfg_path)
        assert opts.device == 'cpu'

    cfg_path = tmp_path / "config.yaml"
    if gpus_flag == 'not_available':
        _fake_test()  # opts is set to gpus but cuda is not available
    elif gpus_flag is None:
        minimal_cfg['opts']['gpus'] = None  # opts has the "gpus" key but don't set
    elif gpus_flag == 'delete':
        del minimal_cfg['opts']['gpus']  # opts doesnt have the "gpus" key


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda is not available')
@pytest.mark.parametrize(
    'device_idx',
    [0, 1, 2, 3, -1, -2]
)
def test_opts_device_index_int(tmp_path, minimal_cfg, device_idx):
    cfg_path = tmp_path / "config.yaml"
    minimal_cfg['opts']['gpus'] = device_idx
    _save_cfg(cfg_path, minimal_cfg)
    if device_idx >= -1:
        opts = Opts.parse(cfg_path)
        if device_idx == -1:
            assert opts.device == 'cuda'
        else:
            assert opts.device == f'cuda:{device_idx}'
    else:
        with pytest.raises(ValueError):
            opts = Opts.parse(cfg_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda is not available')
@pytest.mark.parametrize(
    'device_idx',
    ['0,1', '1,2']
)
def test_opts_device_index_str(tmp_path, minimal_cfg, device_idx):
    cfg_path = tmp_path / "config.yaml"
    minimal_cfg['opts']['gpus'] = device_idx
    _save_cfg(cfg_path, minimal_cfg)
    opts = Opts.parse(cfg_path)
    assert opts.device == f'cuda:{device_idx}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda is not available')
@pytest.mark.parametrize(
    'device_idx',
    [[1, 2], [0], [0, 1]]
)
def test_opts_device_index_list(tmp_path, minimal_cfg, device_idx):
    cfg_path = tmp_path / "config.yaml"
    minimal_cfg['opts']['gpus'] = device_idx
    _save_cfg(cfg_path, minimal_cfg)
    opts = Opts.parse(cfg_path)
    assert opts.device == f'cuda:{",".join(map(str, device_idx))}'

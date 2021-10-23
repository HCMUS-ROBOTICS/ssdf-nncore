import torch


def _create_dummy_image_mask(batch_size: int = 1,
                             channels: int = 3,
                             height: int = 64,
                             width: int = 64,
                             num_classes: int = 10,
                             device: torch.device = 'cpu',
                             ):
    dummy_image = torch.rand(batch_size, channels, height, width, device=device)
    dummy_mask = torch.randint(0, num_classes, (batch_size, height, width), device=device)
    return dummy_image, dummy_mask


def _get_device():
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    return devices

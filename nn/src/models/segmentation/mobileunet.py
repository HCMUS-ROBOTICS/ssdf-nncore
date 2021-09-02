import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import mobilenet_v2

__all__ = ["MobileUnet"]


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        # if self.use_res_connect:
        #     return x + self.conv(x)
        # else:
        #     return self.conv(x)
        return self.conv(x)


class MobileUnet(nn.Module):
    """Some Information about MobileUnet"""

    __constants__ = ["mobilenet"]

    def __init__(self):
        super(MobileUnet, self).__init__()

        mobilenet = mobilenet_v2(pretrained=True, progress=True)
        for param in mobilenet.parameters():
            param.requires_grad_(False)

        self.down1 = nn.Sequential(*mobilenet.features[0:2])
        self.down2 = nn.Sequential(*mobilenet.features[2:4])
        self.down3 = nn.Sequential(*mobilenet.features[4:7])
        self.down4 = nn.Sequential(*mobilenet.features[7:14])
        self.down5 = nn.Sequential(*mobilenet.features[14:19])

        # sizes = [320] + [160]*3 + [96]*3 + [64]*4 + [32]*3 + [24]*2 + [16]
        # self.up1 = torch.jit.script(Up(mobilenet.classifier[1].in_features, 96, 96))
        # self.up2 = torch.jit.script(Up(96, 32, 32))
        # self.up3 = torch.jit.script(Up(32, 24, 24))
        # self.up4 = torch.jit.script(Up(24, 16, 16))
        # self.up5 = torch.jit.script(Up(16, 3, 3))

        self.dconv1 = nn.ConvTranspose2d(1280, 96, 4, padding=1, stride=2)
        self.invres1 = InvertedResidual(192, 96, 1, 6)

        self.dconv2 = nn.ConvTranspose2d(96, 32, 4, padding=1, stride=2)
        self.invres2 = InvertedResidual(64, 32, 1, 6)

        self.dconv3 = nn.ConvTranspose2d(32, 24, 4, padding=1, stride=2)
        self.invres3 = InvertedResidual(48, 24, 1, 6)

        self.dconv4 = nn.ConvTranspose2d(24, 16, 4, padding=1, stride=2)
        self.invres4 = InvertedResidual(32, 16, 1, 6)

        self.dconv5 = nn.ConvTranspose2d(16, 3, 4, padding=1, stride=2)
        self.invres5 = InvertedResidual(6, 3, 1, 6)

        self.conv_last = nn.Conv2d(3, 3, 1)
        self.conv_score = nn.Conv2d(3, 2, 1)

        # init weights...

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, Up):
                module._init_weights()

    def forward(self, x):
        # print((x.shape, 'x'))
        x0 = x
        x1 = x = self.down1(x)
        # print((x1.shape, 'x1'))
        x2 = x = self.down2(x)
        # print((x2.shape, 'x2'))
        x3 = x = self.down3(x)
        # print((x3.shape, 'x3'))
        x4 = x = self.down4(x)
        # print((x4.shape, 'x4'))
        x5 = x = self.down5(x)
        # print((x5.shape, 'x5'))

        up1 = torch.cat([x4, self.dconv1(x)], dim=1)
        up1 = self.invres1(up1)
        # print((up1.shape, 'up1'))

        up2 = torch.cat([x3, self.dconv2(up1)], dim=1)
        up2 = self.invres2(up2)
        # print((up2.shape, 'up2'))

        up3 = torch.cat([x2, self.dconv3(up2)], dim=1)
        up3 = self.invres3(up3)
        # print((up3.shape, 'up3'))

        up4 = torch.cat([x1, self.dconv4(up3)], dim=1)
        up4 = self.invres4(up4)
        # print((up4.shape, 'up4'))

        up5 = torch.cat([x0, self.dconv5(up4)], dim=1)
        up5 = self.invres5(up5)

        x = self.conv_last(up5)
        # print((x.shape, 'conv_last'))

        x = self.conv_score(x)
        # print((x.shape, 'conv_score'))

        # x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear',
        #                               align_corners=False)
        # Use this instead when porting to TensorRT
        # https://github.com/onnx/onnx-tensorrt/issues/192#issuecomment-526182296
        # sh = torch.tensor(x.shape)
        # x = nn.functional.interpolate(
        #     x, size=(sh[2] * 2, sh[3] * 2), mode='bilinear', align_corners=False)
        # print((x.shape, 'interpolate'))

        # x = torch.sigmoid(x)

        return x


class Up(nn.Module):
    def __init__(self, in_channels, in_concat_channels, out_channels):
        super(Up, self).__init__()
        self.in_channels = in_channels
        self.in_concat_channels = in_concat_channels
        self.out_channels = out_channels

        self.up_conv = UpConv(in_channels)
        self.double_conv = DoubleConv(
            self.in_channels + self.in_concat_channels, self.out_channels
        )

    def forward(self, x1, x2):
        """
        :param x1: feature from previous layer
        :param x2: feature to concat
        """
        x1 = self.up_conv(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.double_conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.up_conv = nn.Sequential(
            OrderedDict(
                [
                    ("up", nn.Upsample(scale_factor=2)),
                    (
                        "conv",
                        nn.Conv2d(
                            in_channels, in_channels // 2, kernel_size=3, padding=1
                        ),
                    ),
                ]
            )
        )

    def forward(self, x):
        return self.up_conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


if __name__ == "__main__":
    rand_input = torch.rand((1, 3, 224, 224))
    model = MobileUnet()
    print(model)
    model.eval()
    with torch.no_grad():
        output = model(rand_input)
    print(output.size())

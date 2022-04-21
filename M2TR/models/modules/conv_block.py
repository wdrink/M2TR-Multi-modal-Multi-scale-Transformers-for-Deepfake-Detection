import torch.nn as nn
import torch.nn.functional as F


class Deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            input_channel,
            output_channel,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = F.interpolate(
            x, scale_factor=2, mode='bilinear', align_corners=True
        )
        out = self.conv(x)
        out = self.leaky_relu(out)
        return out


class ConvBN(nn.Module):
    def __init__(self, in_features, out_features):
        self.conv = nn.Conv2d(in_features, out_features, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out

'''
Copyright 2022 fvl

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
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

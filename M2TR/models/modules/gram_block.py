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


class GramMatrix(nn.Module):
    def __init__(self):
        super(GramMatrix, self).__init__()

    def forward(self, x):
        b, c, h, w = x.size()
        feature = x.view(b, c, h * w)
        feature_t = feature.transpose(1, 2)
        gram = feature.bmm(feature_t)
        b, h, w = gram.size()
        gram = gram.view(b, 1, h, w)
        return gram


class GramBlock(nn.Module):
    def __init__(self, in_channels):
        super(GramBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=1, padding=2
        )
        self.gramMatrix = GramMatrix()
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.gramMatrix(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        return x

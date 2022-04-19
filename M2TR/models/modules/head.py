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

from M2TR.models.modules.conv_block import Deconv


class Classifier2D(nn.Module):
    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(Classifier2D, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(dim_in, num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        if not self.training:
            x = self.act(x)
        return x


class Localizer(nn.Module):
    def __init__(self, in_channel, output_channel):
        super(self, Localizer).__init__()
        self.deconv1 = Deconv(in_channel, in_channel)
        hidden_dim = in_channel // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.deconv2 = Deconv(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                hidden_dim, output_channel, kernel_size=3, stride=1, padding=1
            ),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.deconv1(x)
        out = self.conv1(out)
        out = self.deconv2(out)
        out = self.conv2(out)
        return self.sigmoid(out)

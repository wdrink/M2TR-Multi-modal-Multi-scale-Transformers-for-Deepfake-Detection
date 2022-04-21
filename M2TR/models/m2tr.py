import math

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from M2TR.utils.registries import MODEL_REGISTRY

from .base import BaseNetwork
from .xception import Xception
from .efficientnet import EfficientNet
from .modules.head import Classifier2D, Localizer
from .modules.transformer_block import FeedForward2D



class GlobalFilter(nn.Module):
    def __init__(self, dim=32, h=80, w=41, fp32fft=True):
        super().__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02
        )
        self.w = w
        self.h = h
        self.fp32fft = fp32fft

    def forward(self, x):
        b, _, a, b = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()

        if self.fp32fft:
            dtype = x.dtype
            x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm="ortho")

        if self.fp32fft:
            x = x.to(dtype)

        x = x.permute(0, 3, 1, 2).contiguous()

        return x


class FreqBlock(nn.Module):
    def __init__(self, dim, h=80, w=41, fp32fft=True):
        super().__init__()
        self.filter = GlobalFilter(dim, h=h, w=w, fp32fft=fp32fft)
        self.feed_forward = FeedForward2D(in_channel=dim, out_channel=dim)

    def forward(self, x):
        x = x + self.feed_forward(self.filter(x))
        return x


def attention(query, key, value):
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
        query.size(-1)
    )
    p_attn = F.softmax(scores, dim=-1)
    p_val = torch.matmul(p_attn, value)
    return p_val, p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, patchsize, d_model):
        super().__init__()
        self.patchsize = patchsize
        self.query_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0
        )
        self.value_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0
        )
        self.key_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0
        )
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        d_k = c // len(self.patchsize)
        output = []
        _query = self.query_embedding(x)
        _key = self.key_embedding(x)
        _value = self.value_embedding(x)
        attentions = []
        for (width, height), query, key, value in zip(
            self.patchsize,
            torch.chunk(_query, len(self.patchsize), dim=1),
            torch.chunk(_key, len(self.patchsize), dim=1),
            torch.chunk(_value, len(self.patchsize), dim=1),
        ):
            out_w, out_h = w // width, h // height

            # 1) embedding and reshape
            query = query.view(b, d_k, out_h, height, out_w, width)
            query = (
                query.permute(0, 2, 4, 1, 3, 5)
                .contiguous()
                .view(b, out_h * out_w, d_k * height * width)
            )
            key = key.view(b, d_k, out_h, height, out_w, width)
            key = (
                key.permute(0, 2, 4, 1, 3, 5)
                .contiguous()
                .view(b, out_h * out_w, d_k * height * width)
            )
            value = value.view(b, d_k, out_h, height, out_w, width)
            value = (
                value.permute(0, 2, 4, 1, 3, 5)
                .contiguous()
                .view(b, out_h * out_w, d_k * height * width)
            )

            y, _ = attention(query, key, value)

            # 3) "Concat" using a view and apply a final linear.
            y = y.view(b, out_h, out_w, d_k, height, width)
            y = y.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, d_k, h, w)
            attentions.append(y)
            output.append(y)

        output = torch.cat(output, 1)
        self_attention = self.output_linear(output)

        return self_attention


class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, patchsize, in_channel=256):
        super().__init__()
        self.attention = MultiHeadedAttention(patchsize, d_model=in_channel)
        self.feed_forward = FeedForward2D(
            in_channel=in_channel, out_channel=in_channel
        )

    def forward(self, rgb):
        self_attention = self.attention(rgb)
        output = rgb + self_attention
        output = output + self.feed_forward(output)
        return output


class CMA_Block(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super(CMA_Block, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channel, hidden_channel, kernel_size=1, stride=1, padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channel, hidden_channel, kernel_size=1, stride=1, padding=0
        )
        self.conv3 = nn.Conv2d(
            in_channel, hidden_channel, kernel_size=1, stride=1, padding=0
        )

        self.scale = hidden_channel ** -0.5

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                hidden_channel, out_channel, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, rgb, freq):
        _, _, h, w = rgb.size()

        q = self.conv1(rgb)
        k = self.conv2(freq)
        v = self.conv3(freq)

        q = q.view(q.size(0), q.size(1), q.size(2) * q.size(3)).transpose(
            -2, -1
        )
        k = k.view(k.size(0), k.size(1), k.size(2) * k.size(3))

        attn = torch.matmul(q, k) * self.scale
        m = attn.softmax(dim=-1)

        v = v.view(v.size(0), v.size(1), v.size(2) * v.size(3)).transpose(
            -2, -1
        )
        z = torch.matmul(m, v)
        z = z.view(z.size(0), h, w, -1)
        z = z.permute(0, 3, 1, 2).contiguous()

        output = rgb + self.conv4(z)

        return output


class PatchTrans(BaseNetwork):
    def __init__(self, in_channel, in_size):
        super(PatchTrans, self).__init__()
        self.in_size = in_size

        patchsize = [
            (in_size, in_size),
            (in_size // 2, in_size // 2),
            (in_size // 4, in_size // 4),
            (in_size // 8, in_size // 8),
        ]

        self.t = TransformerBlock(patchsize, in_channel=in_channel)

    def forward(self, enc_feat):
        output = self.t(enc_feat)
        return output


@MODEL_REGISTRY.register()
class M2TR(BaseNetwork):
    def __init__(self, model_cfg):
        super(M2TR, self).__init__()
        img_size = model_cfg["IMG_SIZE"]
        backbone = model_cfg["BACKBONE"]
        texture_layer = model_cfg["TEXTURE_LAYER"]
        feature_layer = model_cfg["FEATURE_LAYER"]
        depth = model_cfg["DEPTH"]
        num_classes = model_cfg["NUM_CLASSES"]
        drop_ratio = model_cfg["DROP_RATIO"]
        has_decoder = model_cfg["HAS_DECODER"]

        freq_h = img_size // 4
        freq_w = freq_h // 2 + 1

        if "xception" in backbone:
            self.model = Xception(num_classes)
        elif backbone.split("-")[0] == "efficientnet":
            self.model = EfficientNet({'NAME': backbone, 'PRETRAINED': True})

        self.texture_layer = texture_layer
        self.feature_layer = feature_layer

        with torch.no_grad():
            input = {"img": torch.zeros(1, 3, img_size, img_size)}
            layers = self.model(input)
        texture_dim = layers[self.texture_layer].shape[1]
        feature_dim = layers[self.feature_layer].shape[1]

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PatchTrans(in_channel=texture_dim, in_size=freq_h),
                        FreqBlock(dim=texture_dim, h=freq_h, w=freq_w),
                        CMA_Block(
                            in_channel=texture_dim,
                            hidden_channel=texture_dim,
                            out_channel=texture_dim,
                        ),
                    ]
                )
            )

        self.classifier = Classifier2D(
            feature_dim, num_classes, drop_ratio, "sigmoid"
        )

        self.has_decoder = has_decoder
        if self.has_decoder:
            self.decoder = Localizer(texture_dim, 1)

    def forward(self, x):
        rgb = x["img"]
        B = rgb.size(0)

        layers = {}
        rgb = self.model.extract_textures(rgb, layers)

        for attn, filter, cma in self.layers:
            rgb = attn(rgb)
            freq = filter(rgb)
            rgb = cma(rgb, freq)

        features = self.model.extract_features(rgb, layers)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(B, features.size(1))

        logits = self.classifier(features)

        if self.has_decoder:
            mask = self.decoder(rgb)
            mask = mask.squeeze(-1)

        else:
            mask = None

        output = {"logits": logits, "mask": mask, "features:": features}
        return output


if __name__ == "__main__":
    from torchsummary import summary

    model = M2TR(num_classes=1, has_decoder=False)
    model.cuda()
    summary(model, input_size=(3, 320, 320), batch_size=12, device="cuda")

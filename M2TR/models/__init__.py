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
from .base import BaseNetwork
from .efficientnet import EfficientNet
from .f3net import F3Net
from .gramnet import GramNet
from .m2tr import M2TR
from .matdd import MAT
from .meso4 import Meso4
from .meso_inception4 import MesoInception4
from .modules.transformer_block import FeedForward2D
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .vit import VisionTransformer
from .xception import Xception

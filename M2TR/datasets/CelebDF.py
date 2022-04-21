import os

import torch

from M2TR.datasets.dataset import DeepFakeDataset
from M2TR.utils.registries import DATASET_REGISTRY

from .utils import get_image_from_path

'''
DATASET:
  DATASET_NAME: CelebDF
  ROOT_DIR: /some_where/celeb-df-v2
  TRAIN_INFO_TXT: '/some_where/celeb-df-v2/splits/train.txt'
  VAL_INFO_TXT: '/some_where/celeb-df-v2/splits/eval.txt'
  TEST_INFO_TXT: '/some_where/celeb-df-v2/splits/eval.txt'
  IMG_SIZE: 380
  SCALE_RATE: 1.0
'''


@DATASET_REGISTRY.register()
class CelebDF(DeepFakeDataset):
    def __getitem__(self, idx):
        info_line = self.info_list[idx]
        image_info = info_line.strip('\n').split()
        image_path = image_info[0]
        image_abs_path = os.path.join(self.root_dir, image_path)

        img, _ = get_image_from_path(
            image_abs_path, None, self.mode, self.dataset_cfg
        )
        img_label_binary = int(image_info[1])

        sample = {
            'img': img,
            'bin_label': [int(img_label_binary)],
        }

        sample['img'] = torch.FloatTensor(sample['img'])
        sample['bin_label'] = torch.FloatTensor(sample['bin_label'])
        sample['bin_label_onehot'] = self.label_to_one_hot(
            sample['bin_label'], 2
        ).squeeze()
        return sample

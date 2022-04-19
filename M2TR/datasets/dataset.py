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
import os

import torch
from torch.utils.data import Dataset


class DeepFakeDataset(Dataset):
    def __init__(
        self,
        dataset_cfg,
        mode='train',
    ):
        dataset_name = dataset_cfg['DATASET_NAME']
        assert dataset_name in [
            'ForgeryNet',
            'FFDF',
            'CelebDF',
        ], 'no dataset'
        assert mode in [
            'train',
            'val',
            'test',
        ], 'wrong mode'
        self.dataset_name = dataset_name
        self.mode = mode
        self.dataset_cfg = dataset_cfg
        self.root_dir = dataset_cfg['ROOT_DIR']
        info_txt_tag = mode.upper() + '_INFO_TXT'
        if dataset_cfg[info_txt_tag] != '':
            self.info_txt = dataset_cfg[info_txt_tag]
        else:
            self.info_txt = os.path.join(
                self.root_dir,
                self.dataset_name + '_splits_' + mode + '.txt',
            )
        self.info_list = open(self.info_txt).readlines()

    def __len__(self):
        return len(self.info_list)

    def label_to_one_hot(self, x, class_count):
        return torch.eye(class_count)[x.long(), :]

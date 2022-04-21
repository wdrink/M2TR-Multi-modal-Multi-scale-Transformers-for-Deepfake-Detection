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

import os
import random

import albumentations
import albumentations.pytorch
import numpy as np
import torchvision
from PIL import Image


class ResizeRandomCrop:
    def __init__(self, img_size=320, scale_rate=8 / 7, p=0.5):
        self.img_size = img_size
        self.scale_rate = scale_rate
        self.p = p

    def __call__(self, image, mask=None):
        if random.uniform(0, 1) < self.p:
            S1 = int(self.img_size * self.scale_rate)
            S2 = S1
            resize_func = torchvision.transforms.Resize((S1, S2))
            image = resize_func(image)
            crop_params = torchvision.transforms.RandomCrop.get_params(
                image, (self.img_size, self.img_size)
            )
            image = torchvision.transforms.functional.crop(image, *crop_params)
            if mask is not None:
                mask = resize_func(mask)
                mask = torchvision.transforms.functional.crop(
                    mask, *crop_params
                )

        else:
            resize_func = torchvision.transforms.Resize(
                (self.img_size, self.img_size)
            )
            image = resize_func(image)
            if mask is not None:
                mask = resize_func(mask)

        return image, mask


def transforms_mask(mask_size):
    return albumentations.Compose(
        [
            albumentations.Resize(mask_size, mask_size),
            albumentations.pytorch.transforms.ToTensorV2(),
        ]
    )


def get_augmentations_from_list(augs: list, aug_cfg, one_of_p=1):
    ops = []
    for aug in augs:
        if isinstance(aug, list):
            op = albumentations.OneOf
            param = get_augmentations_from_list(aug, aug_cfg)
            param = [param, one_of_p]
        else:
            op = getattr(albumentations, aug)
            param = (
                aug_cfg[aug.upper() + '_PARAMS']
                if aug.upper() + '_PARAMS' in aug_cfg
                else []
            )
        ops.append(op(*tuple(param)))
    return ops


def get_transformations(
    mode,
    dataset_cfg,
):
    if mode == 'train':
        aug_cfg = dataset_cfg['TRAIN_AUGMENTATIONS']
    else:
        aug_cfg = dataset_cfg['TEST_AUGMENTATIONS']
    ops = get_augmentations_from_list(aug_cfg['COMPOSE'], aug_cfg)
    ops.append(albumentations.pytorch.transforms.ToTensorV2())
    augmentations = albumentations.Compose(ops, p=1)
    return augmentations


def get_image_from_path(img_path, mask_path, mode, dataset_cfg):
    img_size = dataset_cfg['IMG_SIZE']
    scale_rate = dataset_cfg['SCALE_RATE']

    img = Image.open(img_path)
    if mask_path is not None and os.path.exists(mask_path):
        mask = Image.open(mask_path).convert('L')
    else:
        mask = Image.fromarray(np.zeros((img_size, img_size)))

    trans_list = get_transformations(
        mode,
        dataset_cfg,
    )
    if mode == 'train':
        crop = ResizeRandomCrop(img_size=img_size, scale_rate=scale_rate, p=-1)
        img, mask = crop(image=img, mask=mask)

        img = np.asarray(img)
        img = trans_list(image=img)['image']

        mask = np.asarray(mask)
        mask = transforms_mask(img_size)(image=mask)['image']

    else:
        img = np.asarray(img)
        img = trans_list(image=img)['image']
        mask = np.asarray(mask)
        mask = transforms_mask(img_size)(image=mask)['image']

    return img, mask.float()


def get_mask_path_from_img_path(dataset_name, root_dir, img_info):
    if dataset_name == 'ForgeryNet':
        root_dir = os.path.join(root_dir, 'spatial_localize')
        fore_path = img_info.split('/')[0]
        if 'train' in fore_path:
            img_info = img_info.replace('train_release', 'train_mask_release')
        else:
            img_info = img_info[20:]

        mask_complete_path = os.path.join(root_dir, img_info)

    elif 'FFDF' in dataset_name:
        mask_info = img_info.replace('images', 'masks')
        mask_complete_path = os.path.join(root_dir, mask_info)

    return mask_complete_path

import argparse

import torch
import yaml

import M2TR.utils.logging as logging
from M2TR.utils.checkpoint import get_path_to_checkpoint

logger = logging.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Provide training and testing pipeline."
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--shard_id",
        dest="shard_id",
        help="The shard id of current node, Starts from 0 to NUM_SHARDS - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--lr",
        dest="base_lr",
        help="The base learning rate",
        type=float,
    )
    return parser.parse_args()


def merge_a_into_b(a, b):
    for k, v in a.items():
        if isinstance(v, dict) and k in b:
            assert isinstance(
                b[k], dict
            ), "Cannot inherit key '{}' from base!".format(k)
            merge_a_into_b(v, b[k])
        else:
            b[k] = v


def load_config(args):
    with open('./configs/default.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    logger.info('Use cfg_file: ', './configs/' + args.cfg_file)
    with open('./configs/' + args.cfg_file, 'r') as file:  # TODO use PATH.join?
        custom_cfg = yaml.safe_load(file)
    merge_a_into_b(custom_cfg, cfg)
    if args.shard_id is not None:
        cfg['SHARD_ID'] = args.shard_id
    if args.base_lr is not None:
        cfg['OPTIMIZER']['BASE_LR'] = args.base_lr

    if cfg['TRAIN']['ENABLE']:
        cfg['TEST']['CHECKPOINT_TEST_PATH'] = get_path_to_checkpoint(
            cfg['TRAIN']['CHECKPOINT_SAVE_PATH'], cfg['TRAIN']['MAX_EPOCH'], cfg
        )
    cfg['DATASET']['TRAIN_AUGMENTATIONS']['RESIZE_PARAMS'] = [
        cfg['DATASET']['IMG_SIZE'],
        cfg['DATASET']['IMG_SIZE'],
    ]
    cfg['DATASET']['TEST_AUGMENTATIONS']['RESIZE_PARAMS'] = [
        cfg['DATASET']['IMG_SIZE'],
        cfg['DATASET']['IMG_SIZE'],
    ]

    logger.info(cfg)
    return cfg


def launch_func(cfg, func, daemon=False):
    if cfg['NUM_GPUS'] > 1:
        torch.multiprocessing.spawn(
            func,
            nprocs=cfg['NUM_GPUS'],
            args=(
                cfg['NUM_GPUS'],
                cfg['INIT_METHOD'],
                cfg['SHARD_ID'],
                cfg['NUM_SHARDS'],
                cfg['DIST_BACKEND'],
                cfg,
            ),
            daemon=daemon,
        )
    else:
        func(
            local_rank=0,
            num_proc=1,
            init_method=cfg['INIT_METHOD'],
            shard_id=0,
            num_shards=1,
            backend=cfg['DIST_BACKEND'],
            cfg=cfg,
        )

from tools.test import test
from tools.train import train
from tools.utils import launch_func, load_config, parse_args


def main():
    args = parse_args()
    cfg = load_config(args)
    if cfg['TRAIN']['ENABLE']:
        launch_func(cfg=cfg, func=train)
    if cfg['TEST']['ENABLE']:
        launch_func(cfg=cfg, func=test)


if __name__ == '__main__':
    main()

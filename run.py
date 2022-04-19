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

import os
import random
from warnings import warn

import numpy as np
import torch
import yaml
from texttable import Texttable


def print_args(args):
    _dict = vars(args)
    _key = sorted(_dict.items(), key=lambda x: x[0])
    t = Texttable()
    t.add_row(["Parameter", "Value"])
    for k, _ in _key:
        t.add_row([k, _dict[k]])
    print(t.draw())


def set_seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_num)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)


def AcontainsB(A, listB):
    # A: string; listB: list of strings
    for s in listB:
        if s in A: return True
    return False


def yaml_parser(model):
    filename = os.path.join('options/configs', f'{model}.yml')
    if os.path.exists(filename):
        with open(filename, 'r') as yaml_f:
            configs = yaml.load(yaml_f, Loader=yaml.FullLoader)
        return configs
    else:
        warn(f'configs of {model} not found, use the default setting instead')
        return {}


def overwrite_with_yaml(args, model, dataset):
    configs = yaml_parser(model)
    if dataset not in configs.keys():
        warn(f'{model} have no specific settings on {dataset}. Use the default setting instead.')
        return args
    for k, v in configs[dataset].items():
        if k in args.__dict__:
            args.__dict__[k] = v
        else:
            warn(f"Ignored unknown parameter {k} in yaml.")
    return args

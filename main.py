import gc

import numpy as np
import torch

from options.base_options import BaseOptions
from trainer import trainer
from utils import set_seed, print_args, overwrite_with_yaml


def main(args):
    list_test_acc = []
    list_valid_acc = []
    list_train_loss = []
    if args.compare_model:
        args = overwrite_with_yaml(args, args.type_model, args.dataset)
    print_args(args)
    for seed in range(args.N_exp):
        print(f'seed (which_run) = <{seed}>')
        args.random_seed = seed
        set_seed(args)
        torch.cuda.empty_cache()
        trnr = trainer(args, seed)
        train_loss, valid_acc, test_acc = trnr.train_and_test()
        list_test_acc.append(test_acc)
        list_valid_acc.append(valid_acc)
        list_train_loss.append(train_loss)

        del trnr
        torch.cuda.empty_cache()
        gc.collect()

        # record training data
        print('mean and std of test acc: {:.4f}±{:.4f}'.format(
            np.mean(list_test_acc), np.std(list_test_acc)))

    print('final mean and std of test acc with <{}> runs: {:.4f}±{:.4f}'.format(
        args.N_exp, np.mean(list_test_acc), np.std(list_test_acc)))


if __name__ == "__main__":
    args = BaseOptions().initialize()
    main(args)

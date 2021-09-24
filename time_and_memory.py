import importlib
import gc
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch.cuda import reset_max_memory_allocated, max_memory_allocated
from ogb.nodeproppred import Evaluator

from Dataloader import load_data, load_ogbn
from tricks import TricksComb
from utils import AcontainsB
from options.base_options import BaseOptions
from utils import set_seed, print_args, overwrite_with_yaml


class trainer(object):
    def __init__(self, args, which_run):
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.args = args
        self.device = torch.device(f'cuda:{args.cuda_num}' if args.cuda else 'cpu')
        self.which_run = which_run
        # set dataloader
        if self.dataset == 'ogbn-arxiv':
            self.data, self.split_idx = load_ogbn(self.dataset)
            self.data.to(self.device)
            self.train_idx = self.split_idx['train'].to(self.device)
            self.evaluator = Evaluator(name='ogbn-arxiv')
            self.loss_fn = torch.nn.functional.nll_loss
        else:
            self.data = load_data(self.dataset, self.which_run)
            self.loss_fn = torch.nn.functional.nll_loss
            self.data.to(self.device)

        if args.compare_model:  # only compare model
            Model = getattr(importlib.import_module("models"), self.type_model)
            self.model = Model(args)
        else:  # compare tricks combinations
            self.model = TricksComb(args)

        self.model.to(self.device)
        self.optimizer = self.model.optimizer

    def train(self):
        time_per_epoch = 0.
        preburn = 10
        reset_max_memory_allocated(self.device)

        for epoch in range(self.epochs):
            start_time = time.time()
            acc_train = self.run_trainSet()
            if epoch >= preburn:
                time_per_epoch += time.time() - start_time
        
        time_per_epoch /= self.epochs - preburn
        # number of model parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        max_memory = max_memory_allocated(self.device)

        return time_per_epoch, num_params, max_memory
    

    def run_trainSet(self):
        self.model.train()
        loss = 0.
        if self.dataset == 'ogbn-arxiv':
            pred = self.model(self.data.x, self.data.edge_index)
            pred = F.log_softmax(pred[self.train_idx], 1)
            loss = self.loss_fn(pred, self.data.y.squeeze(1)[self.train_idx])
        else:
            raw_logits = self.model(self.data.x, self.data.edge_index)
            logits = F.log_softmax(raw_logits[self.data.train_mask], 1)
            loss = self.loss_fn(logits, self.data.y[self.data.train_mask])
            # label smoothing loss
            if AcontainsB(self.type_trick, ['LabelSmoothing']):
                smooth_loss = -raw_logits[self.data.train_mask].mean(dim=-1).mean()
                loss = 0.97 * loss + 0.03 * smooth_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


def main(args):
    with open(args.log_file_name, 'a') as log_file:
        args.N_exp = 1
        args.epochs = 20
        time_per_epoch = 0.
        num_params = 0
        max_memery = 0.
        if args.compare_model:
            args = overwrite_with_yaml(args, args.type_model, args.dataset)
        for seed in range(args.N_exp):
            args.random_seed = seed
            set_seed(args)
            torch.cuda.empty_cache()
            trnr = trainer(args, seed)
            time_per_epoch, num_params, max_memory = trnr.train()

            del trnr
            torch.cuda.empty_cache()
            gc.collect()
        print('---> {} | {}-{} | {}'.format(args.dataset, args.num_layers, args.type_model, args.type_trick), file=log_file)
        print('time_per_epoch: {:.2f}ms, num_params: {:.4f}M, max_memory: {:.2f}MB'.format(
            time_per_epoch*1000, num_params/1e6, max_memory/(1024*1024)), file=log_file)


if __name__ == "__main__":
    args = BaseOptions().initialize()
    main(args)

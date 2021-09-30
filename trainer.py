import importlib

import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator

from Dataloader import load_data, load_ogbn
from tricks import TricksComb, TricksCombSGC
from utils import AcontainsB


def evaluate(output, labels, mask):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()


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

    def train_and_test(self):
        best_val_acc = 0.
        best_train_loss = 100
        best_test_acc = 0.
        best_train_acc = 0.
        best_val_loss = 100.
        patience = self.args.patience
        bad_counter = 0.
        val_loss_history = []

        for epoch in range(self.epochs):

            acc_train, acc_val, acc_test, loss_train, loss_val = self.train_net()

            val_loss_history.append(loss_val)

            if self.dataset != 'ogbn-arxiv':
                if loss_val < best_val_loss:
                    best_val_loss = loss_val
                    best_test_acc = acc_test
                    best_val_acc = acc_val
                    best_train_loss = loss_train
                    best_train_acc = acc_train
                    bad_counter = 0
                else:
                    bad_counter += 1
            else:
                if acc_val > best_val_acc:
                    best_val_loss = loss_val
                    best_test_acc = acc_test
                    best_val_acc = acc_val
                    best_train_loss = loss_train
                    best_train_acc = acc_train
                    bad_counter = 0
                else:
                    bad_counter += 1

            # if epoch % 20 == 0:
            if epoch % 20 == 0 or epoch == 1:
                log = 'Epoch: {:03d}, Train loss: {:.4f}, Val loss: {:.4f}, Test acc: {:.4f}'
                print(log.format(epoch, loss_train, loss_val, acc_test))
            if bad_counter == patience:
                # self.save_records(is_last=True)
                break

        print('train_loss: {:.4f}, val_acc: {:.4f}, test_acc:{:.4f}'
              .format(best_train_loss, best_val_acc, best_test_acc))
        return best_train_loss, best_val_acc, best_test_acc

    def train_net(self):
        try:
            loss_train = self.run_trainSet()
            acc_train, acc_val, acc_test, loss_val = self.run_testSet()
            return acc_train, acc_val, acc_test, loss_train, loss_val
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
            else:
                raise e

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

    @torch.no_grad()
    def run_testSet(self):
        self.model.eval()
        # torch.cuda.empty_cache()
        if self.dataset == 'ogbn-arxiv':
            out = self.model(self.data.x, self.data.edge_index)
            out = F.log_softmax(out, 1)
            y_pred = out.argmax(dim=-1, keepdim=True)

            train_acc = self.evaluator.eval({
                'y_true': self.data.y[self.split_idx['train']],
                'y_pred': y_pred[self.split_idx['train']],
            })['acc']
            valid_acc = self.evaluator.eval({
                'y_true': self.data.y[self.split_idx['valid']],
                'y_pred': y_pred[self.split_idx['valid']],
            })['acc']
            test_acc = self.evaluator.eval({
                'y_true': self.data.y[self.split_idx['test']],
                'y_pred': y_pred[self.split_idx['test']],
            })['acc']

            return train_acc, valid_acc, test_acc, 0.

        else:
            logits = self.model(self.data.x, self.data.edge_index)
            logits = F.log_softmax(logits, 1)
            acc_train = evaluate(logits, self.data.y, self.data.train_mask)
            acc_val = evaluate(logits, self.data.y, self.data.val_mask)
            acc_test = evaluate(logits, self.data.y, self.data.test_mask)
            val_loss = self.loss_fn(logits[self.data.val_mask], self.data.y[self.data.val_mask])
            return acc_train, acc_val, acc_test, val_loss

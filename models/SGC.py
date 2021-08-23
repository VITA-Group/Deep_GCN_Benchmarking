import torch
from torch import nn

# from torch_geometric.nn import SGConv
from models.SGC_layer import SGConv


class SGC(nn.Module):
    def __init__(self, args):
        super(SGC, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.cached = self.transductive = args.transductive
        # bn used in dataset 'obgn-arxiv'
        # lin_first used in Coauthor_physics due to space limit
        self.bn = True if args.type_norm == 'batch' else False
        self.lin_first = True if args.dataset == 'Coauthor_Physics' else False
        self.SGC = SGConv(self.num_feats, self.num_classes, K=self.num_layers,
                          cached=self.cached, bias=False, bn=self.bn, dropout=self.dropout,
                          lin_first=self.lin_first)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.lr, weight_decay=self.weight_decay)


    def forward(self, x, edge_index):
        # implemented based on https://github.com/Tiiiger/SGC/blob/master/citation.py
        x = self.SGC(x, edge_index)
        return x









import math

import torch
import torch.nn.functional as F
from torch import nn

from tricks.tricks.dropouts import DropoutTrick
from tricks.tricks.norms import run_norm_if_any, appendNormLayer
from tricks.tricks.others import GCNIdMapConv as GCNConv
from tricks.tricks.skipConnections import InitialConnection, DenseConnection, ResidualConnection
from utils import AcontainsB


class TricksComb(nn.Module):
    def __init__(self, args):
        super(TricksComb, self).__init__()
        self.args = args
        for k, v in vars(args).items():
            setattr(self, k, v)
        # cannot cache graph structure when use graph dropout tricks
        self.cached = self.transductive = args.transductive
        if AcontainsB(self.type_trick, ['DropEdge', 'DropNode', 'FastGCN', 'LADIES']):
            self.cached = False
        # set self.has_residual_MLP as True when has residual connection
        # to keep same hidden dimension
        self.has_residual_MLP = False
        if AcontainsB(self.type_trick, ['Jumping', 'Initial', 'Residual', 'Dense']):
            self.has_residual_MLP = True
        elif self.type_model=='SGC':
            self.has_residual_MLP = True
        # graph network initialize
        self.layers_GCN = nn.ModuleList([])
        self.layers_res = nn.ModuleList([])
        self.layers_norm = nn.ModuleList([])
        self.layers_MLP = nn.ModuleList([])
        # set MLP layer
        self.layers_MLP.append(nn.Linear(self.num_feats, self.dim_hidden))
        if not self.has_residual_MLP:
            self.layers_GCN.append(GCNConv(self.num_feats, self.dim_hidden, cached=self.cached))
        appendNormLayer(self, args, self.dim_hidden)

        for i in range(self.num_layers):
            if (not self.has_residual_MLP) and (
                    0 < i < self.num_layers - 1):  # if don't want 0_th MLP, then 0-th layer is assigned outside the for loop
                self.layers_GCN.append(GCNConv(self.dim_hidden, self.dim_hidden, cached=self.cached))
                appendNormLayer(self, args, self.dim_hidden)
            elif self.has_residual_MLP:
                self.layers_GCN.append(GCNConv(self.dim_hidden, self.dim_hidden, cached=self.cached))
                appendNormLayer(self, args, self.dim_hidden)

            # set residual connection type
            if AcontainsB(self.type_trick, ['Residual']):
                self.layers_res.append(ResidualConnection(alpha=self.alpha))
            elif AcontainsB(self.type_trick, ['Initial']):
                self.layers_res.append(InitialConnection(alpha=self.alpha))
            elif AcontainsB(self.type_trick, ['Dense']):
                if self.layer_agg in ['concat', 'maxpool']:
                    self.layers_res.append(
                        DenseConnection((i + 2) * self.dim_hidden, self.dim_hidden, self.layer_agg))
                elif self.layer_agg == 'attention':
                    self.layers_res.append(
                        DenseConnection(self.dim_hidden, self.dim_hidden, self.layer_agg))

        if not self.has_residual_MLP:
            self.layers_GCN.append(GCNConv(self.dim_hidden, self.num_classes, cached=self.cached))
            appendNormLayer(self, args, self.num_classes)

        self.graph_dropout = DropoutTrick(args)
        if AcontainsB(self.type_trick, ['NoDropout']):
            self.dropout = 0.0
            self.embedding_dropout = 0.0

        if AcontainsB(self.type_trick, ['Jumping']):
            if self.layer_agg in ['concat', 'maxpool']:
                self.layers_res.append(
                    DenseConnection((self.num_layers + 1) * self.dim_hidden, self.num_classes, self.layer_agg))
            elif self.layer_agg == 'attention':
                self.layers_res.append(
                    DenseConnection(self.dim_hidden, self.num_classes, self.layer_agg))
        else:
            self.layers_MLP.append(nn.Linear(self.dim_hidden, self.num_classes))

        # set optimizer
        self.reg_params = list(self.layers_GCN.parameters())
        self.non_reg_params = list(self.layers_MLP.parameters())
        self.optimizer = torch.optim.Adam([
            dict(params=self.reg_params, weight_decay=self.weight_decay1),
            dict(params=self.non_reg_params, weight_decay=self.weight_decay2)
        ], lr=self.lr)

        # set lambda
        if AcontainsB(self.type_trick, ['IdentityMapping']):
            self.lamda = args.lamda
        elif self.type_model == 'SGC':
            self.lamda = 0.
        elif self.type_model == 'GCN':
            self.lamda = 1.

    def forward(self, x, edge_index):
        x_list = []
        new_adjs = self.graph_dropout(edge_index)
        if self.has_residual_MLP:
            x = F.dropout(x, p=self.embedding_dropout, training=self.training)
            x = self.layers_MLP[0](x)
            x = F.relu(x)
            x_list.append(x)

        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            edge_index, _ = new_adjs[i]
            beta = math.log(self.lamda / (i + 1) + 1) if AcontainsB(self.type_trick,
                                                                    ['IdentityMapping']) else self.lamda
            x = self.layers_GCN[i](x, edge_index, beta)
            x = run_norm_if_any(self, x, i)
            if self.has_residual_MLP or i < self.num_layers - 1:
                x = F.relu(x)
            x_list.append(x)
            if AcontainsB(self.type_trick, ['Initial', 'Dense', 'Residual']):
                x = self.layers_res[i](x_list)

        x = F.dropout(x, p=self.embedding_dropout, training=self.training)
        if self.has_residual_MLP:
            if AcontainsB(self.type_trick, ['Jumping']):
                x = self.layers_res[0](x_list)
            else:
                x = self.layers_MLP[-1](x)

        return x

import math

import torch
import torch.nn.functional as F
from torch import nn

from tricks.tricks.dropouts import DropoutTrick
from tricks.tricks.norms import run_norm_if_any, appendNormLayer
from tricks.tricks.others import GCNIdMapConv as GCNConv
from tricks.tricks.skipConnections import InitialConnection, DenseConnection, ResidualConnection
from utils import AcontainsB

import math

import torch
import torch.nn.functional as F
from torch import nn

from tricks.tricks.dropouts import DropoutTrick
from tricks.tricks.norms import run_norm_if_any, appendNormLayer
from torch_geometric.nn import SGConv
from tricks.tricks.skipConnections import InitialConnection, DenseConnection, ResidualConnection
from utils import AcontainsB

class TricksComb(nn.Module):
    def __init__(self, args):
        super(TricksComb, self).__init__()
        
        # self.has_residual_MLP = args.has_residual_MLP
        
        # self.args = args
        # self.dataset = args.dataset
        # self.num_layers = args.num_layers
        # self.num_feats = args.num_feats
        # self.num_classes = args.num_classes
        # self.dim_hidden = args.dim_hidden
        # self.dropout = args.dropout
        # self.alpha = args.res_alpha
        # self.cached = self.transductive = args.transductive
        # self.type_res_trick = args.type_trick
        # self.aggregation = args.layer_agg

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

        # graph network initialize
        self.layers_res = nn.ModuleList([])
        self.layers_norm = nn.ModuleList([])
        self.layers_MLP = nn.ModuleList([])
        if self.has_residual_MLP:
            self.layer_SGC = SGConv(self.dim_hidden, self.dim_hidden, K=self.num_layers, cached=self.cached)
        else:
            self.layer_SGC = SGConv(self.num_feats, self.num_feats, K=self.num_layers, cached=self.cached)

        # set MLP layer
        self.layers_MLP.append(nn.Linear(self.num_feats, self.dim_hidden))

        # set residual connection type
        for i in range(self.num_layers):
            appendNormLayer(self, args, self.dim_hidden if self.has_residual_MLP else self.num_feats)
           
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
        
        # set jumping connection type and readout
        if AcontainsB(self.type_trick, ['Jumping']):
            if self.layer_agg in ['concat', 'maxpool']:
                self.layers_res.append(
                    DenseConnection((self.num_layers + 1) * self.dim_hidden, self.num_classes, self.layer_agg))
            elif self.layer_agg == 'attention':
                self.layers_res.append(
                    DenseConnection(self.dim_hidden, self.num_classes, self.layer_agg))
        elif self.has_residual_MLP:
            self.layers_MLP.append(nn.Linear(self.dim_hidden, self.num_classes))
        else:
            self.layers_MLP.append(nn.Linear(self.num_feats, self.num_classes))

        self.graph_dropout = DropoutTrick(args)
        if AcontainsB(self.type_trick, ['NoDropout']):
            self.dropout = 0.0
            self.embedding_dropout = 0.0

        # set optimizer
        self.reg_params = list(self.layer_SGC.parameters())
        self.non_reg_params = list(self.layers_MLP.parameters())
        self.optimizer = torch.optim.Adam([
            dict(params=self.reg_params, weight_decay=self.weight_decay1),
            dict(params=self.non_reg_params, weight_decay=self.weight_decay2)
        ], lr=self.lr)

    def forward(self, x, edge_index):

        edge_weight = None
        # edge_index, edge_weight = gcn_norm(
        #     edge_index, edge_weight, x.size(0), False, dtype=x.dtype)

        x_list = []
        new_adjs = self.graph_dropout(edge_index, edge_weight, adj_norm=True, num_nodes=x.size(0))
        if self.has_residual_MLP:
            x = self.layers_MLP[0](x)
            x = F.relu(x)
            x_list.append(x)

        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            edge_index, edge_weight = new_adjs[i]
            x = self.layer_SGC.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

            x = run_norm_if_any(self, x, i)
            # x = F.relu(x)
            x_list.append(x)
            if AcontainsB(self.type_trick, ['Initial', 'Dense', 'Residual']):
                x = self.layers_res[i](x_list)


        if AcontainsB(self.type_trick, ['Jumping']):
            x = self.layers_res[0](x_list)
        else:
            x = self.layers_MLP[-1](x)

        return x


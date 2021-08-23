from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, matmul


class APPNP(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, args, cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(APPNP, self).__init__(**kwargs)

        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        for k, v in vars(args).items():
            setattr(self, k, v)

        self.cached = self.transductive = args.transductive

        self.input_trans = torch.nn.Linear(self.num_feats, self.dim_hidden)
        self.output_trans = torch.nn.Linear(self.dim_hidden, self.num_classes)
        self.type_norm = 'None' if self.dataset != 'ogbn-arxiv' else 'batch'
        if self.type_norm == 'batch':
            self.input_bn = torch.nn.BatchNorm1d(self.dim_hidden)
            self.layers_bn = torch.nn.ModuleList([])
            for _ in range(self.num_layers):
                self.layers_bn.append(torch.nn.BatchNorm1d(self.num_classes))

        self.reg_params = list(self.input_trans.parameters())
        self.non_reg_params = list(self.output_trans.parameters())
        if self.type_norm == 'batch':
            for bn in self.layers_bn:
                self.reg_params += list(bn.parameters())

        self.optimizer = torch.optim.Adam([
            dict(params=self.reg_params, weight_decay=self.weight_decay1),
            dict(params=self.non_reg_params, weight_decay=self.weight_decay2)
        ], lr=self.lr)

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        # implemented based on: https://github.com/klicperajo/ppnp/blob/master/ppnp/pytorch/ppnp.py
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(0), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(0), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # input transformation according to the official implementation
        x = F.dropout(x, p=self.embedding_dropout, training=self.training)
        x = self.input_trans(x)
        if self.type_norm == 'batch':
            x = self.input_bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.embedding_dropout, training=self.training)
        h = self.output_trans(x)
        x = h

        for k in range(self.num_layers):
            if self.dropout > 0 and self.training:
                if isinstance(edge_index, Tensor):
                    assert edge_weight is not None
                    edge_weight = F.dropout(edge_weight, p=self.dropout, training=self.training)
                else:
                    value = edge_index.storage.value()
                    assert value is not None
                    value = F.dropout(value, p=self.dropout, training=self.training)
                    edge_index = edge_index.set_value(value, layout='coo')

            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                               size=None)
            if self.type_norm == 'batch':
                x = self.layers_bn[k](x)
            x = x * (1 - self.alpha)
            x += self.alpha * h

        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(K={}, alpha={})'.format(self.__class__.__name__, self.K,
                                           self.alpha)

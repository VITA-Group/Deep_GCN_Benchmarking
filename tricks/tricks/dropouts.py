import torch
import torch_scatter
from torch import nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import dropout_adj, subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes


# implemented based on GCNModel: https://github.com/DropEdge/DropEdge/blob/master/src/models.py
# Baseblock MultiLayerGCNBlock with nbaseblocklayer=1
class DropEdge(nn.Module):
    """
    DropEdge: Dropping edges using a uniform distribution.
    """
    def __init__(self, drop_rate):
        super(DropEdge, self).__init__()
        self.drop_rate = drop_rate
        self.undirected = False

    def forward(self, edge_index, edge_attr=None, edge_weight=None, num_nodes=None):
        return dropout_adj(edge_index, p=self.drop_rate, edge_attr=edge_attr,
                           force_undirected=self.undirected, training=self.training)

class DropNode(nn.Module):
    """
    DropNode: Sampling node using a uniform distribution.
    """

    def __init__(self, drop_rate):
        super(DropNode, self).__init__()
        self.drop_rate = drop_rate

    def forward(self, edge_index, edge_attr=None, edge_weight=None, num_nodes=None):
        if not self.training:
            return edge_index, edge_attr

        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        nodes = torch.arange(num_nodes, dtype=torch.int64)
        mask = torch.full_like(nodes, 1 - self.drop_rate, dtype=torch.float32)
        mask = torch.bernoulli(mask).to(torch.bool)
        subnodes = nodes[mask]

        return subgraph(subnodes, edge_index, edge_attr=edge_attr, num_nodes=num_nodes)

class FastGCN(nn.Module):
    """
    FastGCN: Sampling N nodes using a importance distribution.
    """
    def __init__(self, drop_rate):
        super(FastGCN, self).__init__()
        self.drop_rate = drop_rate

    def forward(self, edge_index, edge_attr=None, edge_weight=None, num_nodes=None):
        if not self.training:
            return edge_index, edge_attr

        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.shape[1],), device=edge_index.device)

        # Importance sampling: q(u) \propto \sum_{v \in N(u)} w^2(u,v)
        row, col = edge_index[0], edge_index[1]
        weight = torch_scatter.scatter_add(edge_weight**2, col, dim=0, dim_size=num_nodes)
        subnodes = torch.multinomial(weight, int(num_nodes*(1-self.drop_rate)), replacement=False)

        return subgraph(subnodes, edge_index, edge_attr=edge_attr, num_nodes=num_nodes)

class LADIES(nn.Module):
    """
    LADIES: Sampling N nodes dependent on the sampled nodes in the next layer.
    """
    def __init__(self, drop_rate, num_layers):
        super(LADIES, self).__init__()
        self.drop_rate = drop_rate
        self.num_layers = num_layers

    def forward(self, edge_index, edge_attr=None, edge_weight=None, num_nodes=None):
        if not self.training:
            return [(edge_index, edge_attr)]

        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.shape[1],), device=edge_index.device)

        sampled_edges = []
        last_edge_index = edge_index
        row_mask = torch.ones(edge_weight.shape[0], dtype=torch.bool)
        for i in range(self.num_layers):
            # Importance sampling: q(u) \propto \sum_{v \in N(u)} w^2(u,v)
            row, col = edge_index[0], edge_index[1]
            new_edge_weight = torch.zeros_like(edge_weight)
            new_edge_weight[row_mask] = edge_weight[row_mask]
            weight = torch_scatter.scatter_add(new_edge_weight**2, col, dim=0, dim_size=num_nodes)
            subnodes = torch.multinomial(weight, int(num_nodes*(1-self.drop_rate)), replacement=False)

            # create row mask for next iteration
            row_mask = torch.zeros(num_nodes, dtype=torch.bool)
            row_mask[subnodes] = True
            row_mask = row_mask[row]
            assert row_mask.shape[0] == edge_weight.shape[0]

            # record the sampled edges for sampling in the previous layer
            new_edge_index, new_edge_attr = subgraph(subnodes, edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
            sampled_edges.append((new_edge_index, new_edge_attr))
        # reverse the samples to the layer order
        sampled_edges.reverse()
        return sampled_edges


def AcontainsB(A,listB):
    # A: string; listB: list of strings
    for s in listB:
        if s in A: return True
    return False

class DroppedEdges(list):
    def __getitem__(self, i):
        if self.__len__() == 1:
            return super().__getitem__(0)
        else:
            return super().__getitem__(i)

class DropoutTrick(nn.Module):
    def __init__(self, args):
        super(DropoutTrick, self).__init__()
        self.type_trick = args.type_trick
        self.num_layers = args.num_layers
        self.layerwise_drop = args.layerwise_dropout

        if AcontainsB(self.type_trick, ['DropEdge']):
            self.graph_dropout = DropEdge(args.graph_dropout)
        elif AcontainsB(self.type_trick, ['DropNode']):
            self.graph_dropout = DropNode(args.graph_dropout)
        elif AcontainsB(self.type_trick, ['FastGCN']):
            self.graph_dropout = FastGCN(args.graph_dropout)
        elif AcontainsB(self.type_trick, ['LADIES']):
            self.layerwise_drop = True
            assert self.layerwise_drop, 'LADIES requires layer-wise dropout flag on'
            self.graph_dropout = LADIES(args.graph_dropout, args.num_layers)
        else:
            self.graph_dropout = None


    def forward(self, edge_index, edge_weight=None, adj_norm=False, num_nodes=-1):
        if self.graph_dropout is not None:
            if AcontainsB(self.type_trick, ['LADIES']):
                for dp_edges, dp_weights in self.graph_dropout(edge_index, edge_attr=edge_weight, edge_weight=edge_weight):
                    if adj_norm:
                            dp_edges, dp_weights = gcn_norm(dp_edges, dp_weights, num_nodes, False)
                    new_adjs = DroppedEdges([(dp_edges, dp_weights) ])
            else:
                new_adjs = DroppedEdges()
                if self.layerwise_drop:
                    for _ in range(self.num_layers):
                        dp_edges, dp_weights = self.graph_dropout(edge_index, edge_attr=edge_weight, edge_weight=edge_weight)
                        if adj_norm:
                            dp_edges, dp_weights = gcn_norm(dp_edges, dp_weights, num_nodes, False)
                        new_adjs.append((dp_edges, dp_weights))
                else:
                    dp_edges, dp_weights = self.graph_dropout(edge_index, edge_attr=edge_weight, edge_weight=edge_weight)
                    if adj_norm:
                            dp_edges, dp_weights = gcn_norm(dp_edges, dp_weights, num_nodes, False)
                    new_adjs.append((dp_edges, dp_weights))
        else:
            # no dropout
            if adj_norm:
                edge_index, edge_weight = gcn_norm(edge_index, edge_weight, num_nodes, False)
            new_adjs = DroppedEdges([(edge_index, edge_weight)])
        return new_adjs


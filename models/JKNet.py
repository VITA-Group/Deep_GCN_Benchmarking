import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class JKNet(torch.nn.Module):
    """An implementation of Jumping Knowledge Network (arxiv 1806.03536) which
    combine layers with Maxpool.
    Args:
        in_features (int): Size of each input node.
        out_features (int): Size of each output node.
        n_layers (int): Number of the convolution layers.
        n_units (int): Size of the mtorchiddle layers.
        aggregation (str): 'sum', 'mean' or 'max'.
                           Specify the way to aggregate the neighbourhoods.
    """
    def __init__(self, args):
        super(JKNet, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.cached = self.transductive = args.transductive
        self.layers_GCN = torch.nn.ModuleList([])
        self.layers_bn = torch.nn.ModuleList([])

        self.layers_GCN.append(GCNConv(self.num_feats, self.dim_hidden, cached=self.cached))
        if self.type_norm == 'batch':
            self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))

        for i in range(1, self.num_layers):
            self.layers_GCN.append(
                GCNConv(self.dim_hidden, self.dim_hidden, cached=self.cached))

            if self.type_norm == 'batch':
                self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))

        self.last_linear = torch.nn.Linear(self.dim_hidden * self.num_layers, self.num_classes)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, x, edge_index):
        # implemented based on: https://github.com/ShinKyuY/Representation_Learning_on_Graphs_with_Jumping_Knowledge_Networks/blob/master/JK-Net.ipynb

        layer_outputs = []
        for i in range(self.num_layers):
            x = self.layers_GCN[i](x, edge_index)
            if self.type_norm == 'batch':
                x = self.layers_bn[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            layer_outputs.append(x)

        h = torch.cat(layer_outputs, dim=1)
        x = self.last_linear(h)
        return x
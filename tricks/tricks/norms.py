import torch
import torch.nn as nn
import torch.nn.functional as F


class comb_norm(torch.nn.Module):
    def __init__(self, norm_list):
        super().__init__()
        self.norm_list = nn.ModuleList(norm_list)

    def forward(self, x):
        for mod in self.norm_list:
            x = mod(x)
        return x


class pair_norm(torch.nn.Module):
    def __init__(self):
        super(pair_norm, self).__init__()
        # print(f'------ ›››››››››  {self._get_name()}')

    def forward(self, x):
        col_mean = x.mean(dim=0)
        x = x - col_mean
        rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
        x = x / rownorm_mean
        return x


class mean_norm(torch.nn.Module):
    def __init__(self):
        super(mean_norm, self).__init__()
        # print(f'------ ›››››››››  {self._get_name()}')

    def forward(self, x):
        col_mean = x.mean(dim=0)
        x = x - col_mean
        return x


class node_norm(torch.nn.Module):
    def __init__(self, node_norm_type="n", unbiased=False, eps=1e-5, power_root=2, **kwargs):
        super(node_norm, self).__init__()
        self.unbiased = unbiased
        self.eps = eps
        self.node_norm_type = node_norm_type
        self.power = 1 / power_root
        # print(f'------ ›››››››››  {self._get_name()}')

    def forward(self, x):
        # in GCN+Cora, 
        # n v srv pr
        # 16 layer:  _19.8_  15.7 17.4 17.3
        # 32 layer:  20.3 _25.5_ 16.2 16.3

        if self.node_norm_type == "n":
            mean = torch.mean(x, dim=1, keepdim=True)
            std = (
                    torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps
            ).sqrt()
            x = (x - mean) / std
        elif self.node_norm_type == "v":
            std = (
                    torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps
            ).sqrt()
            x = x / std

        elif self.node_norm_type == "m":
            mean = torch.mean(x, dim=1, keepdim=True)
            x = x - mean
        elif self.node_norm_type == "srv":  # squre root of variance
            std = (
                    torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps
            ).sqrt()
            x = x / torch.sqrt(std)
        elif self.node_norm_type == "pr":
            std = (
                    torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps
            ).sqrt()
            x = x / torch.pow(std, self.power)
        return x

    def __repr__(self):
        original_str = super().__repr__()
        components = list(original_str)
        node_norm_type_str = f"node_norm_type={self.node_norm_type}"
        components.insert(-1, node_norm_type_str)
        new_str = "".join(components)
        return new_str


class group_norm(torch.nn.Module):
    def __init__(self, dim_to_norm=None, dim_hidden=16, num_groups=None, skip_weight=None, **w):
        super(group_norm, self).__init__()
        self.num_groups = num_groups
        self.skip_weight = skip_weight

        dim_hidden = dim_hidden if dim_to_norm is None else dim_to_norm
        self.dim_hidden = dim_hidden

        # print(f'\n\n{dim_to_norm}\n\n');raise

        self.bn = torch.nn.BatchNorm1d(dim_hidden * self.num_groups, momentum=0.3)
        self.group_func = torch.nn.Linear(dim_hidden, self.num_groups, bias=True)
        # print(f'------ ›››››››››  {self._get_name()}')

    def forward(self, x):
        if self.num_groups == 1:
            x_temp = self.bn(x)
        else:
            score_cluster = F.softmax(self.group_func(x), dim=1)
            x_temp = torch.cat([score_cluster[:, group].unsqueeze(dim=1) * x for group in range(self.num_groups)],
                               dim=1)
            x_temp = self.bn(x_temp).view(-1, self.num_groups, self.dim_hidden).sum(dim=1)

        x = x + x_temp * self.skip_weight
        return x


def AcontainsB(A, listB):
    # A: string; listB: list of strings
    for s in listB:
        if s in A: return True
    return False


def appendNormLayer(net, args, dim_to_norm=None):
    if AcontainsB(args.type_trick, ['BatchNorm']):
        net.layers_norm.append(torch.nn.BatchNorm1d(net.dim_hidden if dim_to_norm is None else dim_to_norm))
    elif AcontainsB(args.type_trick, ['PairNorm']):
        net.layers_norm.append(pair_norm())
    elif AcontainsB(args.type_trick, ['NodeNorm']):
        net.layers_norm.append(node_norm(**vars(net.args)))
    elif AcontainsB(args.type_trick, ['MeanNorm']):
        net.layers_norm.append(mean_norm())
    elif AcontainsB(args.type_trick, ['GroupNorm']):
        net.layers_norm.append(group_norm(dim_to_norm, **vars(reset_weight_GroupNorm(args))))
    elif AcontainsB(args.type_trick, ['CombNorm']):
        net.layers_norm.append(
            comb_norm([group_norm(dim_to_norm, **vars(reset_weight_GroupNorm(args))), node_norm(**vars(net.args))]))


def run_norm_if_any(net, x, ilayer):
    if AcontainsB(net.args.type_trick, ['BatchNorm', 'PairNorm', 'NodeNorm', 'MeanNorm', 'GroupNorm', 'CombNorm']):
        return net.layers_norm[ilayer](x)
    else:
        return x


def reset_weight_GroupNorm(args):
    if args.num_groups is not None:
        return args

    args.miss_rate = 0.

    if args.dataset == 'Citeseer' and args.miss_rate == 0.:
        if args.type_model in ['GAT', 'GCN']:
            args.skip_weight = 0.001 if args.num_layers < 6 else 0.005
        else:
            args.skip_weight = 0.0005 if args.num_layers < 60 else 0.002

    elif args.dataset == 'ogbn-arxiv' and args.miss_rate == 0.:
        if args.type_model in ['GAT', 'GCN']:
            args.skip_weight = 0.001 if args.num_layers < 6 else 0.005
        else:
            args.skip_weight = 0.0005 if args.num_layers < 60 else 0.002

    elif args.dataset == 'Pubmed' and args.miss_rate == 0.:
        if args.type_model in ['GCN']:
            args.skip_weight = 0.001 if args.num_layers < 6 else 0.01
        elif args.type_model in ['GAT']:
            args.skip_weight = 0.005 if args.num_layers < 6 else 0.01
        else:
            args.skip_weight = 0.05

    elif args.dataset == 'Cora' and args.miss_rate == 0.:
        if args.type_model in ['GCN']:
            args.skip_weight = 0.001 if args.num_layers < 6 else 0.03
        elif args.type_model in ['GAT']:
            args.skip_weight = 0.001 if args.num_layers < 6 else 0.01
        else:
            args.skip_weight = 0.01 if args.num_layers < 60 else 0.005

    elif args.dataset == 'CoauthorCS' and args.miss_rate == 0.:
        if args.type_model in ['GAT', 'GCN']:
            args.skip_weight = 0.001 if args.num_layers < 6 else 0.03
        else:
            args.epochs = 500
            args.skip_weight = 0.001 if args.num_layers < 10 else .5
    elif args.dataset in ['CoauthorCS', 'CoauthorPhysics', 'AmazonComputers', 'AmazonPhoto',
                          'TEXAS', 'WISCONSIN', 'CORNELL']:
        args.skip_weight = 0.005

    else:
        raise NotImplementedError

    # -wz
    if args.dataset == 'Pubmed':
        args.num_groups = 5
    else:
        args.num_groups = 10

    return args

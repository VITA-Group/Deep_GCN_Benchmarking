import argparse


class BaseOptions():

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""

    def initialize(self):
        parser = argparse.ArgumentParser(description='Constrained learing')

        parser.add_argument("--dataset", type=str, default="Cora", required=False,
                            help="The input dataset.",
                            choices=['Cora', 'Citeseer', 'Pubmed', 'ogbn-arxiv',
                                     'CoauthorCS', 'CoauthorPhysics', 'AmazonComputers', 'AmazonPhoto',
                                     'TEXAS', 'WISCONSIN', 'ACTOR', 'CORNELL'])
        # build up the common parameter
        parser.add_argument('--random_seed', type=int, default=100)
        parser.add_argument('--N_exp', type=int, default=100)
        parser.add_argument('--resume', action='store_true', default=False)
        parser.add_argument("--cuda", type=bool, default=True, required=False,
                            help="run in cuda mode")
        parser.add_argument('--cuda_num', type=int, default=0, help="GPU number")
        parser.add_argument('--log_file_name', type=str, default='time_and_memory.log')

        parser.add_argument('--compare_model', type=int, default=0,
                            help="0: test tricks, 1: test models")

        parser.add_argument('--type_model', type=str, default="GCN",
                            choices=['GCN', 'GAT', 'SGC', 'GCNII', 'DAGNN', 'GPRGNN', 'APPNP', 'JKNet', 'DeeperGCN'])
        parser.add_argument('--type_trick', type=str, default="None")
        parser.add_argument('--layer_agg', type=str, default='concat',
                            choices=['concat', 'maxpool', 'attention', 'mean'],
                            help='aggregation function for skip connections')

        parser.add_argument('--num_layers', type=int, default=64)
        parser.add_argument("--epochs", type=int, default=1000,
                            help="number of training the one shot model")
        parser.add_argument('--patience', type=int, default=100,
                            help="patience step for early stopping")  # 5e-4
        parser.add_argument("--multi_label", type=bool, default=False,
                            help="multi_label or single_label task")
        parser.add_argument("--dropout", type=float, default=0.6,
                            help="dropout for GCN")
        parser.add_argument('--embedding_dropout', type=float, default=0.6,
                            help='dropout for embeddings')
        parser.add_argument("--lr", type=float, default=0.005,
                            help="learning rate")
        parser.add_argument('--weight_decay', type=float, default=5e-4,
                            help="weight decay")  # 5e-4
        parser.add_argument('--dim_hidden', type=int, default=64)
        parser.add_argument('--transductive', type=bool, default=True,
                            help='transductive or inductive setting')
        parser.add_argument('--activation', type=str, default="relu", required=False)

        # Hyperparameters for specific model, such as GCNII, EdgeDropping, APPNNP, PairNorm
        parser.add_argument('--alpha', type=float, default=0.1,
                            help="residual weight for input embedding")
        parser.add_argument('--lamda', type=float, default=0.5,
                            help="used in identity_mapping and GCNII")
        parser.add_argument('--weight_decay1', type=float, default=0.01, help='weight decay in some models')
        parser.add_argument('--weight_decay2', type=float, default=5e-4, help='weight decay in some models')
        parser.add_argument('--type_norm', type=str, default="None")
        parser.add_argument('--adj_dropout', type=float, default=0.5,
                            help="dropout rate in APPNP")  # 5e-4
        parser.add_argument('--edge_dropout', type=float, default=0.2,
                            help="dropout rate in EdgeDrop")  # 5e-4

        parser.add_argument('--node_norm_type', type=str, default="n", choices=['n', 'v', 'm', 'srv', 'pr'])
        parser.add_argument('--skip_weight', type=float, default=None)
        parser.add_argument('--num_groups', type=int, default=None)
        parser.add_argument('--has_residual_MLP', type=bool, default=False)

        # Hyperparameters for random dropout
        parser.add_argument('--graph_dropout', type=float, default=0.2,
                            help="graph dropout rate (for dropout tricks)")  # 5e-4
        parser.add_argument('--layerwise_dropout', action='store_true', default=False)

        args = parser.parse_args()
        args = self.reset_dataset_dependent_parameters(args)

        return args

    ## setting the common hyperparameters used for comparing different methods of a trick
    def reset_dataset_dependent_parameters(self, args):
        if args.dataset == 'Cora':
            args.num_feats = 1433
            args.num_classes = 7
            args.dropout = 0.6  # 0.5
            args.lr = 0.005  # 0.005
            args.weight_decay = 5e-4
            args.epochs = 1000
            args.patience = 100
            args.dim_hidden = 64
            args.activation = 'relu'

            # args.N_exp = 100

        elif args.dataset == 'Pubmed':
            args.num_feats = 500
            args.num_classes = 3
            args.dropout = 0.5
            args.lr = 0.01
            args.weight_decay = 5e-4
            args.epochs = 1000
            args.patience = 100
            args.dim_hidden = 256
            args.activation = 'relu'

        elif args.dataset == 'Citeseer':
            args.num_feats = 3703
            args.num_classes = 6

            args.dropout = 0.7
            args.lr = 0.01
            args.lamda = 0.6
            args.weight_decay = 5e-4
            args.epochs = 1000
            args.patience = 100
            args.dim_hidden = 256
            args.activation = 'relu'

            args.res_alpha = 0.2

        elif args.dataset == 'ogbn-arxiv':
            args.num_feats = 128
            args.num_classes = 40
            args.dropout = 0.1
            args.lr = 0.005
            args.weight_decay = 0.
            args.epochs = 1000
            args.patience = 200
            args.dim_hidden = 256


        # ==============================================
        # ========== below are other datasets ==========

        elif args.dataset == 'CoauthorPhysics':
            args.epochs = 1000
            args.patience = 100
            args.dim_hidden = 256
            args.activation = 'relu'

            args.num_feats = 8415
            args.num_classes = 5

            args.dropout = 0.8
            args.lr = 0.005
            args.weight_decay = 0.



        elif args.dataset == 'CoauthorCS':
            args.epochs = 1000
            args.patience = 100
            args.dim_hidden = 256
            args.activation = 'relu'

            args.num_feats = 6805
            args.num_classes = 15

            args.dropout = 0.8
            args.lr = 0.005
            args.weight_decay = 0.




        elif args.dataset == 'TEXAS':
            args.epochs = 1000
            args.patience = 100
            args.dim_hidden = 256
            args.activation = 'relu'

            args.num_feats = 1703
            args.num_classes = 5

            args.dropout = 0.6
            args.lr = 0.005
            args.weight_decay = 5e-4

            args.res_alpha = 0.9




        elif args.dataset == 'WISCONSIN':
            args.epochs = 1000
            args.patience = 100
            args.dim_hidden = 256
            args.activation = 'relu'

            args.num_feats = 1703
            args.num_classes = 5

            args.dropout = 0.6
            args.lr = 0.005
            args.weight_decay = 5e-4

            args.res_alpha = 0.9


        elif args.dataset == 'CORNELL':
            args.epochs = 1000
            args.patience = 100
            args.dim_hidden = 256
            args.activation = 'relu'

            args.num_feats = 1703
            args.num_classes = 5

            args.dropout = 0.
            args.lr = 0.005
            args.weight_decay = 5e-4

            args.res_alpha = 0.9


        elif args.dataset == 'ACTOR':
            args.epochs = 1000
            args.patience = 100
            args.dim_hidden = 256
            args.activation = 'relu'

            args.num_feats = 932
            args.num_classes = 5

            args.dropout = 0.
            args.lr = 0.005
            args.weight_decay = 5e-4

            args.res_alpha = 0.9

        elif args.dataset == 'AmazonComputers':
            args.epochs = 1000
            args.patience = 100
            args.dim_hidden = 256
            args.activation = 'relu'

            args.num_feats = 767
            args.num_classes = 10

            args.dropout = 0.5
            args.lr = 0.005
            args.weight_decay = 5e-5

        elif args.dataset == 'AmazonPhoto':
            args.epochs = 1000
            args.patience = 100
            args.dim_hidden = 256
            args.activation = 'relu'

            args.num_feats = 745
            args.num_classes = 8

            args.dropout = 0.5
            args.lr = 0.005
            args.weight_decay = 5e-4

        return args

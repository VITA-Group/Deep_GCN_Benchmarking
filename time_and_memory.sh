datasets=(Cora Citeseer Pubmed)
num_layers=(2 4 8 16 32 64)
backbones=(GCN SGC)
type_tricks=(None DropEdge DropNode FastGCN LADIES BatchNorm PairNorm NodeNorm MeanNorm GroupNorm CombNorm Residual Initial Jumping Dense IdentityMapping)

for dataset in "${datasets[@]}"; do
    for num_layer in "${num_layers[@]}"; do
        for backbone in "${backbones[@]}"; do
            for type_trick in "${type_tricks[@]}"; do
                python time_and_memory.py --cuda_num=7 --dataset="$dataset" --type_trick="$type_trick" --num_layers="$num_layer" --type_model="$backbone"
            done
        done
    done
done

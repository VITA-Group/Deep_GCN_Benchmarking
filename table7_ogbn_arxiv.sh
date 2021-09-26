datasets=(ogbn-arxiv)
num_layers=(4 8 24 64 128)
type_models=(SGC GCNII DAGNN GPRGNN APPNP JKNet)

for dataset in "${datasets[@]}"; do
    for num_layer in "${num_layers[@]}"; do
        for type_model in "${type_models[@]}"; do
                python main.py --dataset="$dataset" --num_layers="$num_layer" --type_model="$type_model"
        done
    done
done
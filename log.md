> delete when finishing cleaning

## Done

#### 8.11

- move the `load_data` and `load_ogbn_data` from `trainer.py` to `Dataloader.py`
- delete all the "recording" functions that read or write files
- clean the `main.py`

## TODO

#### by 8.15

*benchmarks*

- [x] remove the hyperparamter settings inside the model files
- [x] set an optimizer for each model (instead of use the same optimizer in `trainer.py`)
- [x] make the model files hierarchical? (GCNII_layer and GCNII should not be in the same folder)

*trick combinations*

- [x] remove `trick_comb_arxiv` and `trick_comb_cora`
- [x] combine all tricks together and clean them up
- [x] add `Identity Mapping` as a independent setting
## Usage
```python
# DBLP
python main.py --dataset dblp --diff heat --hidden_dim 128 --n_classes 64 --n_layers 2 --fanout 20,20 --snapshots 4 --views 4 --strategy sequential --epochs 200 --GPU 0 

# Bitcoinotc
python main.py --dataset bitcoinotc --diff ppr --hidden_dim 128 --n_classes 64 --n_layers 2 --fanout 10,10 --snapshots 4 --views 4 --strategy sequential --dataloader_size 64 --epochs 25 --GPU 0 

# TAX
python main.py --dataset tax --diff heat --hidden_dim 128 --n_classes 64 --n_layers 2 --fanout 20,20 --snapshots 4 --views 4 --strategy sequential --epochs 200 --GPU 0

# BITotc
python main.py --dataset bitotc --diff ppr --hidden_dim 128 --n_classes 64 --n_layers 2 --fanout 10,10 --snapshots 4 --views 4 --strategy sequential --epochs 50 --GPU 0 

# BITalpha
python main.py --dataset bitalpha --diff ppr --hidden_dim 128 --n_classes 64 --n_layers 2 --fanout 20,20 --snapshots 5 --views 5 --strategy sequential --epochs 100 --GPU 0 

# TAX51
python main.py --dataset tax51 --diff heat --hidden_dim 128 --n_classes 64 --n_layers 2 --fanout 20,20 --snapshots 4 --views 4 --strategy sequential --epochs 200 --GPU 0 

# reddit
python main.py --dataset reddit --diff heat --hidden_dim 128 --n_classes 64 --n_layers 2 --fanout 20,20 --snapshots 4 --views 4 --strategy sequential --epochs 200 --GPU 0
```

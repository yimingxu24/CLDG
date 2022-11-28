# CLDG: Contrastive Learning on Dynamic Graphs (ICDE'23)
## Code structure
| *folder*  |                         description                          |
| :-------: | :----------------------------------------------------------: |
|   Data    |      Datasets.       |
|   CLDG    | CLDG implementation code is provided. |

## Datasets

| **Dataset** | # nodes |  # temporal edges  | # classes |
| :---------: | :--------: | :-----: | :--------: |
|  **DBLP**   |     25,387      | 185,480 |     10      |
|  **Bitcoinotc**   |     5,881      | 35,592 |     3      |
|  **TAX**   |     27,097      | 315,478 |     12      |
|  **BITotc**   |     4,863      | 28,473 |     7      |
|  **BITalpha**   |     3,219      | 19,364 |     7      |
| **TAX51** | 132,524 | 467,279 | 51 |
| **Reddit** | 898,194 | 2,575,464 | 3 |

## Usage
```python
# DBLP
python main.py --dataset dblp --hidden_dim 128 --n_classes 64 --n_layers 2 --fanout 20,20 --snapshots 4 --views 4 --strategy sequential --epochs 200 --GPU 0

# Bitcoinotc
python main.py --dataset bitcoinotc --hidden_dim 128 --n_classes 64 --n_layers 2 --fanout 10,10 --snapshots 4 --views 3 --strategy sequential --dataloader_size 64 --epochs 25 --GPU 0

# TAX
python main.py --dataset tax --hidden_dim 128 --n_classes 64 --n_layers 2 --fanout 20,20 --snapshots 4 --views 4 --strategy sequential --epochs 200 --GPU 0

# BITotc
python main.py --dataset bitotc --hidden_dim 128 --n_classes 64 --n_layers 2 --fanout 10,10 --snapshots 4 --views 4 --strategy random --epochs 50 --GPU 0

# BITalpha
python main.py --dataset bitalpha --hidden_dim 128 --n_classes 64 --n_layers 2 --fanout 20,20 --snapshots 6 --views 4 --strategy sequential --epochs 200 --GPU 0

# TAX51
python main.py --dataset tax51 --hidden_dim 128 --n_classes 64 --n_layers 2 --fanout 20,20 --snapshots 8 --views 5 --strategy random --epochs 200 --GPU 0

# reddit
python main.py --dataset reddit --hidden_dim 128 --n_classes 64 --n_layers 2 --fanout 20,20 --snapshots 5 --views 4 --strategy random --epochs 200 --GPU 0
```

## Dependencies

- Python 3.7
- PyTorch 1.9.0+cu111
- dgl-cu110 0.7.1
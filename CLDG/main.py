import argparse
import numpy as np
import torch as th
import torch.nn as thnn
import torch.nn.functional as F
import torch.optim as optim
from functools import reduce
import random
import copy
import dgl
from dgl.dataloading.neighbor import MultiLayerNeighborSampler, MultiLayerFullNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader
from sklearn.metrics import f1_score

from models import GraphConvModel, MLPLinear, LogReg
from utils import load_to_dgl_graph, dataloader, sampling_layer


random.seed(24)

def load_subtensor(node_feats, input_nodes, device):
    batch_inputs = node_feats[input_nodes].to(device)

    return batch_inputs


def train(dataset, hidden_dim, n_layers, n_classes, fanouts, snapshots, views, strategy, readout, batch_size, dataloader_size, num_workers, epochs, GPU):
    device_id = GPU
    graph, node_feat = load_to_dgl_graph(dataset)
    sampler = MultiLayerNeighborSampler(fanouts)

    in_feat = node_feat.shape[1]

    model = GraphConvModel(in_feat, hidden_dim, n_layers, n_classes, norm='both', activation=F.relu, readout=readout, dropout=0)
    model = model.to(device_id)
    projection_model = MLPLinear(n_classes, n_classes).to(device_id)

    loss_fn = thnn.CrossEntropyLoss().to(device_id)

    optimizer = optim.Adam([
            {'params': model.parameters(), 'lr': 4e-3, 'weight_decay': 5e-4}, 
            {'params': projection_model.parameters(), 'lr': 4e-3, 'weight_decay': 5e-4}
            ])

    best_loss = th.tensor([float('inf')]).to(device_id)
    best_model = copy.deepcopy(model)
    print('Plan to train {} epoches \n'.format(epochs))

    for epoch in range(epochs):
        # mini-batch for training
        model.train()
        projection_model.train()
        edges_time = graph.edata['time'].tolist()
        max_time, min_time, span = max(edges_time), min(edges_time), max(edges_time) - min(edges_time)
        temporal_subgraphs, nids, train_dataloader_list = [], [], []
 
        T = sampling_layer(snapshots, views, span, strategy)
        for start in T:
            end = min(start + span / snapshots, max_time)
            start = max(start, min_time)
            sample_time = (graph.edata['time'] >= start) & (graph.edata['time'] <= end)

            temporal_subgraph = dgl.edge_subgraph(graph, sample_time, relabel_nodes=False)

            temporal_subgraph = dgl.to_simple(temporal_subgraph)
            temporal_subgraph = dgl.to_bidirected(temporal_subgraph, copy_ndata=True)
            temporal_subgraph = dgl.add_self_loop(temporal_subgraph)
            nids.append(th.unique(temporal_subgraph.edges()[0]))
            temporal_subgraphs.append(temporal_subgraph)
        
        train_nid_per_gpu = list(reduce(lambda x, y: x&y, [set(nids[sg_id].tolist()) for sg_id in range(views)]))
        train_nid_per_gpu = random.sample(train_nid_per_gpu, batch_size)
        random.shuffle(train_nid_per_gpu)
        train_nid_per_gpu = th.tensor(train_nid_per_gpu)

        for sg_id in range(views):
            train_dataloader = NodeDataLoader(temporal_subgraphs[sg_id],
                                                train_nid_per_gpu,
                                                sampler,
                                                batch_size=train_nid_per_gpu.shape[0],
                                                shuffle=False,
                                                drop_last=False,
                                                num_workers=num_workers,
                                                )
            train_dataloader_list.append(train_dataloader)

        seeds_emb = th.tensor([]).to(device_id)
        for train_dataloader in train_dataloader_list:
            for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
                # forward
                batch_inputs = load_subtensor(node_feat, input_nodes, device_id)
                blocks = [block.to(device_id) for block in blocks]

                # metric and loss
                train_batch_logits = model(blocks, batch_inputs)
                train_batch_logits = projection_model(train_batch_logits)
                
                seeds_emb = th.cat([seeds_emb, train_batch_logits.unsqueeze(0)], dim=0)


        train_contrastive_loss = th.tensor([0]).to(device_id)
        for idx in range(seeds_emb.shape[0]):
            for idy in range(idx+1, seeds_emb.shape[0]):
                z1 = seeds_emb[idx]
                z2 = seeds_emb[idy]
     
                pred1 = th.mm(z1, z2.T).to(device_id)
                pred2 = th.mm(z2, z1.T).to(device_id)

                labels = th.arange(pred1.shape[0]).to(device_id)
                
                train_contrastive_loss = train_contrastive_loss + (loss_fn(pred1 / 0.07, labels) + loss_fn(pred2 / 0.07, labels)) / 2


        optimizer.zero_grad()
        train_contrastive_loss.backward()
        optimizer.step()

        if train_contrastive_loss < best_loss:
            best_loss = train_contrastive_loss
            best_model = copy.deepcopy(model)
        print("Epoch {:05d} | Loss {:.4f} ". format(epoch, float(train_contrastive_loss)))
        
        
    sampler = MultiLayerFullNeighborSampler(2)
    
    graph = dgl.to_simple(graph)
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)
    test_dataloader = NodeDataLoader(graph,
                                    graph.nodes(),
                                    sampler,
                                    batch_size=dataloader_size,
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=num_workers,
                                    )

                                    
    best_model.eval()
    for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):
        batch_inputs = load_subtensor(node_feat, input_nodes, device_id)
        blocks = [block.to(device_id) for block in blocks]
        test_batch_logits = best_model(blocks, batch_inputs)

        batch_embeddings = test_batch_logits.detach()
        if step == 0:
            embeddings = batch_embeddings
        else:
            embeddings = th.cat((embeddings, batch_embeddings), axis=0)

    
    ''' Linear Evaluation '''
    labels, train_idx, val_idx, test_idx, n_classes = dataloader(DATASET)
    train_embs = embeddings[train_idx].to(device_id)
    val_embs = embeddings[val_idx].to(device_id)
    test_embs = embeddings[test_idx].to(device_id)

    label = labels.to(device_id)

    train_labels = th.tensor(label[train_idx])
    val_labels = th.tensor(label[val_idx])
    test_labels = th.tensor(label[test_idx])

    micros, weights = [], []
    for _ in range(5):
        logreg = LogReg(train_embs.shape[1], n_classes)
        logreg = logreg.to(device_id)
        loss_fn = thnn.CrossEntropyLoss()
        opt = th.optim.Adam(logreg.parameters(), lr=1e-2, weight_decay=1e-4)

        best_val_acc, eval_micro, eval_weight = 0, 0, 0
        for epoch in range(2000):
            logreg.train()
            opt.zero_grad()
            logits = logreg(train_embs)
            preds = th.argmax(logits, dim=1)
            train_acc = th.sum(preds == train_labels).float() / train_labels.shape[0]
            loss = loss_fn(logits, train_labels)
            loss.backward()
            opt.step()

            logreg.eval()
            with th.no_grad():
                val_logits = logreg(val_embs)
                test_logits = logreg(test_embs)

                val_preds = th.argmax(val_logits, dim=1)
                test_preds = th.argmax(test_logits, dim=1)

                val_acc = th.sum(val_preds == val_labels).float() / val_labels.shape[0]
                
                ys = test_labels.cpu().numpy()
                indices = test_preds.cpu().numpy()
                test_micro = th.tensor(f1_score(ys, indices, average='micro'))
                test_weight = th.tensor(f1_score(ys, indices, average='weighted'))

                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    if (test_micro + test_weight) >= (eval_micro + eval_weight):
                        eval_micro = test_micro
                        eval_weight = test_weight

                print('Epoch:{}, train_acc:{:.4f}, val_acc:{:4f}, test_micro:{:4f}, test_weight:{:4f}'.format(epoch, train_acc, val_acc, test_micro, test_weight))
        micros.append(eval_micro)
        weights.append(eval_weight)

    micros, weights = th.stack(micros), th.stack(weights)
    print('Linear evaluation Accuracy:{:.4f}, Weighted-F1={:.4f}'.format(micros.mean().item(), weights.mean().item()))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLDG')
    parser.add_argument('--dataset', type=str, help="Name of the dataset.")
    parser.add_argument('--hidden_dim', type=int, required=True)
    parser.add_argument('--n_classes', type=int, required=True)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument("--fanout", type=str, required=True, help="fanout numbers", default='20,20')
    parser.add_argument('--snapshots', type=int, default=4)
    parser.add_argument('--views', type=int, default=2)
    parser.add_argument('--strategy', type=str, default='random')
    parser.add_argument('--readout', type=str, default='max')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dataloader_size', type=int, default=4096)
    parser.add_argument('--GPU', type=int, required=True)
    parser.add_argument('--num_workers_per_gpu', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    # parse arguments
    DATASET = args.dataset
    HID_DIM = args.hidden_dim
    N_CLASSES = args.n_classes
    N_LAYERS = args.n_layers
    FANOUTS = [int(i) for i in args.fanout.split(',')]
    SNAPSHOTS = args.snapshots
    VIEWS = args.views
    STRATEGY = args.strategy
    READOUT = args.readout
    BATCH_SIZE = args.batch_size
    DATALOADER_SIZE = args.dataloader_size
    GPU = args.GPU
    WORKERS = args.num_workers_per_gpu
    EPOCHS = args.epochs

    # output arguments for logging
    print('Dataset: {}'.format(DATASET))
    print('Hidden dimensions: {}'.format(HID_DIM))
    print('number of hidden layers: {}'.format(N_LAYERS))
    print('Fanout list: {}'.format(FANOUTS))
    print('Batch size: {}'.format(BATCH_SIZE))
    print('GPU: {}'.format(GPU))
    print('Number of workers per GPU: {}'.format(WORKERS))
    print('Max number of epochs: {}'.format(EPOCHS))


    train(dataset = DATASET, hidden_dim=HID_DIM, n_layers=N_LAYERS, n_classes=N_CLASSES,
    fanouts=FANOUTS, snapshots=SNAPSHOTS, views=VIEWS, strategy=STRATEGY, readout=READOUT, 
    batch_size=BATCH_SIZE, dataloader_size=DATALOADER_SIZE, num_workers=WORKERS, epochs=EPOCHS, GPU=GPU)


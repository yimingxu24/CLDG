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
from sklearn.metrics import f1_score, roc_auc_score

from models import GraphConvModel, MLPLinear, LogReg
from utils import load_to_dgl_graph, dataloader, sampling_layer

from scipy.linalg import fractional_matrix_power, inv
import networkx as nx


def compute_ppr(graph: nx.Graph, alpha=0.2, self_loop=True):
    a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])  # A^ = A + I_n
    d = np.diag(np.sum(a, 1))  # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)  # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)  # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))  # a(I_n-(1-a)A~)^-1

def compute_heat(graph: nx.Graph, t=10, self_loop=True):
    a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])
    d = np.diag(np.sum(a, 1))
    return np.exp(t * (np.matmul(a, inv(d)) - 1))

def load_subtensor(node_feats, input_nodes, device):
    batch_inputs = node_feats[input_nodes].to(device)

    return batch_inputs

def train(dataset, hidden_dim, n_layers, n_classes, fanouts, snapshots, views, strategy, readout, batch_size, dataloader_size, alpha, threshold, diff, num_workers, epochs, GPU):
    device_id = GPU
    inject_graph, inject_node_feat, anomaly_label = load_to_dgl_graph(dataset, snapshots)
    sampler = MultiLayerNeighborSampler(fanouts)

    in_feat = inject_node_feat[0].shape[1]

    model = GraphConvModel(in_feat, hidden_dim, n_layers, n_classes, norm='both', activation=F.relu, readout=readout, dropout=0)
    model = model.to(device_id)
    diff_model = GraphConvModel(in_feat, hidden_dim, n_layers, n_classes, norm='both', activation=F.relu, readout=readout, dropout=0)
    diff_model = diff_model.to(device_id)

    loss_fn = thnn.CrossEntropyLoss().to(device_id)

    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 4e-3, 'weight_decay': 5e-4}, 
        {'params': diff_model.parameters(), 'lr': 4e-3, 'weight_decay': 5e-4}
        ])

    best_loss = th.tensor([float('inf')]).to(device_id)
    best_model = copy.deepcopy(model)
    best_diff_model = copy.deepcopy(diff_model)

    print('Plan to train {} epoches \n'.format(epochs))

    for epoch in range(epochs):
        # mini-batch for training
        model.train()
        diff_model.train()

        edges_time = inject_graph.edata['time'].tolist()
        max_time, min_time, span = max(edges_time), min(edges_time), max(edges_time) - min(edges_time)
        temporal_subgraphs, nids, train_dataloader_list = [], [], []
 
        T, T_idx = sampling_layer(snapshots, views, span, strategy)

        for start in T:
            end = min(start + span / snapshots, max_time)
            start = max(start, min_time)
            sample_time = (inject_graph.edata['time'] >= start) & (inject_graph.edata['time'] <= end)

            temporal_subgraph = dgl.edge_subgraph(inject_graph, sample_time, relabel_nodes=False)

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
        diff_emb = th.tensor([]).to(device_id)
        for i_, train_dataloader in enumerate(train_dataloader_list):
            for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):           

                diff_blocks, diff_weights = [], []
                for i in range(len(blocks)):
                    diff_graph = dgl.node_subgraph(temporal_subgraphs[i_], blocks[i].srcdata[dgl.NID])
                    nx_g = dgl.to_networkx(diff_graph)

                    if diff == 'ppr':
                        diff_adj = compute_ppr(nx_g, alpha)
                    elif diff == 'heat':
                        diff_adj = compute_heat(nx_g)

                    mask_matrix = np.ones([diff_adj.shape[0], diff_adj.shape[1]])
      
                    for row in range(diff_adj.shape[0]):
                        for col in range(blocks[i].dstdata[dgl.NID].shape[0], diff_adj.shape[0]):
                            mask_matrix[row][col] = 0
                    diff_adj = diff_adj * mask_matrix

                    diff_edges = np.nonzero(diff_adj)
                    diff_weight = th.tensor(diff_adj[diff_edges], dtype=th.float).to(device_id)
                    diff_graph = dgl.graph(diff_edges)

                    block = dgl.to_block(diff_graph, diff_graph.nodes()[:blocks[i].dstdata[dgl.NID].shape[0]])
                    diff_blocks.append(block)
                    diff_weights.append(diff_weight)
  
                # forward
                batch_inputs = load_subtensor(inject_node_feat[T_idx[i_]], input_nodes, device_id)

                blocks = [block.to(device_id) for block in blocks]

                # metric and loss
                train_batch_logits = model(blocks, batch_inputs)
                seeds_emb = th.cat([seeds_emb, train_batch_logits.unsqueeze(0)], dim=0)
                

                diff_blocks = [block.to(device_id) for block in diff_blocks]
                train_batch_diff_logits = diff_model(diff_blocks, batch_inputs, diff_weights)
                diff_emb = th.cat([diff_emb, train_batch_diff_logits.unsqueeze(0)], dim=0)


        train_contrastive_loss = th.tensor([0]).to(device_id)
        for idx in range(seeds_emb.shape[0]):
            for idy in range(idx+1, seeds_emb.shape[0]):
                z1, z2, z3, z4 = seeds_emb[idx], seeds_emb[idy], diff_emb[idx], diff_emb[idy]
    
                pred1, pred2 = th.mm(z1, z2.T).to(device_id), th.mm(z2, z1.T).to(device_id)

                logits_pred1_max, _ = th.max(pred1, dim=1, keepdim=True)
                pred1 = pred1 - logits_pred1_max

                logits_pred2_max, _ = th.max(pred2, dim=1, keepdim=True)
                pred2 = pred2 - logits_pred2_max

                pred3, pred4 = th.mm(z1, z3.T).to(device_id), th.mm(z3, z1.T).to(device_id)
                
                logits_pred3_max, _ = th.max(pred3, dim=1, keepdim=True)
                pred3 = pred3 - logits_pred3_max

                logits_pred4_max, _ = th.max(pred4, dim=1, keepdim=True)
                pred4 = pred4 - logits_pred4_max

                pred5, pred6 = th.mm(z2, z4.T).to(device_id), th.mm(z4, z2.T).to(device_id)

                logits_pred5_max, _ = th.max(pred5, dim=1, keepdim=True)
                pred5 = pred5 - logits_pred5_max

                logits_pred6_max, _ = th.max(pred6, dim=1, keepdim=True)
                pred6 = pred6 - logits_pred6_max

                pred7, pred8 = th.mm(z3, z4.T).to(device_id), th.mm(z4, z3.T).to(device_id)

                logits_pred7_max, _ = th.max(pred7, dim=1, keepdim=True)
                pred7 = pred7 - logits_pred7_max

                logits_pred8_max, _ = th.max(pred8, dim=1, keepdim=True)
                pred8 = pred8 - logits_pred8_max

                labels = th.arange(pred1.shape[0]).to(device_id)

                loss_fn_ll = (loss_fn(pred1 / 0.07, labels) + loss_fn(pred2 / 0.07, labels)) / 2
                loss_fn_lg = (loss_fn(pred3 / 0.07, labels) + loss_fn(pred4 / 0.07, labels)) / 4 + (loss_fn(pred5 / 0.07, labels) + loss_fn(pred6 / 0.07, labels)) / 4
                loss_fn_gg = (loss_fn(pred7 / 0.07, labels) + loss_fn(pred8 / 0.07, labels)) / 4

                train_contrastive_loss = train_contrastive_loss + loss_fn_ll + loss_fn_lg + loss_fn_gg

        optimizer.zero_grad()
        train_contrastive_loss.backward()
        optimizer.step()

        if train_contrastive_loss < best_loss:
            best_loss = train_contrastive_loss
            best_model = copy.deepcopy(model)
        print("Epoch {:05d} | Loss {:.4f} ". format(epoch, float(train_contrastive_loss)))
        
        
    sampler = MultiLayerFullNeighborSampler(n_layers)
    temporal_subgraphs, test_dataloader_list = [], []
    T = [span * i / snapshots for i in range(snapshots)]
    for start in T:
        end = min(start + span / snapshots, max_time)
        start = max(start, min_time)
        sample_time = (inject_graph.edata['time'] >= start) & (inject_graph.edata['time'] <= end)
        temporal_subgraph = dgl.edge_subgraph(inject_graph, sample_time, relabel_nodes=False)

        temporal_subgraph = dgl.to_simple(temporal_subgraph)
        temporal_subgraph = dgl.to_bidirected(temporal_subgraph, copy_ndata=True)
        temporal_subgraph = dgl.add_self_loop(temporal_subgraph)
        nids.append(th.unique(temporal_subgraph.edges()[0]))
        temporal_subgraphs.append(temporal_subgraph)

    for sg_id in range(views):
        test_dataloader = NodeDataLoader(temporal_subgraphs[sg_id],
                                        temporal_subgraphs[sg_id].nodes(),
                                        sampler,
                                        batch_size=dataloader_size,
                                        shuffle=False,
                                        drop_last=False,
                                        num_workers=num_workers,
                                        )
        test_dataloader_list.append(test_dataloader)
                           
    best_model.eval()
    snapshots_embeddings = []
    for i_, test_dataloader in enumerate(test_dataloader_list):
        for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):
            batch_inputs = load_subtensor(inject_node_feat[i_], input_nodes, device_id)
            blocks = [block.to(device_id) for block in blocks]
            test_batch_logits = best_model(blocks, batch_inputs)

            batch_embeddings = test_batch_logits.detach()
            if step == 0:
                embeddings = batch_embeddings
            else:
                embeddings = th.cat((embeddings, batch_embeddings), axis=0)
        snapshots_embeddings.append(embeddings)
    
    ''' Discriminator '''
    ano_score = None

    for idx in range(len(snapshots_embeddings)):
        for idy in range(idx+1, len(snapshots_embeddings)):
            similarity = 1 - F.cosine_similarity(snapshots_embeddings[idx], snapshots_embeddings[idy], dim=1)

            if ano_score == None:
                ano_score = similarity.unsqueeze(1)
            else:
                ano_score = th.cat((ano_score, similarity.unsqueeze(1)), axis=1)

    ano_score = th.mean(ano_score, axis=1) + th.std(ano_score, axis=1)
    auc = roc_auc_score(anomaly_label, ano_score.cpu().numpy())
    print('AUC:{:.4f}'.format(auc))


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
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--threshold', type=float, default=1e-5)
    parser.add_argument('--diff', type=str, default='ppr')


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
    ALPHA = args.alpha
    THRESHOLD = args.threshold
    DIFF = args.diff

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
    batch_size=BATCH_SIZE, dataloader_size=DATALOADER_SIZE, alpha = ALPHA, threshold = THRESHOLD, diff = DIFF, num_workers=WORKERS, epochs=EPOCHS, GPU=GPU)


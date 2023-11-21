import os
import dgl
import random
import torch as th
import pandas as pd
import numpy as np
import math
from scipy.spatial.distance import euclidean

random.seed(24)


def load_to_dgl_graph(dataset, s):
    edges = pd.read_csv(os.path.join('../Data/', dataset, '{}.txt'.format(dataset)), sep=' ', names=['start_idx', 'end_idx', 'time'])
    
    src_nid = edges.start_idx.to_numpy()
    dst_nid = edges.end_idx.to_numpy()

    graph = dgl.graph((src_nid, dst_nid))
    graph.edata['time'] = th.Tensor(edges.time.tolist())

    node_feat = position_encoding(max_len=graph.num_nodes(), emb_size=128)

    # m: num of fully connected nodes
    # n: num of fully connected clusters
    # k: another 𝑘 nodes as a candidate set
    m, n, k = 15, 20, 50 

    if dataset == 'bitcoinotc' or dataset == 'bitotc' or dataset == 'bitalpha':
        n = 10
    elif dataset == 'dblp' or dataset == 'tax':
        n = 20
    elif dataset == 'tax51' or dataset == 'reddit':
        n = 200
    inject_graph, inject_node_feat, anomaly_label = inject_anomaly(graph, node_feat, m, n, k, s)

    return inject_graph, inject_node_feat, anomaly_label

def inject_anomaly(g, feat, m, n, k, s):
    num_node = g.num_nodes()
    all_idx = list(range(g.num_nodes()))
    random.shuffle(all_idx)
    anomaly_idx = all_idx[:m*n*2]

    structure_anomaly_idx = anomaly_idx[:m*n]
    attribute_anomaly_idx = anomaly_idx[m*n:]
    label = np.zeros((num_node,1),dtype=np.uint8)
    label[anomaly_idx,0] = 1

    str_anomaly_label = np.zeros((num_node,1),dtype=np.uint8)
    str_anomaly_label[structure_anomaly_idx,0] = 1
    attr_anomaly_label = np.zeros((num_node,1),dtype=np.uint8)
    attr_anomaly_label[attribute_anomaly_idx,0] = 1

    # Disturb structure
    print('Constructing structured anomaly nodes...')
    u_list, v_list, t_list = [], [], []
    max_time, min_time = max(g.edata['time'].tolist()), min(g.edata['time'].tolist())
    for n_ in range(n):
        current_nodes = structure_anomaly_idx[n_*m:(n_+1)*m]
        t = random.uniform(min_time, max_time)
        for i in current_nodes:
            for j in current_nodes:
               u_list.append(i)
               v_list.append(j)
               t_list.append(t)

    ori_num_edge = g.num_edges()
    g = dgl.add_edges(g, th.tensor(u_list), th.tensor(v_list), {'time': th.tensor(t_list)})

    num_add_edge = g.num_edges() - ori_num_edge
    print('Done. {:d} structured nodes are constructed. ({:.0f} edges are added) \n'.format(len(structure_anomaly_idx), num_add_edge))

    # Disturb attribute
    print('Constructing attributed anomaly nodes...')
    feat_list = []
    ori_feat = feat
    attribute_anomaly_idx_list = split_list(attribute_anomaly_idx, s)
    for lst in attribute_anomaly_idx_list:
        feat = ori_feat
        for i_ in lst:
            picked_list = random.sample(all_idx, k)
            max_dist = 0
            for j_ in picked_list:
                cur_dist = euclidean(ori_feat[i_], ori_feat[j_])
                if cur_dist > max_dist:
                    max_dist = cur_dist
                    max_idx = j_

            feat[i_] = feat[max_idx]
        feat_list.append(feat)
    print('Done. {:d} attributed nodes are constructed. \n'.format(len(attribute_anomaly_idx)))


    return g, feat_list, label

def dataloader(dataset):
    edges = pd.read_csv(os.path.join('../../Data/', dataset, '{}.txt'.format(dataset)), sep=' ', names=['start_idx', 'end_idx', 'time'])
    label = pd.read_csv(os.path.join('../../Data/', dataset, 'node2label.txt'), sep=' ', names=['nodeidx', 'label'])

    src_nid = edges.start_idx.to_numpy()
    dst_nid = edges.end_idx.to_numpy()

    graph = dgl.graph((src_nid, dst_nid))
    
    labels = th.full((graph.number_of_nodes(),), -1).cuda()

    nodeidx, lab = label.nodeidx.tolist(), label.label.tolist()

    for i in range(len(nodeidx)):
        labels[nodeidx[i]] = lab[i] - min(lab)

    train_mask = th.full((graph.number_of_nodes(),), False)
    val_mask = th.full((graph.number_of_nodes(),), False)
    test_mask = th.full((graph.number_of_nodes(),), False)

    random.seed(24)
    train_mask_index, val_mask_index, test_mask_index = th.LongTensor([]), th.LongTensor([]), th.LongTensor([])
    for i in range(min(labels), max(labels) + 1):
        index = [j for j in label[label.label==i].nodeidx.tolist()]
        random.shuffle(index)
        train_mask_index = th.cat((train_mask_index, th.LongTensor(index[:int(len(index) / 10)])), 0)
        val_mask_index = th.cat((val_mask_index, th.LongTensor(index[int(len(index) / 10):int(len(index) / 5)])), 0)
        test_mask_index = th.cat((test_mask_index, th.LongTensor(index[int(len(index) / 5):])), 0)

    train_mask.index_fill_(0, train_mask_index, True).cuda()
    val_mask.index_fill_(0, val_mask_index, True).cuda()
    test_mask.index_fill_(0, test_mask_index, True).cuda()
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
    n_classes = label.label.nunique()

    return labels, train_idx, val_idx, test_idx, n_classes

def position_encoding(max_len, emb_size):
    pe = th.zeros(max_len, emb_size)
    position = th.arange(0, max_len).unsqueeze(1)

    div_term = th.exp(th.arange(0, emb_size, 2) * -(math.log(10000.0) / emb_size))

    pe[:, 0::2] = th.sin(position * div_term)
    pe[:, 1::2] = th.cos(position * div_term)
    return pe

def split_list(lst, s):
    avg_length = len(lst) // s
    remainder = len(lst) % s
    result = [lst[i * avg_length + min(i, remainder):(i + 1) * avg_length + min(i + 1, remainder)] for i in range(s)]
    return result

def sampling_layer(snapshots, views, span, strategy):
    T = []
    if strategy == 'random':
        T = [random.uniform(0, span * (snapshots - 1) / snapshots) for _ in range(views)]
    elif strategy == 'low_overlap':
        if (0.75 * views + 0.25) > snapshots:
            return "The number of sampled views exceeds the maximum value of the current policy."
        start = random.uniform(0, span - (0.75 * views + 0.25) * span /  snapshots)
        T = [start + (0.75 * i * span) / snapshots for i in range(views)]
    elif strategy == 'high_overlap':
        if (0.25 * views + 0.75) > snapshots:
            return "The number of sampled views exceeds the maximum value of the current policy."
        start = random.uniform(0, span - (0.25 * views + 0.75) * span /  snapshots)
        T = [start + (0.25 * i * span) / snapshots for i in range(views)]
    elif strategy == 'sequential':
        T = [span * i / snapshots for i in range(snapshots)]
        ori_T = T
        if views > snapshots:
            return "The number of sampled views exceeds the maximum value of the current policy."
        T = random.sample(T, views)
        T_idx = [ori_T.index(i) for i in T]

    return T, T_idx

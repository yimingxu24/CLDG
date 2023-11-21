import torch as th
import torch.nn as thnn
import torch.nn.functional as F
import dgl.nn as dglnn


class LogReg(thnn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = thnn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret


class MLPLinear(thnn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPLinear, self).__init__()
        self.linear1 = thnn.Linear(in_dim, out_dim)
        self.linear2 = thnn.Linear(out_dim, out_dim)
        self.act = thnn.LeakyReLU(0.2)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, x):
        x = self.act(F.normalize(self.linear1(x), p=2, dim=1))
        x = self.act(F.normalize(self.linear2(x), p=2, dim=1))

        return x


class GraphConvModel(thnn.Module):

    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 norm,
                 activation,
                 readout,
                 dropout):
        super(GraphConvModel, self).__init__()
        self.in_feats = in_feats
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.norm = norm
        self.activation = activation
        self.readout = readout
        self.dropout = thnn.Dropout(dropout)

        self.layers = thnn.ModuleList()

        # build multiple layers
        self.layers.append(dglnn.GraphConv(in_feats=self.in_feats,
                                           out_feats=self.hidden_dim,
                                           norm=self.norm,
                                           activation=self.activation,
                                           allow_zero_in_degree=True))
        for l in range(1, (self.n_layers - 1)):
            self.layers.append(dglnn.GraphConv(in_feats=self.hidden_dim,
                                               out_feats=self.hidden_dim,
                                               norm=self.norm,
                                               activation=self.activation,
                                               allow_zero_in_degree=True))
        self.layers.append(dglnn.GraphConv(in_feats=self.hidden_dim,
                                           out_feats=self.n_classes,
                                           norm=self.norm,
                                           activation=self.activation))
        self.linear = thnn.Linear(self.n_classes, self.n_classes) 

        self.act = thnn.LeakyReLU(0.2)

    def forward(self, blocks, features, data_mask=None):
        h = features
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            if data_mask != None:
                h = layer(block, h, edge_weight=data_mask[l])
            else:
                h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.dropout(h)

        h = self.act(F.normalize(self.linear(h), p=2, dim=1))

        return h


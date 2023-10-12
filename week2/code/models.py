import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, trial):
        super(GCN, self).__init__()
        # parameters
        self.nhid = nhid
        self.dropout = dropout
        self.trial = trial
        # layers
        layers = []
        in_features = nfeat
        for i in range(nhid):
            out_features = self.trial.suggest_int(f'n_utils_l{i}', nclass, nfeat)
            layers.append(GCNConv(in_features, out_features, normalize=True))
            in_features = out_features
        layers.append(GCNConv(in_features, nclass, normalize=True))
        self.layers = nn.Sequential(*layers)
        

    def forward(self, x, adj):
        for i in range(self.nhid + 1):
            x = F.relu(self.layers[i](x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)
import time
import argparse
import numpy as np

import torch
import torch.optim as optim

# log
choice = int(input('请选择所使用数据集\n1 为 Cora, 2 为 Citeseer, 3 为 Pumbed:'))
path_name = None
dataset_name = None
import sys
sys.path.append('D:/Kuang/Code/Data/')
from log_file import *
if choice == 1:
    file_name = 'Cora_Dataset_Log.txt'
    path_name = '/tmp/Cora'
    dataset_name = 'Cora'
elif choice == 2:
    file_name = 'Citeseer_Dataset_Log.txt'
    path_name = './tmp/Citeseer/'
    dataset_name = 'Citeseer'
elif choice == 3:
    file_name = 'Pumbed_Dataset_Log.txt'
    path_name = '/tmp/Pubmed'
    dataset_name = 'Pubmed'
else: exit(114514)

# file = open_log_file(file_name)

# utils
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
dataset = Planetoid(root=path_name, name=dataset_name, transform=T.NormalizeFeatures())
data = dataset[0]
adj, features, labels = data.edge_index, data.x, data.y


# train
import optuna
import torch.nn.functional as F
from models import GCN

def objective(trial):
    # define
    n_hidden_layers = trial.suggest_int('n_hidden_layers', 0, 3)
    lr = trial.suggest_loguniform('lr', 10 ** (-2.5), 10 ** (-1.5))
    weight_decay = trial.suggest_loguniform('weight_decay', 10 ** (-2.5), 10 ** (-1.5))
    # model
    model = GCN(nfeat=features.shape[1],
                nhid=n_hidden_layers,
                nclass=labels.max().item() + 1,
                dropout=args.dropout,
                trial=trial)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[data.train_mask], labels[data.train_mask])
        # acc_train = accuracy(output[data.train_mask], labels[data.train_mask])
        loss_train.backward()
        optimizer.step()
    # return
    model.eval()
    return accuracy(output[data.val_mask], labels[data.val_mask])


study = optuna.create_study(direction='maximize', study_name=dataset_name, storage='sqlite:///db.sqlite3')
study.optimize(objective, n_trials=50)

# test
import plotly
print(study.best_params)
history = optuna.visualization.plot_optimization_history(study)
plotly.offline.plot(history)

# close_log_file(file)
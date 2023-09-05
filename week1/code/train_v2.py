from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

# from utils import load_data, accuracy
from models import GCN

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
file = open_log_file(file_name)

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
# adj, features, labels, idx_train, idx_val, idx_test = load_data(path_name, dataset_name)
# from torch.utils.data import DataLoader
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
dataset = Planetoid(root=path_name, name=dataset_name, transform=T.NormalizeFeatures())
data = dataset[0]
adj, features, labels = data.edge_index, data.x, data.y


# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    # idx_  train = idx_train.cuda()
    # idx_val = idx_val.cuda()
    # idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[data.train_mask], labels[data.train_mask])
    acc_train = accuracy(output[data.train_mask], labels[data.train_mask])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[data.val_mask], labels[data.val_mask])
    acc_val = accuracy(output[data.val_mask], labels[data.val_mask])
    log("[EPOCH: {:3d}/{:d}]".format(epoch + 1, args.epochs) + \
            "训练损失为: [{:.4f}]".format(loss_train.item()) + \
            "训练精度为: [{:.4f}]".format(acc_train) + \
            "验证损失为: [{:.4f}]".format(loss_val.item()) + \
            "验证精度为: [{:.4f}]".format(acc_val), file, True)


def test():
    model = torch.load('./Model/Cora_Model.pth')
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[data.test_mask], labels[data.test_mask])
    acc_test = accuracy(output[data.test_mask], labels[data.test_mask])
    log("\n\n测试损失为: [{:.4f}]".format(loss_test.item()) + \
            "测试精度为: [{:.4f}]".format(acc_test), file, True)


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()

close_log_file(file)
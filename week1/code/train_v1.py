from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
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
    path_name = '/tmp/Citeseer'
    dataset_name = 'Citeseer'
elif choice == 3:
    file_name = 'Pumbed_Dataset_Log.txt'
    path_name = '/tmp/Pubmed'
    dataset_name = 'Pubmed'
else: exit(114514)
file = open_log_file(file_name)


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
parser.add_argument('--hidden', type=int, default=16,
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
dataset = Planetoid(root=path_name, name=dataset_name)
data = dataset[0]   # ? just for a test
adj, features, labels = data.edge_index, data.x, data.y
idx_train = range(140)
idx_val = range(200, 500)
idx_test = range(500, 1500)
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

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
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    # print('Epoch: {:04d}'.format(epoch+1),
    #       'loss_train: {:.4f}'.format(loss_train.item()),
    #       'acc_train: {:.4f}'.format(acc_train.item()),
    #       'loss_val: {:.4f}'.format(loss_val.item()),
    #       'acc_val: {:.4f}'.format(acc_val.item()),
    #       'time: {:.4f}s'.format(time.time() - t))
    log("[EPOCH: {:3d}/{:d}]".format(epoch + 1, args.epochs) + \
            "训练损失为: [{:.4f}]".format(loss_train.item()) + \
            "训练精度为: [{:.4f}]".format(acc_train) + \
            "验证损失为: [{:.4f}]".format(loss_val.item()) + \
            "验证精度为: [{:.4f}]".format(acc_val), file, True)


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    # print("Test set results:",
    #       "loss= {:.4f}".format(loss_test.item()),
    #       "accuracy= {:.4f}".format(acc_test.item()))
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

torch.save(model, './Model/Cora_Model.pth')
close_log_file(file)
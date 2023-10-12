"""
Initializing a dataset is straightforward. 
An initialization of a dataset will automatically 
download its raw files and process them 
to the previously described Data format. E.g., 
to load the ENZYMES dataset 
(consisting of 600 graphs within 6 classes), type:
"""
import torch
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
print(dataset)  # Data(edge_index=[2, 168], x=[37, 3], y=[1])
print(len(dataset))
print(dataset.num_classes)
print(dataset.num_node_features, '\n')

"""
We now have access to all 600 graphs in the dataset:
"""

data = dataset[0]
print(data)
print(data.is_undirected(), '\n')

"""
我们可以看到数据集中的第一个图包含 37 个节点，
每个节点具有 3 个特征。有 168/2 = 84 条无向边，
并且该图仅分配给一个类。
此外，数据对象恰好包含一个图级目标。

We can even use slices, long or bool tensors 
to split the dataset. E.g., 
to create a 90/10 train/test split, type:
"""

train_dataset = dataset[:540]
print(train_dataset)

test_dataset = dataset[540:]
print(test_dataset, '\n')

"""
If you are unsure whether the dataset 
is already shuffled before you split, 
you can randomly permutate it by running:
"""

dataset = dataset.shuffle()
# equivalent of doing so
perm = torch.randperm(len(dataset))
dataset = dataset[perm]

"""
Let's try another one! Let's download Cora, 
the standard benchmark dataset 
for semi-supervised graph node classification:
"""

from torch_geometric.datasets import Planetoid

# headers = {'User-Agent:', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'}

dataset = Planetoid(root='/tmp/Cora', name='Cora')
print(dataset)
print(len(dataset))
print(dataset.num_classes)
print(dataset.num_node_features, '\n')

"""
BUG:(fixed)
raise RemoteDisconnected("Remote end closed connection without"
http.client.RemoteDisconnected: 
Remote end closed connection without response

Then Cora is not installed.

After 30 minutes, problem is solved by itself.
"""

# unprepared coding
data = dataset[0]
print(data)
print(data.is_undirected())
print(data.train_mask.sum().item()) # 140
print(data.val_mask.sum().item())   # 500
print(data.test_mask.sum().item())  # 1000

"""
This time, the Data objects holds a label for each node,
and additional node-level attributes: train_mask, val_mask and test_mask, where

train_mask denotes against which nodes to train (140 nodes),

val_mask denotes which nodes to use for validation, 
e.g., to perform early stopping (500 nodes),

test_mask denotes against which nodes to test (1000 nodes).
"""
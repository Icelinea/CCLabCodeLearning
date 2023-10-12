"""
We will use a simple GCN layer and 
replicate the experiments on the Cora citation dataset. 
For a high-level explanation on GCN, 
have a look at its blog post.

We first need to load the Cora dataset:
"""

from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')

"""
Note that we do not need to use transforms or a dataloader. 
Now let's implement a two-layer GCN:
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 32)
        self.conv2 = GCNConv(32, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

"""
构造函数定义了两个GCNConv层，
它们在我们网络的前向传递中被调用。
请注意，非线性未集成在调用中conv，
因此需要在之后应用（这在所有运营商中都是一致的）PyG). 
在这里，我们选择使用 ReLU 作为我们的中间非线性，
并最终输出类数的 softmax 分布。
让我们在训练节点上训练这个模型 200 个 epoches：
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

"""
Finally, we can evaluate our model on the test nodes:
"""

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')

"""
The easiest way to learn more about Graph Neural Networks
is to study the examples in the examples/
"""

"""
Exercises
1.What does edge_index.t().contiguous() do?

2.Load the "IMDB-BINARY" dataset from 
the TUDataset benchmark suite and randomly 
split it into 80%/10%/10% training, validation and test graphs.

3.What does each number of the following output mean?
print(batch)
>>> DataBatch(batch=[1082], 
    edge_index=[2, 4066], x=[1082, 21], y=[32])
"""
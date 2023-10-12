"""
GCN layer:

This formula can be divided into the following steps:
1.Add self-loops to the adjacency matrix.
2.Linearly transform node feature matrix.
3.Compute normalization coefficients.
4.Normalize node features in ().
5.Sum up neighboring node features ("add" aggregation).
6.Apply a final bias vector.

Steps 1-3 are typically computed before 
message passing takes place. 
Steps 4-5 can be easily processed 
using the MessagePassing base class. 
The full layer implementation is shown below:
"""

import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')    # "Add" aggregation (Step 5)
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.Tensor(out_channels))
        """
        Q: 为啥这里要独立的将 bias 设置出来而不直接放在 linear 层内
        A: 可能是要按照实现step顺序?
        """
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        # 没看懂代码原理
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out

    def message(self, x_j, norm):
        # x's shape [E, out_channels]
        # Step 4: Normalize Node Features
        return norm.view(-1, 1) * x_j

if __name__ == '__main__':
    """
    That is all that it takes to create a simple message passing layer.
    You can use this layer as a building block for deep architectures.
    Initializing and calling it is straightforward:
    """
    conv = GCNConv(16, 32)
    # x = conv(x, edge_index)
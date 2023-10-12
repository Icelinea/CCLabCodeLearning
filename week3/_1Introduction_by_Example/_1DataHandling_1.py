"""
PyG is described by an instance of torch_geometric.data.Data, which holds the following attributes by default:

data.x: Node feature matrix with shape [num_nodes, num_node_features]

data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long

data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]

data.y: Target to train against (may have arbitrary shape), e.g., 
node-level targets of shape [num_nodes, *] or graph-level targets of shape [1, *]

data.pos: Node position matrix with shape [num_nodes, num_dimensions]


We show a simple example of an unweighted and 
undirected graph with three nodes and four edges. 
Each node contains exactly one feature:
"""

import torch
from torch_geometric.data import Data

# 0->1  1->0  1->2  2->1  --  'four' edges
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)

x = torch.tensor([[-1], [0], [1]], dtype=torch.long)

data = Data(x=x, edge_index=edge_index)
print(data)
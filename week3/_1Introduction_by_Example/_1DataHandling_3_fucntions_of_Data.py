"""
Easy Function Usage

Note that it is necessary that the elements 
in edge_index only hold indices in the range 
{ 0, ..., num_nodes - 1}. 
This is needed as we want our final data representation 
to be as compact as possible, e.g., 
we want to index the source and 
destination node features of the first edge (0, 1) 
via x[0] and x[1], respectively. 
You can always check that your final Data objects 
fulfill these requirements by running validate():
"""

import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)

x = torch.tensor([[-1], [0], [1]], dtype=torch.long)

data = Data(x=x, edge_index=edge_index.t().contiguous())

print('validate:', data.validate(raise_on_error=True), '\n')

"""
Besides holding a number of node-level, 
edge-level or graph-level attributes, 
Data provides a number of useful utility functions, e.g.:
"""

print('keys:', data.keys)

print('index:', data['x'])

for key, item in data:
    print(f'{key} found in data')

print('\n', 'edge_attr' in data)

print('\n', data.num_nodes, data.num_edges, data.num_node_features)

print('\n', data.has_isolated_nodes(), data.has_self_loops(), data.is_directed())

# Transfer data object to GPU
device = torch.device('cuda')
# data = data.to(device)

"""
You can find a complete list of all methods at torch_geometric.data.Data.
"""
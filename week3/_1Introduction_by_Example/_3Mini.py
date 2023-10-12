"""
PyG contains its own torch_geometric.loader.DataLoader, 
which already takes care of this concatenation process. 
Let's learn about it in an example:
"""

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# for batch in loader:
#     print(batch)
#     print(batch.num_graphs)

print('\n')

"""
torch_geometric.data.Batch inherits from 
torch_geometric.data.Data and contains 
an additional attribute called batch.

batch is a column vector which 
maps each node to its respective graph in the batch:

You can use it to, e.g., average node features 
in the node dimension for each graph individually:
"""

from torch_geometric.utils import scatter
# from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

# dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
# loader = DataLoader(dataset, batch_size=32, shuffle=True)

for data in loader:
    print(data)
    print(data.num_graphs)

    x = scatter(data.x, data.batch, dim=0, reduce='mean')
    print(x.size(), '\n')
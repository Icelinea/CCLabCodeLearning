from torch_geometric.datasets import OGB_MAG

dataset = OGB_MAG(root='./tmp/OGB_MAG', preprocess='metapath2vec')
data = dataset[0]

# just unable to download the package
"""
Let's look at an example, 
where we apply transforms on the ShapeNet dataset 
(containing 17,000 3D shape point clouds and 
per point labels from 16 shape categories).
"""

from torch_geometric.datasets import ShapeNet

dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'])

print(dataset[0])

"""
We can convert the point cloud dataset 
into a graph dataset by generating nearest 
neighbor graphs from the point clouds via transforms:
(我们可以通过变换从点云生成最近邻图，将点云数据集转换为图数据集:)
"""

import torch_geometric.transforms as T
# from torch_geometric.datasets import ShapeNet

dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'], 
    pre_transform=T.KNNGraph(k=6))

print(dataset[0], '\n')

"""
In addition, we can use the transform argument to 
randomly augment a Data object, e.g.,
translating each node position by a small number:
(此外, 我们可以使用transform参数来随机扩充一个Data对象, 例如, 将每个节点位置平移一个小数:)
"""

# import torch_geometric.transforms as T
# from torch_geometric.datasets import ShapeNet

dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'],
    pre_transform=T.KNNGraph(k=6),
    transform=T.RandomJitter(0.01))

print(dataset[0], '\n')

"""
在将数据保存到磁盘之前，
我们使用pre_transform来转换数据（从而加快加载时间）。
请注意，下一次初始化数据集时，它已经包含图边，即使您没有通过任何转换。
如果pre_transform与已处理数据集中的不匹配，您将收到警告。

You can find a complete list of 
all implemented transforms at torch_geometric.transforms.
"""
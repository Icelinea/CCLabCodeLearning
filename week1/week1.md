## Week1
### 简要介绍
相对于[原 pygcn 项目](https://github.com/tkipf/pygcn)，主要代码和模块为 ```train_v2``` 和 ```models```，且仅将 ```utils``` 模块中的函数 ```accuracy``` 并入 ```train_v2``` 代码，以及日志辅助文件 ```log_file```；
最终版本 ```train_v2``` 对三个数据集的运行结果在 ```week1/log``` 路径下；
### 学习过程与收获
1. 理解论文思想并尝试性复现了论文中使用正则化的 GCN 模型代码
2. 简单研究原项目中所使用模块与目前 pytorch 模块的实现区别，例如原模块 ```GraphConvolution``` 与 ```GCNConv``` 中基类 ```MessagePassing``` 对后者的加速作用
### 疑问与待解决 Issue
1. **模块 ```Planetoid``` 所下载的数据集与原项目的数据集区别**  
   ```Planetoid``` 下的数据集会在原 ```models``` 使用的卷积层 ```GraphConvolution``` 中出现维度报错；
   由于不理解 ```Planetoid``` 下的数据集 ```data.pt``` 形式与原项目的数据集 ```cora.cites``` 和 ```cora.content``` 形式两者之间组成方式的区别，因此将 ```models``` 中的卷积层改为 ```torch_geometric.nn``` 中的 ```GCNConv``` 层；
---
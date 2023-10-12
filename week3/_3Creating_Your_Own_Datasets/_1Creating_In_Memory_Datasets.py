"""
In order to create a torch_geometric.data.InMemoryDataset,
you need to implement four fundamental methods:
InMemoryDataset.raw_file_names(): 
    A list of files in the raw_dir which needs to be found in order to skip the download.
InMemoryDataset.processed_file_names(): 
    A list of files in the processed_dir which needs to be found in order to skip the processing.
InMemoryDataset.download(): 
    Downloads raw data into raw_dir.
InMemoryDataset.process(): 
    Processes raw data and saves it into the processed_dir.
"""

import torch
from torch_geometric.data import InMemoryDataset, download_url

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to 'self.raw_dir'
        # download_url(url, self.raw_dir)
        # ...
        1

    def process(self):
        # Read data to huge 'Data' list
        data_list = [...]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
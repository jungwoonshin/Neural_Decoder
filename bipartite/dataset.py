import torch
from torch_geometric.data import InMemoryDataset


class MyOwnDataset(InMemoryDataset):
    def __init__(self,data,dataset_name, root=None, transform=None, pre_transform=None):
        self.data = data
        self.root = '/data/' +dataset_name
        super(MyOwnDataset, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data']

    @property
    def processed_file_names(self):
        return ['data.pt']

    # def download(self):
    #     pass
    #     # Download to `self.raw_dir`.

    def process(self):
        # pass
        # Read data into huge `Data` list.
        data_list = [self.data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
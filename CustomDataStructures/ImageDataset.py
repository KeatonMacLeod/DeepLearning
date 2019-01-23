import torch
from torch.utils import data


class ImageDataset(data.Dataset):
    def __init__(self, list_IDs, labels, data_directory):
        self.labels = labels
        self.list_IDs = list_IDs
        self.data_directory = data_directory

    # Returns the number of samples
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load(self.data_directory + "\\" + ID)
        y = self.labels[ID]

        return X, y

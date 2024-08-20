import torch
from torch.utils.data import Dataset


class SoloData(Dataset):
    def __init__(self, xy):
        self.x = torch.from_numpy(xy[:, 6:])
        self.y = torch.from_numpy(xy[:, :6])
        self.n_samples = xy.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

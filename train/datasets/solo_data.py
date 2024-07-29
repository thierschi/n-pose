import numpy as np
import torch
from torch.utils.data import Dataset


class SoloData(Dataset):
    def __init__(self, path: str):
        xy = np.loadtxt(path, delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 8:])
        self.y = torch.from_numpy(xy[:, 1:8])
        self.n_samples = xy.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

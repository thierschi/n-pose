import numpy as np
import torch
from torch.utils.data import Dataset


class SoloData(Dataset):
    def __init__(self, xy):
        x = xy[:, 3:]
        kp_l = x[:, 0: 10]
        kp_r = x[:, 110: 120]

        joined = np.concatenate((kp_l, kp_r), axis=1)

        self.x = torch.from_numpy(joined)

        self.y = torch.from_numpy(xy[:, 0:3])
        self.n_samples = xy.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

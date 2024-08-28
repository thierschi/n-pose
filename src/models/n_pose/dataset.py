import numpy as np
import torch
from torch.utils.data import Dataset


class SoloData(Dataset):
    def __init__(self, xy, add_direction=False):
        self.x = torch.from_numpy(xy[:, 6:])

        y = xy[:, :6]
        if add_direction:
            new_y = []
            for i in range(y.shape[0]):
                p1 = y[i, :3]
                p2 = p1 + y[i, 3:]
                new_row = [p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]]
                new_y.append(new_row)
            y = np.array(new_y)

        self.y = torch.from_numpy(y)
        # self.y = torch.from_numpy(xy[:, :6])
        self.n_samples = xy.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

import torch
import torch.nn as nn


class EuclideanLoss(nn.Module):
    """
    Loss function that computes the Euclidean distance between two tensors.
    """

    def __init__(self):
        super(EuclideanLoss, self).__init__()

    def forward(self, y_pred, y_true):
        euclidean_loss = torch.sqrt(torch.sum((y_pred - y_true) ** 2, dim=-1))
        return torch.mean(euclidean_loss)

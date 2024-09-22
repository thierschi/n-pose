import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    """
    Loss function that computes the Root Mean Squared Error (RMSE) between two tensors.
    """

    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        mse_loss = nn.MSELoss()(y_pred, y_true)
        rmse_loss = torch.sqrt(mse_loss)
        return rmse_loss

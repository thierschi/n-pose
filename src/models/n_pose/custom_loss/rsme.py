import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        mse_loss = nn.MSELoss()(y_pred, y_true)
        rmse_loss = torch.sqrt(mse_loss)
        return rmse_loss

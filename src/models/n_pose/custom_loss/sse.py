import torch
import torch.nn as nn


class SSELoss(nn.Module):
    def __init__(self):
        super(SSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        sse_loss = torch.sum((y_pred - y_true) ** 2)
        return sse_loss

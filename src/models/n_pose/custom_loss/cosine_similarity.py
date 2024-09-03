import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, y_pred, y_true):
        cosine_similarity = F.cosine_similarity(y_pred, y_true, dim=-1)
        loss = 1 - cosine_similarity.mean()
        return loss

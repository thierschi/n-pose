import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarityLoss(nn.Module):
    """
    Loss function that computes the cosine similarity between two tensors.
    """

    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, y_pred, y_true):
        cosine_similarity = F.cosine_similarity(y_pred, y_true, dim=-1)
        # Subtract from 1 to get a loss between 0 and 2 where 0 is the best
        loss = 1 - cosine_similarity.mean()
        return loss

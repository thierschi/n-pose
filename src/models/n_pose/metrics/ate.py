import torch


class ATE:
    """
    Average Translation Error (ATE) metric.
    """
    total_error: float
    count: int

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_error = 0.0
        self.count = 0

    def update(self, predicted, ground_truth):
        # Consider only the first three elements
        predicted = predicted[:, :3]
        ground_truth = ground_truth[:, :3]
        error = torch.norm(predicted - ground_truth, dim=1).sum().item()
        self.total_error += error
        self.count += predicted.size(0)

    def compute(self) -> float:
        """
        Compute the Average Translation Error (ATE) over all previously added samples.
        :return: The ATE value.
        """
        if self.count == 0:
            return 0.0
        return self.total_error / self.count

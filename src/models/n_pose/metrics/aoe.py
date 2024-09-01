import torch


class AOE:
    """
    Average Orientation Error (AOE) metric.
    """
    total_error: float
    count: int

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_error = 0.0
        self.count = 0

    def update(self, predicted, ground_truth):
        # Consider only the last three elements
        predicted = predicted[:, -3:]
        ground_truth = ground_truth[:, -3:]

        # Normalize the vectors
        predicted = predicted / torch.norm(predicted, dim=1, keepdim=True)
        ground_truth = ground_truth / torch.norm(ground_truth, dim=1, keepdim=True)

        # Calculate the angular error
        dot_product = (predicted * ground_truth).sum(dim=1)
        angular_error = torch.acos(torch.clamp(dot_product, -1.0, 1.0))

        self.total_error += angular_error.sum().item()
        self.count += predicted.size(0)

    def compute(self):
        if self.count == 0:
            return 0.0
        return self.total_error / self.count

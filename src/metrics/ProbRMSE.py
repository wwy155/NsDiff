import torch
from torchmetrics import Metric

class ProbRMSE(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total_mse", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, true: torch.Tensor):
        """
        Args:
            pred: Tensor of predicted distributions, shape (B, O, N, S).
            true: Tensor of true values, shape (B, O, N).
        """
        # Compute mean along S-axis
        pred_mean = pred.mean(dim=-1)  # Shape: (B, O, N)

        # Ensure the true tensor matches the shape
        assert true.shape == pred_mean.shape, "Shapes of true values and pred_mean must match"

        # Compute squared error
        squared_error = (pred_mean - true) ** 2

        # Sum errors and count total samples
        self.total_mse += squared_error.sum()
        self.total_samples += squared_error.numel()

    def compute(self):
        # Compute root mean squared error
        return torch.sqrt(self.total_mse / self.total_samples)

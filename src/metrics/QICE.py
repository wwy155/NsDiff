import torch
from torchmetrics import Metric
import numpy as np

class QICE(Metric):
    def __init__(self, n_bins: int = 10, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.n_bins = n_bins
        # Add states for each quantile's coverage ratio
        self.add_state("quantile_bin_counts", default=torch.zeros(self.n_bins), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update the metric with the predictions and targets.
        Args:
            preds: Tensor of shape (N, S) containing generated predictions
            targets: Tensor of shape (N, 1) containing ground truth values
        """
        
        preds = preds.view(-1, preds.size(3))  # Reshape to (B * O * N, S)
        targets = targets.view(-1)  # Reshape to (B * O * N,)

        preds_np = preds.cpu().numpy()  # Shape (N, S)
        targets_np = targets.cpu().numpy().T  # Shape (1, N)
        
        # Generate quantiles based on the number of bins
        quantile_list = np.arange(self.n_bins + 1) * (100 / self.n_bins)
        
        # Calculate the quantiles for the predicted values
        y_pred_quantiles = np.percentile(preds_np, q=quantile_list, axis=1)  # Shape (n_bins+1, N)
        
        # Calculate which quantile interval the true target belongs to
        quantile_membership_array = ((targets_np - y_pred_quantiles) > 0).astype(int)  # Shape (n_bins+1, N)
        y_true_quantile_membership = quantile_membership_array.sum(axis=0)  # Shape (N,)
        
        # Count the number of targets in each bin
        y_true_quantile_bin_count = np.array(
            [(y_true_quantile_membership == v).sum() for v in np.arange(self.n_bins + 2)]  # Shape (n_bins+2,)
        )
        
        # Combine outliers into the first and last bins
        y_true_quantile_bin_count[1] += y_true_quantile_bin_count[0]
        y_true_quantile_bin_count[-2] += y_true_quantile_bin_count[-1]
        y_true_quantile_bin_count_ = y_true_quantile_bin_count[1:-1]  # Exclude first and last bin
        
        # Update the quantile bin counts for each update
        self.quantile_bin_counts += torch.tensor(y_true_quantile_bin_count_).to(self.device)
        self.total_samples += preds.size(0)
        
    def compute(self):
        """
        Compute the QICE score (geometric mean of coverage ratios).
        Returns:
            The QICE score as a float.
        """
        # Normalize the counts by the total number of samples
        
        y_true_ratio_by_bin = self.quantile_bin_counts.float() / self.total_samples.item()
        print(y_true_ratio_by_bin.shape, torch.sum(y_true_ratio_by_bin),  torch.abs(
            torch.sum(y_true_ratio_by_bin) - 1))
        assert torch.abs(
            torch.sum(y_true_ratio_by_bin) - 1) < 1e-5, "Sum of quantile coverage ratios shall be 1!"
        qice_coverage_ratio = torch.abs(torch.ones(self.n_bins) / self.n_bins - y_true_ratio_by_bin).mean()
        return qice_coverage_ratio

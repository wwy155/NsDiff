# import torch
# from torchmetrics import Metric

# class PICP(Metric):
#     def __init__(self, low_percentile: int = 5, high_percentile: int = 95, dist_sync_on_step=False):
#         super().__init__(dist_sync_on_step=dist_sync_on_step)
#         self.low_percentile = low_percentile
#         self.high_percentile = high_percentile
#         self.add_state("coverage", default=torch.tensor(0.0), dist_reduce_fx="sum")
#         self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

#     def update(self, all_gen_y: torch.Tensor, y_true: torch.Tensor ):
#         all_gen_y = all_gen_y.view(-1, all_gen_y.size(3))  # Reshape to (B * O * N, S)
#         y_true = y_true.view(-1)  # Reshape to (B * O * N,)

#         low, high = self.low_percentile, self.high_percentile
#         CI_y_pred = torch.percentile(all_gen_y.squeeze(), q=torch.tensor([low, high]).float(), dim=1)
#         y_in_range = (y_true >= CI_y_pred[0]) & (y_true <= CI_y_pred[1])
        
        
#         coverage = y_in_range.float().mean()
#         self.coverage += coverage
#         self.total_samples += y_true.size(0)

#     def compute(self):
#         return self.coverage / self.total_samples


import torch
from torchmetrics import Metric

class PICP(Metric):
    def __init__(self, low_percentile: int = 5, high_percentile: int = 95, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.add_state("coverage", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, all_gen_y: torch.Tensor, y_true: torch.Tensor):
        # Reshape to (B * O * N, S)
        all_gen_y = all_gen_y.view(-1, all_gen_y.size(3)).cpu()
        y_true = y_true.view(-1).cpu()  # Reshape to (B * O * N,)

        # Compute the low and high percentiles using torch.quantile
        low, high = self.low_percentile, self.high_percentile
        CI_y_pred = torch.quantile(all_gen_y, torch.tensor([low / 100.0, high / 100.0]).float(), dim=1)
        
        # Determine whether the true values are within the prediction intervals
        y_in_range = (y_true >= CI_y_pred[0]) & (y_true <= CI_y_pred[1])
        
        coverage = y_in_range.float().mean()
        self.coverage += coverage.to(self.device)
        self.total_samples += y_true.size(0)

    def compute(self):
        return self.coverage / self.total_samples

import torch
from torchmetrics import Metric
import CRPS.CRPS as pscore  # Assuming `pscore` is the function to compute CRPS

class CRPSSum(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total_crps", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, true: torch.Tensor):
        """
        Args:
            pred: Tensor of predicted distributions, shape (N, S).
            true: Tensor of true values, shape (N,).
        """
        
        pred = pred.sum(dim=2)
        true = true.sum(dim=2)
        

        pred = pred.view(-1, pred.size(2))  # Reshape to (B * O , S)
        true = true.view(-1)  # Reshape to (B * O,)

        
        pred_np = pred.cpu().numpy()
        true_np = true.cpu().numpy()

        crps_sum = 0.0
        for i in range(len(true_np)):
            res = pscore(pred_np[i], true_np[i]).compute()
            crps_sum += res[0]

        self.total_crps += torch.tensor(crps_sum).to(self.device)
        self.total_samples += pred.size(0)

    def compute(self):
        return self.total_crps / self.total_samples

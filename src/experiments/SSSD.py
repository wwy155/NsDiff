from dataclasses import dataclass, field
import sys
from typing import List, Dict
import os

import torch
from dataclasses import dataclass, asdict, field
from torch_timeseries.nn.embedding import freq_map
from src.models.SSSD import SSSDSAImputer, calc_diffusion_hyperparams
from src.experiments.prob_forecast import ProbForecastExp
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from tqdm import tqdm
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible
import time
# import multiprocessing
import torch.multiprocessing as mp

import numpy as np
import torch.distributed as dist
import torch
from tqdm import tqdm
import concurrent.futures


def std_normal(size):
    """
    Generate the standard Gaussian variable of a certain size
    """

    return torch.normal(0, 1, size=size).cuda()


@dataclass
class SSSDParameters:
    beta_start: float =  0.0001
    beta_end: float =  0.5
    num_steps: int =  20
    num_samples: int =  100
   
    d_model: int=128 
    n_layers: int=6 
    pool: List[int] = field(default_factory=lambda : [2, 2])
    expand: int=2 
    ff: int= 2 
    glu: bool=True
    unet: bool=True 
    dropout: float =0.1
    in_channels: int=1
    out_channels: int=1
    diffusion_step_embed_dim_in: int=128 
    diffusion_step_embed_dim_mid: int=512
    diffusion_step_embed_dim_out: int=512
    label_embed_dim: int = 128
    label_embed_classes: int = 71
    bidirectional: bool=True
    s4_lmax: int=1
    s4_d_state: int=64
    s4_dropout: float=0.0
    s4_bidirectional: bool=True


@dataclass
class SSSDSForecast(ProbForecastExp, SSSDParameters):
    model_type: str = "SSSD"
    def _init_model(self):
        self.model = SSSDSAImputer(
            d_model=self.d_model,
            n_layers=self.n_layers,
            pool=self.pool,
            expand=self.expand,
            ff=self.ff,
            glu=self.glu,
            unet=self.unet,
            dropout=self.dropout,
            in_channels=self.dataset.num_features,
            out_channels=self.dataset.num_features,
            diffusion_step_embed_dim_in=self.diffusion_step_embed_dim_in, 
            diffusion_step_embed_dim_mid=self.diffusion_step_embed_dim_mid,
            diffusion_step_embed_dim_out=self.diffusion_step_embed_dim_out,
            label_embed_dim=self.label_embed_dim,
            label_embed_classes=self.label_embed_classes,
            bidirectional=self.bidirectional,
            s4_lmax=self.s4_lmax,
            s4_d_state=self.s4_d_state,
            s4_dropout=self.s4_dropout,
            s4_bidirectional=self.s4_bidirectional,

        )
        self.model = self.model.to(self.device)
        
        self.diffu_params = calc_diffusion_hyperparams(self.num_steps, self.beta_start, self.beta_end)
        
        self.gt_mask = torch.concat([
                torch.ones(size=(self.windows, self.dataset.num_features)),
                torch.zeros(size=(self.pred_len, self.dataset.num_features)),
            ]).to(self.device).bool()
        
        self.observation_mask = ~self.gt_mask


    def _process_train_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        # pip install pykeops==1.5 
        X = (
            torch.concat([batch_x, batch_y], dim=1).to(self.device).transpose(1,2).float(),
            torch.concat([batch_x, batch_y], dim=1).to(self.device).transpose(1,2).float(),
            self.gt_mask.unsqueeze(0).expand(batch_x.shape[0], -1, -1).transpose(1,2).to(self.device),
            self.observation_mask.unsqueeze(0).expand(batch_x.shape[0], -1, -1).transpose(1,2).to(self.device),
            # torch.tensor(self.num_steps).unsqueeze(0).expand(batch_x.shape[0], -1).to(self.device).long(),
            # torch.concat([batch_x_date_enc, batch_y_date_enc], dim=1)[:, :, 0],
        )
        
        T, Alpha_bar = self.num_steps, self.diffu_params["Alpha_bar"].to(self.device)

        audio = X[0]
        cond = X[1]
        mask = X[2]
        loss_mask = X[3]

        B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
        diffusion_steps = torch.randint(T, size=(B, 1, 1)).to(self.device).long()  # randomly sample diffusion steps from 1~T

        z = std_normal(audio.shape)
        # if only_generate_missing == 1:
        #     z = audio * mask.float() + z * (1 - mask).float()
        transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * audio + torch.sqrt(
            1 - Alpha_bar[diffusion_steps]) * z  # compute x_t from q(x_t|x_0)
        epsilon_theta = self.model(
            (transformed_X, cond, mask, diffusion_steps.view(B, 1),))  # predict \epsilon according to \epsilon_\theta



        # if only_generate_missing == 1:
        # return loss_fn(epsilon_theta[loss_mask], z[loss_mask])
        # elif only_generate_missing == 0:
        #     return loss_fn(epsilon_theta, z)

        return epsilon_theta[loss_mask], z[loss_mask]
        
        # pred = self.model(batch_input).transpose(1,2) # B, L+O,N
        # return pred[:, -self.pred_len:, :], batch_y


    def _process_val_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        X = (
            torch.concat([batch_x, batch_y], dim=1).to(self.device).transpose(1,2).float(),
            torch.concat([batch_x, batch_y], dim=1).to(self.device).transpose(1,2).float(),
            self.gt_mask.unsqueeze(0).expand(batch_x.shape[0], -1, -1).transpose(1,2).to(self.device),
            self.observation_mask.unsqueeze(0).expand(batch_x.shape[0], -1, -1).transpose(1,2).to(self.device),
            # torch.tensor(self.num_steps).unsqueeze(0).expand(batch_x.shape[0], -1).to(self.device).long(),
            # torch.concat([batch_x_date_enc, batch_y_date_enc], dim=1)[:, :, 0],
        )

        bs = X[0].shape[0]
        T, Alpha, Alpha_bar, Sigma = self.diffu_params["T"], self.diffu_params["Alpha"], self.diffu_params["Alpha_bar"], self.diffu_params["Sigma"]
        size = (bs*self.num_samples,X[0].shape[1] , X[0].shape[2] )
        X = [x.repeat(self.num_samples, 1, 1) for x in X]
        
        cond = X[1]
        mask = X[2]

        x = std_normal(size)

        with torch.no_grad():
            for t in range(T - 1, -1, -1):
                # if only_generate_missing == 1:
                #     x = x * (1 - mask).float() + cond * mask.float()
                diffusion_steps = (t * torch.ones((size[0], 1))).cuda()  # use the corresponding reverse step
                epsilon_theta = self.model((x, cond, mask, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta
                # update x_{t-1} to \mu_\theta(x_t)
                x = (x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
                if t > 0:
                    x = x + Sigma[t] * std_normal(size)  # add the variance term to x_{t-1}
        x = x.reshape(bs, self.num_samples, X[0].shape[1] , X[0].shape[2]).permute(0, 3, 2, 1)
        return x[:, -self.pred_len:, :, :], batch_y



if __name__ == "__main__":
    import fire
    # torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    fire.Fire(SSSDSForecast)
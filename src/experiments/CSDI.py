from dataclasses import dataclass, field
import sys
from typing import List, Dict
import os

import torch
from dataclasses import dataclass, asdict, field
from torch_timeseries.nn.embedding import freq_map
from src.models.CSDI import CSDI_Forecasting
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



@dataclass
class CSDIParameters:
    layers:  int = 2  # 4
    channels: int =  64 
    nheads:  int = 8
    diffusion_embedding_dim: int =  128
    beta_start: float =  0.0001
    beta_end: float =  0.5
    num_steps: int =  20
    schedule:  str = "quad"
    is_linear: bool =  True
    is_unconditional: int = 0
    timeemb: int =  64
    featureemb: int =  16
    num_samples: int =  100
    target_strategy: str =  "test"
    # num_sample_features: int =  64


@dataclass
class CSDIForecast(ProbForecastExp, CSDIParameters):
    model_type: str = "CSDI"
    def _init_model(self):
        
        configs = {
            "diffusion": {
                "layers": self.layers,
                "channels": self.channels,
                "nheads": self.nheads,
                "diffusion_embedding_dim": self.diffusion_embedding_dim,
                "beta_start": self.beta_start,
                "beta_end": self.beta_end,
                "num_steps": self.num_steps,
                "schedule": self.schedule,
                "is_linear": self.is_linear,
            },
            "model":{
                "is_unconditional": self.is_unconditional,
                "timeemb": self.timeemb,
                "featureemb": self.featureemb,
                "target_strategy": self.target_strategy,
                "num_sample_features": self.dataset.num_features, #  use all features to predict
            }
        }
        self.model = CSDI_Forecasting(
            config=configs,
            device=self.device,
            target_dim=self.dataset.num_features
            )
        self.model = self.model.to(self.device)
        
        
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
        
        batch_input = {
            "observed_data": torch.concat([batch_x, batch_y], dim=1),
            "observed_mask": self.observation_mask.unsqueeze(0).expand(batch_x.shape[0], -1, -1),
            "timepoints": torch.concat([batch_x_date_enc, batch_y_date_enc], dim=1)[:, :, 0],
            "gt_mask": self.gt_mask.unsqueeze(0).expand(batch_x.shape[0], -1, -1),
        }
        
        noise, pred_noise = self.model(batch_input, is_train=self.num_samples)
        return noise, pred_noise


    def _process_val_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        
        batch_input = {
            "observed_data": torch.concat([batch_x, batch_y], dim=1),
            "observed_mask": self.observation_mask.unsqueeze(0).expand(batch_x.shape[0], -1, -1),
            "timepoints": torch.concat([batch_x_date_enc, batch_y_date_enc], dim=1)[:, :, 0],
            "gt_mask": self.gt_mask.unsqueeze(0).expand(batch_x.shape[0], -1, -1),
        }
        
        batch_x = batch_x.to(self.device).float()
        batch_y = batch_y.to(self.device).float()
        batch_x_date_enc = batch_x_date_enc.to(self.device).float()
        batch_y_date_enc = batch_y_date_enc.to(self.device).float()

        
        samples, observed_data, target_mask, observed_mask, observed_tp = self.model.evaluate(batch_input, self.num_samples, 5)
        samples = samples[:, :, :, -self.pred_len:]
        return samples[:, :, :, -self.pred_len:].permute(0, 3, 2, 1), batch_y


if __name__ == "__main__":
    import fire
    # torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    fire.Fire(CSDIForecast)
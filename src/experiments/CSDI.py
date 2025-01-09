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
import wandb


@dataclass
class CSDIParameters:
    layers:  int = 4  # 4
    channels: int =  64 
    nheads:  int = 8
    diffusion_embedding_dim: int =  128
    beta_start: float =  0.0001
    beta_end: float =  0.5
    num_steps: int =  50
    schedule:  str = "quad"
    is_linear: bool =  True
    is_unconditional: int = 0
    timeemb: int =  128
    featureemb: int =  16
    num_samples: int =  100
    target_strategy: str =  "test"
    num_sample_features: int =  64


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
                "num_sample_features": self.num_sample_features #self.dataset.num_features, #  use all features to predict
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

        
        samples, observed_data, target_mask, observed_mask, observed_tp = self.model.evaluate(batch_input, self.num_samples, 10)
        samples = samples[:, :, :, -self.pred_len:]
        return samples[:, :, :, -self.pred_len:].permute(0, 3, 2, 1), batch_y


    def run(self, seed=42) -> Dict[str, float]:
        
        if self._use_wandb() and not self._init_wandb(self.project, seed): return {}
        
        self._setup_run(seed)
        if self._check_run_exist(seed):
            self._resume_run(seed)

        self._run_print(f"run : {self.current_run} in seed: {seed}")

        parameter_tables, model_parameters_num = count_parameters(self.model)
        self._run_print(f"parameter_tables: {parameter_tables}")
        self._run_print(f"model parameters: {model_parameters_num}")

        if self._use_wandb():
            wandb.run.summary["parameters"] = model_parameters_num

        # for resumable reproducibility_
        while self.current_epoch < self.epochs:
            epoch_start_time = time.time()
            if self.early_stopper.early_stop is True:
                self._run_print(
                    f"val loss no decreased for patience={self.patience} epochs,  early stopping ...."
                )
                break

            # for resumable reproducibility
            reproducible(seed + self.current_epoch)
            train_losses = self._train()
            self._run_print(
                "Epoch: {} cost time: {}s".format(
                    self.current_epoch + 1, time.time() - epoch_start_time
                )
            )
            self._run_print(f"Traininng loss : {np.mean(train_losses)}")

            val_result = self._val()
            test_result = self._test()

            self.current_epoch = self.current_epoch + 1
            self.early_stopper(val_result['crps'], model=self.model)

            self._save_run_check_point(seed)

            if self._use_wandb():
                wandb.log({'training_loss' : np.mean(train_losses)}, step=self.current_epoch)
                wandb.log( {f"val_{k}": v for k, v in val_result.items()}, step=self.current_epoch)
                # wandb.log( {f"test_{k}": v for k, v in test_result.items()}, step=self.current_epoch)

            # self.scheduler.step()

        self._load_best_model()
        best_test_result = self._test()
        if self._use_wandb():
            for k, v in best_test_result.items(): wandb.run.summary[f"best_test_{k}"] = v 
        
        if self._use_wandb():  wandb.finish()
        return best_test_result
    
    
if __name__ == "__main__":
    import fire
    # torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    fire.Fire(CSDIForecast)
from dataclasses import dataclass, field
import sys
from typing import List, Dict
import os

import torch
from src.experiments.forecast import ForecastExp 
from dataclasses import dataclass, asdict, field
from torch_timeseries.nn.embedding import freq_map
from src.models.D3VAE import D3VAE
from src.experiments.prob_forecast import ProbForecastExp
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from tqdm import tqdm
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible
import time
# import multiprocessing
from torch_timeseries.dataset import *
from torch_timeseries.scaler import *

import torch.multiprocessing as mp
from torch_timeseries.utils.parse_type import parse_type
from torch_timeseries.utils.reproduce import reproducible
from torch_timeseries.core import TimeSeriesDataset, BaseIrrelevant, BaseRelevant
from torch_timeseries.dataloader import SlidingWindowTS

import numpy as np
import torch.distributed as dist
import torch
from tqdm import tqdm
import concurrent.futures



@dataclass
class D3VAEParameters:
    embedding_dimension: int = 64
    dropout_rate: int = 0.1
    beta_schedule: str = 'linear'
    beta_start: float = 0.0
    beta_end: int = 0.1
    diff_steps: int = 50
    scale: int = 0.1
    mult: int = 1
    channel_mult: int = 2
    num_preprocess_blocks: int = 1
    num_preprocess_cells: int = 3
    num_channels_enc: int = 32 #
    arch_instance: str = 'res_mbconv'
    num_latent_per_group: int = 8
    num_channels_dec : int = 16 # default 32 but OOM 
    num_postprocess_blocks: int = 1
    num_postprocess_cells: int = 2
    hidden_size: int = 128 # default 128 but OOM 
    num_layers: int = 2
    groups_per_scale: int = 2
    psi : float = 0.5
    lambda1 : float = 1
    gamma : float = 0.01

@dataclass
class D3VAEForecast(ProbForecastExp, D3VAEParameters):
    model_type: str = "D3VAE"
    def _init_model(self):
        seq_len = max(self.pred_len, self.windows)
        self.model = D3VAE(
            input_dim=self.dataset.num_features,
            embedding_dimension=self.embedding_dimension,
            freq=self.dataset.freq,#freq_map[self.dataset.freq],
            dropout_rate=self.dropout_rate,
            beta_schedule=self.beta_schedule,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            diff_steps=self.diff_steps,
            sequence_length=seq_len,
            scale=self.scale,
            mult=self.mult,
            channel_mult=self.channel_mult,
            prediction_length=seq_len,
            num_preprocess_blocks=self.num_preprocess_blocks,
            num_preprocess_cells=self.num_preprocess_cells,
            num_channels_enc=self.num_channels_enc,
            arch_instance=self.arch_instance,
            num_latent_per_group=self.num_latent_per_group,
            num_channels_dec=self.num_channels_dec,
            num_postprocess_blocks=self.num_postprocess_blocks,
            num_postprocess_cells=self.num_postprocess_cells,
            hidden_size=self.hidden_size,
            target_dim=self.dataset.num_features,
            num_layers=self.num_layers,
            groups_per_scale=self.groups_per_scale
        )

        self.model = self.model.to(self.device)
        

    def _init_data_loader(self):
        
        self._init_dataset()
        
        self.scaler = parse_type(self.scaler_type, globals=globals())()

        self.dataloader = SlidingWindowTS(
            self.dataset,
            self.scaler,
            window=self.windows,
            horizon=self.horizon,
            steps=self.pred_len,
            scale_in_train=True,
            shuffle_train=True,
            freq=self.dataset.freq,
            batch_size=self.batch_size,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            num_worker=self.num_worker,
        )


        if self.pred_len < self.windows:
            # train using same window and pred_length
            self.dataloader1 = SlidingWindowTS(
                self.dataset,
                self.scaler,
                window=self.windows,
                horizon=self.horizon,
                steps=self.windows,
                scale_in_train=True,
                shuffle_train=True,
                freq=self.dataset.freq,
                batch_size=self.batch_size,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                num_worker=self.num_worker,
            )
            self.train_loader = self.dataloader1.train_loader
        
            self.val_loader, self.test_loader = (
                self.dataloader.val_loader,
                self.dataloader.test_loader,
            )

        else:
            self.train_loader, self.val_loader, self.test_loader = (
                self.dataloader.train_loader,
                self.dataloader.val_loader,
                self.dataloader.test_loader,
            )
        self.train_steps = len(self.train_loader.dataset)
        self.val_steps = len(self.val_loader.dataset)
        self.test_steps = len(self.test_loader.dataset)

        print(f"train steps: {self.train_steps}")
        print(f"val steps: {self.val_steps}")
        print(f"test steps: {self.test_steps}")
        
        
    def _process_train_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)

        # the original version of D3 VAE assumes seq_len == pred_len, I circumvent this by padding the input to the output
        if self.windows < self.pred_len:
            # pad input if input length is shorter
            batch_x = torch.nn.functional.pad(batch_x.transpose(1, 2), (self.pred_len - self.windows, 0)).transpose(1, 2)
            batch_x_date_enc = torch.nn.functional.pad(batch_x_date_enc.transpose(1, 2), (self.pred_len - self.windows, 0)).transpose(1, 2)
        # elif self.windows > self.pred_len:
        #     # pad pred if pred length is shorter
        #     batch_y = torch.nn.functional.pad(batch_y.transpose(1, 2), (0, self.windows - self.pred_len)).transpose(1, 2)
        #     batch_y_date_enc = torch.nn.functional.pad(batch_y_date_enc.transpose(1, 2), (0, self.windows - self.pred_len)).transpose(1, 2)

        output, y_noisy, total_c, loss = self.model(batch_x,batch_x_date_enc,  batch_y)
        return output, y_noisy, total_c, loss


    def _process_val_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, O, N, S)
        # - label: (B, O, N)
        
        if self.windows < self.pred_len:
            # pad input if input length is shorter
            batch_x = torch.nn.functional.pad(batch_x.transpose(1, 2), (self.pred_len - self.windows, 0)).transpose(1, 2)
            batch_x_date_enc = torch.nn.functional.pad(batch_x_date_enc.transpose(1, 2), (self.pred_len - self.windows, 0)).transpose(1, 2)

        # y, out,output, tc = self.model.prob_pred(batch_x, batch_x_date_enc)
        y, out,output, tc = self.model.pred(batch_x, batch_x_date_enc)
        out = out.permute(0, 2, 3, 1)[:, :self.pred_len, :, :]
        return out, batch_y



    def _train(self):
        with torch.enable_grad(), tqdm(total=len(self.train_loader.dataset)) as progress_bar:
            self.model.train()
            train_loss = []
            for i, (
                batch_x,
                batch_y,
                origin_x,
                origin_y,
                batch_x_date_enc,
                batch_y_date_enc,
            ) in enumerate(self.train_loader):
                self.model_optim.zero_grad()
                
                start = time.time()
                origin_y = origin_y.to(self.device)
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                output, y_noisy, total_c, loss2 = self._process_train_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )
                
                recon = output.log_prob(y_noisy)
                mse_loss = self.loss_func(output.sample(), y_noisy)
                loss1 = - torch.mean(torch.sum(recon, axis=[1, 2, 3]))
                
                # The bigger total correlation means better disentanglement.
                loss = loss1*self.psi + loss2*self.lambda1 + mse_loss - self.gamma*total_c
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                
                progress_bar.update(batch_x.size(0))
                train_loss.append(loss.item())
                
                progress_bar.set_postfix(
                    loss=loss.item(),
                    lr=self.model_optim.param_groups[0]["lr"],
                    epoch=self.current_epoch,
                    refresh=True,
                )
                self.model_optim.step()

            return train_loss

if __name__ == "__main__":
    import fire
    # torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    fire.Fire(D3VAEForecast)
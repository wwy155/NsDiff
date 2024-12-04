from dataclasses import dataclass, field
import sys
from typing import List, Dict
import os

import torch
from dataclasses import dataclass, asdict, field
from torch_timeseries.nn.embedding import freq_map
from src.models.TimeDiff import TimeDiff
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
from types import SimpleNamespace


@dataclass
class TimeDiffParameters:

    beta_start: float =  0.0001
    beta_end: float =  0.5
    num_steps: int =  100
    vis_ar_part: int =  0
    num_samples :int = 100
    vis_MTS_analysis: int =  1
    schedule:  str = "quad"
    use_window_normalization: bool = True
    t0: float =  1e-4
    T: float =  1
    nfe: int =  100
    dim_LSTM: int =  64
    UNet_Type: str =  'CNN'
    D3PM_kernel_size: int =  5
    use_freq_enhance: int =  0
    type_sampler: str =  'dpm'
    parameterization: int =  'x_start'
    ddpm_inp_embed: int =  256
    ddpm_dim_diff_steps: int =  256
    ddpm_channels_conv: float =  256
    ddpm_channels_fusion_I: float =  256
    ddpm_layers_inp: float =  5
    ddpm_layers_I: float =  5
    ddpm_layers_II: float =  5
    cond_ddpm_num_layers: float =  5
    cond_ddpm_channels_conv: float =  64

    ablation_study_case: str =  "none"
    weight_pred_loss: float =  0.0
    ablation_study_F_type: str =  "CNN"
    ablation_study_masking_type: str =  "none"
    ablation_study_masking_tau: float =  0.9
    ot_ode: bool =  True

@dataclass
class TimeDiffForecast(ProbForecastExp, TimeDiffParameters):
    model_type: str = "TimeDiff"
    def _init_model(self):
        
        self.label_len = self.pred_len//2
        args_dict = {
            "seq_len": self.windows,
            "device": self.device,
            "pred_len": self.pred_len,
            "label_len": self.label_len,
            "features" : 'M',
            "vis_ar_part" : self.vis_ar_part,
            "vis_MTS_analysis" : self.vis_MTS_analysis,
            "num_vars" : self.dataset.num_features,
            "freq" : self.dataset.freq,
            "interval" : self.num_steps,
            "beta-max": self.beta_end,
            "use_window_normalization": self.use_window_normalization,
            "t0": self.t0,
            "T": self.T,
            "nfe": self.nfe,
            "dim_LSTM": self.dim_LSTM,
            "diff_steps": self.num_steps,
            "UNet_Type": self.UNet_Type,
            "D3PM_kernel_size": self.D3PM_kernel_size,
            "use_freq_enhance": self.use_freq_enhance,
            "type_sampler": self.type_sampler,
            "parameterization": self.parameterization,
            "ddpm_inp_embed": self.ddpm_inp_embed,
            "ddpm_dim_diff_steps": self.ddpm_dim_diff_steps,
            "ddpm_channels_conv": self.ddpm_channels_conv,
            "ddpm_channels_fusion_I": self.ddpm_channels_fusion_I,
            "ddpm_layers_inp": self.ddpm_layers_inp,
            "ddpm_layers_I": self.ddpm_layers_I,
            "ddpm_layers_II": self.ddpm_layers_II,
            "cond_ddpm_num_layers": self.cond_ddpm_num_layers,
            "cond_ddpm_channels_conv": self.cond_ddpm_channels_conv,
            "ablation_study_case": self.ablation_study_case,
            "weight_pred_loss": self.weight_pred_loss,
            "ablation_study_F_type": self.ablation_study_F_type,
            "ablation_study_masking_type": self.ablation_study_masking_type,
            "ablation_study_masking_tau": self.ablation_study_masking_tau,
            "ot-ode": self.ot_ode, 
        }
        
        args = SimpleNamespace(**args_dict)
        self.model = TimeDiff(args)
        self.model = self.model.to(self.device)
        
        # self.gt_mask = torch.concat([
        #         torch.ones(size=(self.windows, self.dataset.num_features)),
        #         torch.zeros(size=(self.pred_len, self.dataset.num_features)),
        #     ]).to(self.device).bool()
        
        # self.observation_mask = ~self.gt_mask



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
                origin_y = origin_y.to(self.device).float()
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()
                self.model_optim.zero_grad()
                loss = self._process_train_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )
                if self.invtrans_loss:
                    pred = self.scaler.inverse_transform(pred)
                    true = origin_y
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

    def _process_train_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        
        # Time Diff need the batch_x to be a even number
        def get_even(n):
            n = n if n.shape[0]%2 == 0 else torch.concat([n, n[0:1, :, :]], dim=0)
            return n
        
        batch_x = get_even(batch_x)
        batch_y = get_even(batch_y)
        batch_x_date_enc = get_even(batch_x_date_enc)
        batch_y_date_enc = get_even(batch_y_date_enc)
        
        dec_inp_pred = torch.zeros(
            [batch_x.size(0), self.pred_len, self.dataset.num_features]
        ).to(self.device)
        dec_inp_label = batch_x[:, self.label_len :, :].to(self.device)

        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        dec_inp_date_enc = torch.cat(
            [batch_x_date_enc[:, self.label_len :, :], batch_y_date_enc], dim=1
        )

        loss= self.model.train_forward(batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc)
        return loss


    def _process_val_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        
        dec_inp_pred = torch.zeros(
            [batch_x.size(0), self.pred_len, self.dataset.num_features]
        ).to(self.device)
        dec_inp_label = batch_x[:, self.label_len :, :].to(self.device)

        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        dec_inp_date_enc = torch.cat(
            [batch_x_date_enc[:, self.label_len :, :], batch_y_date_enc], dim=1
        )

        outs, x, y , _ , _ = self.model(batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc, None, None, None, self.num_samples)
        outs = outs.permute(0, 2, 3, 1)
        assert (outs.shape[1], outs.shape[2], outs.shape[3]) == (self.pred_len, self.dataset.num_features, self.num_samples)
        return outs, batch_y


if __name__ == "__main__":
    import fire
    # torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    fire.Fire(TimeDiffForecast)
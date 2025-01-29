# import codecs
from dataclasses import asdict, dataclass
import datetime
import hashlib
import json
import os
import random
import time
from typing import Dict, List, Type, Union

import numpy as np
import pandas as pd
import torch
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from tqdm import tqdm
from torch.nn import MSELoss, L1Loss
from torch.optim import *
import src.layer.mu_backbone as ns_Transformer

from torch_timeseries.dataset import *
from src.datasets import *
from torch_timeseries.scaler import *
from src.metrics import CRPS, CRPSSum, QICE, PICP
from src.metrics import ProbMAE, ProbMSE, ProbRMSE
from types import SimpleNamespace

from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.early_stop import EarlyStopping
from torch_timeseries.utils.parse_type import parse_type
from torch_timeseries.utils.reproduce import reproducible
from torch_timeseries.core import TimeSeriesDataset, BaseIrrelevant, BaseRelevant
from torch_timeseries.dataloader import SlidingWindowTS, ETTHLoader, ETTMLoader
from torch_timeseries.experiments import ForecastExp
from torch_timeseries.utils import asdict_exc
import torch.multiprocessing as mp



@dataclass
class NsDiffFParameters:
    beta_start: float =  0.0001
    beta_end: float =  0.5
    d_model: int =  512
    n_heads: int =  8
    e_layers: int =  2
    d_layers: int =  1
    d_ff: int =  1024
    diffusion_steps :int = 100
    moving_avg: int =  25
    factor: int =  3
    distil: bool =  True
    dropout: float =  0.05
    activation: str = 'gelu'
    k_z: float =  1e-2
    k_cond: int =  1
    d_z: int =  8
    CART_input_x_embed_dim : int= 32
    p_hidden_layers : int = 2
    rolling_length : int = 24




@dataclass
class FForecast(ForecastExp, NsDiffFParameters):
    model_type: str = "F"
    def _init_model(self):
        self.label_len = self.windows // 2
        args_dict = {
            "seq_len": self.windows,
            "device": self.device,
            "pred_len": self.pred_len,
            "label_len": self.label_len,
            "features" : 'M',
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "enc_in" : self.dataset.num_features,
            "dec_in" : self.dataset.num_features,
            "c_out" : self.dataset.num_features,
            "d_model" : self.d_model,
            "n_heads" : self.n_heads,
            "e_layers" : self.e_layers,
            "d_layers" : self.d_layers,
            "d_ff" : self.d_ff,
            "moving_avg" : self.moving_avg,
            "timesteps" : self.diffusion_steps,
            "factor" : self.factor,
            "distil" : self.distil,
            "embed" : 'timeF',
            "dropout" :self.dropout,
            "activation" :self.activation,
            "output_attention" : False,
            "do_predict" :True,
            "k_z" :self.k_z,
            "k_cond" :self.k_cond,
            "p_hidden_dims" : [64, 64],
            "freq" :self.dataset.freq,
            "CART_input_x_embed_dim" : self.CART_input_x_embed_dim,
            "p_hidden_layers" : self.p_hidden_layers,
            "d_z" :self.d_z,
            "diffusion_config_dir" : "./configs/tmdm.yml",
        }

        self.args = SimpleNamespace(**args_dict)
        self.model = ns_Transformer.Model(self.args).float().to(self.device)
        # self.cond_pred_model_g = G.SigmaEstimation(self.windows, self.pred_len, self.dataset.num_features, 
        #                                            128, self.rolling_length).float().to(self.device)

    def _run_identifier(self, seed) -> str:
        return str(seed)
    
    def _process_one_batch(
        self,
        batch_x,
        batch_y,
        batch_origin_x,
        batch_origin_y,
        batch_x_mark,
        batch_y_mark,
    ):
        # inputs:
        # batch_x:  (B, T, N)
        # batch_y:  (B, Steps,T)
        # batch_x_date_enc:  (B, T, N)
        # batch_y_date_enc:  (B, T, Steps)

        # outputs:
        # pred: (B, O, N)
        # label:  (B,O,N)
        # for single step you should output (B, N)
        # for multiple steps you should output (B, O, N)
        
        batch_x = batch_x.to(self.device).float()
        batch_y = batch_y.to(self.device).float()
        batch_x_mark = batch_x_mark.to(self.device).float()
        batch_y_mark = batch_y_mark.to(self.device).float()
        
        batch_y_input = torch.concat([batch_x[:, -self.label_len:, :], batch_y], dim=1)
        batch_y_mark_input = torch.concat([batch_x_mark[:, -self.label_len:, :], batch_y_mark], dim=1)

        dec_inp_pred = torch.zeros(
            [batch_x.size(0), self.pred_len, self.dataset.num_features]
        ).to(self.device)
        dec_inp_label = batch_x[:, -self.label_len :, :].to(self.device)

        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        

        y_0_hat_batch, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark_input)
        return y_0_hat_batch, batch_y

    
if __name__ == "__main__":
    import fire
    # torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    fire.Fire(FForecast)
    
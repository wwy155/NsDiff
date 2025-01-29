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
import src.layer.g_backbone as G

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
from src.utils.sigma import wv_sigma, wv_sigma_trailing



@dataclass
class NsDiffFParameters:
    hidden_size : int  = 512
    rolling_length : int  = 24


@dataclass
class GForecast(ForecastExp, NsDiffFParameters):
    model_type: str = "G"
    
    def _init_model(self):
        self.model = G.SigmaEstimation(self.windows, self.pred_len, self.dataset.num_features, 
                                                self.hidden_size, self.rolling_length).float().to(self.device)

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
        
        y_sigma = wv_sigma_trailing(torch.concat([batch_x, batch_y], dim=1), self.rolling_length) 
        y_sigma = y_sigma[:, -self.pred_len:, :] + 10e-8
        gx = self.model(batch_x) # (B, O, N)
        return gx, y_sigma

    
if __name__ == "__main__":
    import fire
    # torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    fire.Fire(GForecast)
    
from dataclasses import dataclass, field
import sys
from typing import List, Dict
import os

import torch
from dataclasses import dataclass, asdict, field
from torch_timeseries.nn.embedding import freq_map
from src.models.DiffusionTS import Diffusion_TS
from src.experiments.prob_forecast import ProbForecastExp
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from tqdm import tqdm
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible
import time
# import multiprocessing
import torch.multiprocessing as mp

from ema_pytorch import EMA
import numpy as np
import torch.distributed as dist
import torch
from tqdm import tqdm
import concurrent.futures
from types import SimpleNamespace


@dataclass
class DiffusionTSParameters:

    # beta_start: float =  0.0001
    # beta_end: float =  0.5
    # num_steps: int =  100
    # vis_ar_part: int =  0
    num_samples :int = 20

    n_layer_enc: int =3
    n_layer_dec: int =6
    d_model: int = 64
    timesteps: int = 500
    sampling_timesteps: int = 200
    loss_type : str ='l1'
    beta_schedule: str ='cosine'
    n_heads: int =4
    mlp_hidden_times: int =4
    eta: float =0.
    attn_pd: float =0.
    resid_pd: float=0.
    kernel_size: int =1
    padding_size: int =0
    use_ff: bool =True
    
    decay : float = 0.995
    update_interval : int = 10
    
    reg_weight: float = None
    
@dataclass
class DiffusionTSForecast(ProbForecastExp, DiffusionTSParameters):
    model_type: str = "DiffusionTS"
    def _init_model(self):
        self.model = Diffusion_TS(
            seq_length= self.windows + self.pred_len,
            feature_size=self.dataset.num_features,
            n_layer_enc=self.n_layer_enc,
            n_layer_dec=self.n_layer_dec,
            d_model=self.d_model,
            timesteps=self.timesteps,
            sampling_timesteps=self.sampling_timesteps,
            loss_type='l2',
            beta_schedule=self.beta_schedule,
            n_heads=self.n_heads,
            mlp_hidden_times=self.mlp_hidden_times,
            eta=self.eta,
            attn_pd=self.attn_pd,
            resid_pd=self.resid_pd,
            kernel_size=self.kernel_size,
            padding_size=self.padding_size,
            use_ff=self.use_ff,
            reg_weight=self.reg_weight,
        )
        self.model = self.model.to(self.device)
        # self.ema = EMA(self.model, beta=self.decay, update_every=self.update_interval).to(self.device)
        self.gt_mask = torch.concat([
                torch.ones(size=(self.windows, self.dataset.num_features)),
                torch.zeros(size=(self.pred_len, self.dataset.num_features)),
            ]).to(self.device).bool()
        
        self.observation_mask = ~self.gt_mask



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
                loss = self._process_train_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )
                if self.invtrans_loss:
                    pred = self.scaler.inverse_transform(pred)
                    true = origin_y
                loss.backward()
                

                # torch.nn.utils.clip_grad_norm_(
                #     self.model.parameters(), self.max_grad_norm
                # )
                
                progress_bar.update(batch_x.size(0))
                
                train_loss.append(loss.item())
                progress_bar.set_postfix(
                    loss=loss.item(),
                    lr=self.model_optim.param_groups[0]["lr"],
                    epoch=self.current_epoch,
                    refresh=True,
                )
                self.model_optim.step()
                self.model_optim.zero_grad()
                

            return train_loss

    def _process_train_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        
        # dec_inp_pred = torch.zeros(
        #     [batch_x.size(0), self.pred_len, self.dataset.num_features]
        # ).to(self.device)
        # dec_inp_label = batch_x[:, self.label_len :, :].to(self.device)

        # dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        # dec_inp_date_enc = torch.cat(
        #     [batch_x_date_enc[:, self.label_len :, :], batch_y_date_enc], dim=1
        # )

        data = torch.concat([batch_x, batch_y], dim=1)
        loss= self.model(data, target=data)
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
        
        # dec_inp_pred = torch.zeros(
        #     [batch_x.size(0), self.pred_len, self.dataset.num_features]
        # ).to(self.device)
        # dec_inp_label = batch_x[:, self.label_len :, :].to(self.device)

        # dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        # dec_inp_date_enc = torch.cat(
        #     [batch_x_date_enc[:, self.label_len :, :], batch_y_date_enc], dim=1
        # )
        x = torch.concat([batch_x, batch_y], dim=1)
        B = x.shape[0] 
        # inpu_date =  torch.concat([batch_x_date_enc, batch_y_date_enc], dim=1)
        
        # t_m : 1 when target(predict), 0 when features
        t_m = self.gt_mask.expand((x.shape[0], -1, -1))
        
        x = x.repeat(self.num_samples, 1, 1)
        t_m = t_m.repeat(self.num_samples, 1, 1)
        
        coef = 1e-1
        stepsize = 5e-2
        model_kwargs = {}
        model_kwargs['coef'] = coef
        model_kwargs['learning_rate'] = stepsize

        sampling_steps =  self.sampling_timesteps

        
        # shape = (self.pred_len, self.dataset.num_features)

        # samples = np.empty([0, shape[0], shape[1]])
        # reals = np.empty([0, shape[0], shape[1]])
        # masks = np.empty([0, shape[0], shape[1]])

        # for idx, (x, t_m) in enumerate(raw_dataloader):
            # x, t_m = x.to(self.device), t_m.to(self.device)
        
        with torch.no_grad():
            # this workd, sample_infill do not work....
            sample = self.model.fast_sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m, model_kwargs=model_kwargs,
            
            # if sampling_steps == self.model.num_timesteps:
            #     sample = self.model.sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m,
            #                                                 model_kwargs=model_kwargs)
            # else:
            #     sample = self.model.fast_sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m, model_kwargs=model_kwargs,
                                                                sampling_timesteps=sampling_steps)
            # samples = np.row_stack([samples, sample[:, -self.pred_len:, :].detach().cpu().numpy()])
            # reals = np.row_stack([reals, x.detach().cpu().numpy()])
            # masks = np.row_stack([masks, t_m.detach().cpu().numpy()])
        sample = sample[:, -self.pred_len:, :]
        sample = sample.reshape(B, self.num_samples, self.pred_len, self.dataset.num_features).permute(0,2,3, 1)
        assert (sample.shape[1], sample.shape[2], sample.shape[3]) == (self.pred_len, self.dataset.num_features, self.num_samples)
        return sample, batch_y


    def _evaluate(self, dataloader):
        self.model.eval()
        self.metrics.reset()
        results = []
        with tqdm(total=len(dataloader.dataset)) as progress_bar:
            for batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc in dataloader:
                batch_size = batch_x.size(0)
                origin_x = origin_x.to(self.device)
                origin_y = origin_y.to(self.device)
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()
                preds, truths = self._process_val_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )
                origin_y = origin_y.to(self.device)
                if self.invtrans_loss:
                    preds = self.scaler.inverse_transform(preds)
                    truths = origin_y
                    
                self.metrics.update(preds.contiguous().cpu().detach(), truths.contiguous().cpu().detach())
                # if isinstance(preds, np.ndarray):
                #     results.append(self.task_pool.apply_async(update_metrics, (preds, truths, self.metrics)))
                # else:
                progress_bar.update(batch_x.shape[0])

        # for result in results:
        #     result.get()  # Ensure the metric update is finished

        result = {name: float(metric.compute()) for name, metric in self.metrics.items()}
        return result

if __name__ == "__main__":
    import fire
    # torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    fire.Fire(DiffusionTSForecast)
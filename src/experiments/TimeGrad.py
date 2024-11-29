from dataclasses import dataclass, field
import sys
from typing import List, Dict
import os

import torch
from src.experiments.forecast import ForecastExp 
from dataclasses import dataclass, asdict, field
from torch_timeseries.nn.embedding import freq_map
from src.models.TimeGrad import TimeGrad
from src.metrics import CRPS, CRPSSum, QICE, PICP
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from tqdm import tqdm
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible
import time
# import multiprocessing
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch.distributed as dist
import torch
from tqdm import tqdm
import concurrent.futures


def update_metrics(preds, truths, metrics):
    """Function to update metrics in a separate process."""
    metrics.update(preds.contiguous(), truths.contiguous())



@dataclass
class TimeGradParameters:
    num_layers : int = 2
    rnn_hidden_size : int = 40
    rnn_type : str = 'LSTM'
    dropout_rate : float = 0.1
    lags_seq : List[int] = field(default_factory= lambda : [1, 24, 168])
    diff_steps : int = 20
    beta_end : float = 0.1
    beta_schedule : str = 'linear'
    residual_layers : int = 8
    residual_channels : int = 8
    context_length : int = 24
    covariate_hidden_size : int = 128
    conditioning_length : int = 100
    dilation_cycle_length : int = 2
    num_samples : int = 100
    
    scale:bool = True

@dataclass
class TimeGradForecast(ForecastExp, TimeGradParameters):
    model_type: str = "TimeGrad"

    def _init_model(self):
        self.model = TimeGrad(
            pred_len=self.pred_len,
            sequence_length=self.windows,
            enc_in=self.dataset.num_features,
            lags=self.lags_seq,
            context_length=self.context_length,
            TimeF = freq_map[self.dataset.freq],
            rnn_hidden_size=self.rnn_hidden_size,
            rnn_type = self.rnn_type,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            scale=self.scale,            
            conditioning_length=self.conditioning_length,
            covariate_hidden_size=self.covariate_hidden_size,
            diff_steps=self.diff_steps,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            residual_layers=self.residual_layers,
            residual_channels=self.residual_channels,
            dilation_cycle_length=self.dilation_cycle_length,
            )
        self.model = self.model.to(self.device)
        
        
    def _init_metrics(self):
        self.metrics = MetricCollection(
            metrics={
                "crps": CRPS(),
                "crps_sum": CRPSSum(),
                "qice": QICE(),
                "picp": PICP(),
            }
        )
        self.metrics.to("cpu")
        ctx = mp.get_context("spawn")  # Options: 'fork', 'spawn', 'forkserver'
        self.task_pool = ctx.Pool(processes=32)

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

        # for resumable reproducibility
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
                wandb.log( {f"test_{k}": v for k, v in test_result.items()}, step=self.current_epoch)

            # self.scheduler.step()

        self._load_best_model()
        best_test_result = self._test()
        if self._use_wandb():
            for k, v in best_test_result.items(): wandb.run.summary[f"best_test_{k}"] = v 
        
        if self._use_wandb():  wandb.finish()
        return best_test_result
    
    
    def _process_train_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        batch_x = batch_x.to(self.device).float()
        batch_y = batch_y.to(self.device).float()
        batch_x_date_enc = batch_x_date_enc.to(self.device).float()
        batch_y_date_enc = batch_y_date_enc.to(self.device).float()

        # no decoder input
        # label_len = 1
        noise, pred_noise = self.model(batch_x, batch_y, batch_x_date_enc, batch_y_date_enc, train=True)
        return noise, pred_noise

    # def _evaluate(self, dataloader):
    #     self.model.eval()
    #     self.metrics.reset()

    #     with torch.no_grad():
    #         with tqdm(total=len(dataloader.dataset)) as progress_bar:
    #             for (
    #                 batch_x,
    #                 batch_y,
    #                 batch_origin_x,
    #                 batch_origin_y,
    #                 batch_x_date_enc,
    #                 batch_y_date_enc,
    #             ) in dataloader:
    #                 batch_size = batch_x.size(0)
    #                 preds, truths = self._process_val_batch(
    #                     batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
    #                 )
    #                 batch_origin_y = batch_origin_y.to(self.device)
    #                 if self.invtrans_loss:
    #                     preds = self.scaler.inverse_transform(preds)
    #                     truths = batch_origin_y
                    
    #                 self.metrics.update(preds.contiguous(), truths.contiguous())

    #                 progress_bar.update(batch_x.shape[0])

    #         result = {
    #             name: float(metric.compute()) for name, metric in self.metrics.items()
    #         }
    #     return result



    def _evaluate(self, dataloader):
        self.model.eval()
        self.metrics.reset()
        results = []
        with tqdm(total=len(dataloader.dataset)) as progress_bar:
            for batch_x, batch_y, batch_origin_x, batch_origin_y, batch_x_date_enc, batch_y_date_enc in dataloader:
                batch_size = batch_x.size(0)
                preds, truths = self._process_val_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )
                batch_origin_y = batch_origin_y.to(self.device)
                if self.invtrans_loss:
                    preds = self.scaler.inverse_transform(preds)
                    truths = batch_origin_y

                results.append(self.task_pool.apply_async(update_metrics, (preds.contiguous().cpu(), truths.contiguous().cpu(), self.metrics)))
                
                progress_bar.update(batch_x.shape[0])

        for result in results:
            result.get()  # Ensure the metric update is finished

        result = {name: float(metric.compute()) for name, metric in self.metrics.items()}
        return result

    # def _evaluate(self, dataloader):
    #     self.model.eval()
    #     self.metrics.reset()

    #     # Prepare multiprocessing environment
    #     results = []
    #     with torch.no_grad():
    #         # Use multiprocessing Pool to process batches in parallel
    #         with multiprocessing.Pool(processes=32) as pool:
    #             with tqdm(total=len(dataloader.dataset)) as progress_bar:
    #                 for (
    #                     batch_x,
    #                     batch_y,
    #                     batch_origin_x,
    #                     batch_origin_y,
    #                     batch_x_date_enc,
    #                     batch_y_date_enc,
    #                 ) in dataloader:
    #                     # Parallelize batch processing
    #                     print(batch_origin_x.shape[0])
    #                     result = pool.apply_async(
    #                         process_batch,
    #                         (batch_x, batch_y, batch_origin_x, batch_origin_y, batch_x_date_enc, batch_y_date_enc, self.model, self.scaler, self.metrics, self.invtrans_loss, self.device)
    #                     )
    #                     results.append(result)
    #                     progress_bar.update(batch_x.shape[0])

    #                 # Wait for all processes to finish
    #                 for result in results:
    #                     result.get()

    #     # Aggregate final results
    #     final_metrics = {
    #         name: float(metric.compute()) for name, metric in self.metrics.items()
    #     }
    #     return final_metrics



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
                origin_y = origin_y.to(self.device)
                self.model_optim.zero_grad()
                pred, true = self._process_train_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )
                if self.invtrans_loss:
                    pred = self.scaler.inverse_transform(pred)
                    true = origin_y
                loss = self.loss_func(pred, true)
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



    def _process_val_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        batch_x = batch_x.to(self.device).float()
        batch_y = batch_y.to(self.device).float()
        batch_x_date_enc = batch_x_date_enc.to(self.device).float()
        batch_y_date_enc = batch_y_date_enc.to(self.device).float()

        # no decoder input
        # label_len = 1
        # B, S, O, N
        output = self.model(batch_x, batch_y, batch_x_date_enc, batch_y_date_enc, train=False)
        output = output.permute(0, 2, 3, 1) # B, O, N, S
        return output, batch_y





if __name__ == "__main__":
    import fire
    # torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    fire.Fire(TimeGradForecast)
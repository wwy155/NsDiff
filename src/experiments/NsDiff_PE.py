from dataclasses import dataclass, field
import sys
from typing import List, Dict
import os
import wandb
import torch
from dataclasses import dataclass, asdict, field
from torch_timeseries.nn.embedding import freq_map
from src.models.NsDiff import NsDiff
import src.layer.mu_backbone as ns_Transformer
import argparse
import src.layer.g_backbone as G
from src.experiments.prob_forecast import ProbForecastExp
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from torch.optim import *
from tqdm import tqdm
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible
import time
# import multiprocessing
import torch.multiprocessing as mp
from torch_timeseries.utils.parse_type import parse_type

from torch_timeseries.utils.early_stop import EarlyStopping
from src.layer.nsdiff_utils import q_sample, p_sample_loop_pe, cal_sigma12, cal_sigma_tilde, cal_forward_noise
import yaml
import numpy as np
import torch.distributed as dist
import torch
from tqdm import tqdm
import concurrent.futures
from types import SimpleNamespace
from src.utils.sigma import wv_sigma, wv_sigma_trailing
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


EPS= 10e-8
class NSDiffEarlyStopping(EarlyStopping):
    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model['model'].state_dict(), os.path.join(self.path, 'model.pth'))
        torch.save(model['cond_pred_model'].state_dict(),os.path.join(self.path, 'cond_pred_model.pth'))
        torch.save(model['cond_pred_model_g'].state_dict(),os.path.join(self.path, 'cond_pred_model_g.pth'))
        self.val_loss_min = val_loss
        
        
        
def log_normal(x, mu, var):
    """Logarithm of normal distribution with mean=mu and variance=var
       log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2

    Args:
       x: (array) corresponding array containing the input
       mu: (array) corresponding array containing the mean
       var: (array) corresponding array containing the variance

    Returns:
       output: (array/float) depending on average parameters the result will be the mean
                            of all the sample losses or an array with the losses per sample
    """
    eps = 1e-8
    if eps > 0.0:
        var = var + eps
    # return -0.5 * torch.sum(
    #     np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)
    return 0.5 * torch.mean(
        np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)



@dataclass
class NsDiffParameters:
    num_samples : int = 100 
    beta_start: float =  0.0001
    beta_end: float =  0.01
    d_model: int =  512
    n_heads: int =  8
    e_layers: int =  2
    d_layers: int =  1
    d_ff: int =  1024
    diffusion_steps :int = 20 # 20
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
    rolling_length : int = 96

@dataclass
class NsDiffForecast(ProbForecastExp, NsDiffParameters):
    model_type: str = "NsDiffPE"
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
            "beta_schedule": "linear",
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
            "diffusion_config_dir" : "./configs/nsdiff.yml",
        }

        with open("./configs/nsdiff.yml", "r") as f:
            config = yaml.unsafe_load(f)
            self.diffusion_config = dict2namespace(config)


        self.args = SimpleNamespace(**args_dict)
        
        self.model = NsDiff(self.args, self.device).to(self.device)
        self.cond_pred_model = ns_Transformer.Model(self.args).float().to(self.device)
        self.cond_pred_model_g = G.SigmaEstimation(self.windows, self.pred_len, self.dataset.num_features, 512, self.rolling_length).float().to(self.device)
        
        
        # model_f_path = f"./results/runs/F/{self.dataset_type}/w{self.windows}h1s{self.pred_len}/1/best_model.pth"
        # model_g_path = f"./results/runs/G/{self.dataset_type}/w{self.windows}h1s{self.pred_len}/1/best_model.pth"
        # self.cond_pred_model.load_state_dict(torch.load(model_f_path, map_location=self.device))
        # self.cond_pred_model_g.load_state_dict(torch.load(model_g_path, map_location=self.device))



    def _init_optimizer(self):
        self.model_optim = parse_type(self.optm_type, globals=globals())(
            [{'params': self.model.parameters()}, {'params': self.cond_pred_model.parameters()}, {'params': self.cond_pred_model_g.parameters()}], 
            lr=self.lr, 
            # weight_decay=self.l2_weight_decay
        )
        # self.model_optim = parse_type(self.optm_type, globals=globals())(
        #     [{'params': self.model.parameters()}], 
        #     lr=self.lr, 
        #     # weight_decay=self.l2_weight_decay
        # )



    def _load_best_model(self):
        self.model.load_state_dict(
            torch.load(os.path.join(self.run_save_dir, 'model.pth'), map_location=self.device)
        )
        self.cond_pred_model.load_state_dict(
            torch.load(os.path.join(self.run_save_dir, 'cond_pred_model.pth'), map_location=self.device)
        )
        self.cond_pred_model_g.load_state_dict(
            torch.load(os.path.join(self.run_save_dir, 'cond_pred_model_g.pth'), map_location=self.device)
        )

    def _setup_early_stopper(self):
        self.best_checkpoint_filepath = os.path.join(
            self.run_save_dir, "model.pth"
        )
        self.best_cond_checkpoint_filepath = os.path.join(
            self.run_save_dir, "cond_pred_model.pth"
        )
        self.best_cond_g_checkpoint_filepath = os.path.join(
            self.run_save_dir, "cond_pred_model_g.pth"
        )
        self.early_stopper = NSDiffEarlyStopping(
            self.patience, verbose=True, path=self.run_save_dir
        )


    def _save_run_check_point(self, seed):


        if not os.path.exists(self.run_save_dir):
            os.makedirs(self.run_save_dir)
        
        
        print(f"Saving run checkpoint to '{self.run_save_dir}'.")

        self.run_state = {
            "model": self.model.state_dict(),
            "cond_pred_model": self.cond_pred_model.state_dict(),
            "cond_pred_model_g": self.cond_pred_model_g.state_dict(),
            "current_epoch": self.current_epoch,
            "optimizer": self.model_optim.state_dict(),
            "rng_state": torch.get_rng_state(),
            "early_stopping": self.early_stopper.get_state(),
        }

        torch.save(self.run_state, f"{self.run_checkpoint_filepath}")
        print("Run state saved ... ")

    def _load_best_model(self):
        self.model.load_state_dict(
            torch.load(self.best_checkpoint_filepath, map_location=self.device)
        )
        self.cond_pred_model.load_state_dict(
            torch.load(self.best_cond_checkpoint_filepath, map_location=self.device)
        )
        self.cond_pred_model_g.load_state_dict(
            torch.load(self.best_cond_g_checkpoint_filepath, map_location=self.device)
        )


    def _resume_run(self, seed):
        # only train loader rshould be checkedpoint to keep the validation and test consistency
        check_point = torch.load(self.run_checkpoint_filepath, map_location=self.device)

        self.model.load_state_dict(check_point["model"])
        self.cond_pred_model.load_state_dict(check_point["cond_pred_model"])
        self.cond_pred_model_g.load_state_dict(check_point["cond_pred_model_g"])
        self.model_optim.load_state_dict(check_point["optimizer"])
        self.current_epoch = check_point["current_epoch"]

        self.early_stopper.set_state(check_point["early_stopping"])

    def _train(self):
        self.model.train()
        self.cond_pred_model.train()
        self.cond_pred_model_g.train()

        with torch.enable_grad(), tqdm(total=len(self.train_loader.dataset)) as progress_bar:
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
                

        self.model.eval()
        self.cond_pred_model.eval()
        self.cond_pred_model_g.eval()
        return train_loss

    def _process_train_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        
        # Time Diff need the batch_x to be a even number
        
        
        # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        # dec_inp = torch.cat([batch_x[:, -self.label_len:, :], dec_inp], dim=1).float().to(self.device)
        y_sigma = wv_sigma(batch_y, self.rolling_length) + EPS
        y_sigma = wv_sigma_trailing(torch.concat([batch_x, batch_y], dim=1), self.rolling_length) 
        y_sigma = y_sigma[:, -self.pred_len:, :] + EPS
        
        y_sigma = torch.ones_like(y_sigma).to(self.device)
        
        batch_y_input = torch.concat([batch_x[:, -self.label_len:, :], batch_y], dim=1)
        batch_y_mark_input = torch.concat([batch_x_mark[:, -self.label_len:, :], batch_y_mark], dim=1)

        dec_inp_pred = torch.zeros(
            [batch_x.size(0), self.pred_len, self.dataset.num_features]
        ).to(self.device)
        dec_inp_label = batch_x[:, -self.label_len :, :].to(self.device)

        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        

        n = batch_x.size(0)
        t = torch.randint(
            low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)
        ).to(self.device)
        t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n]
        y_0_hat_batch, _ = self.cond_pred_model(batch_x, batch_x_mark, dec_inp, batch_y_mark_input)
        gx = self.cond_pred_model_g(batch_x) + EPS # (B, O, N)
        gx = torch.ones_like(gx).to(self.device)
        loss1 = (y_0_hat_batch - batch_y).square().mean()
        loss2 = (torch.sqrt(gx)- torch.sqrt(y_sigma)).square().mean()
        
        
        # loss_vae = log_normal(batch_y, y_0_hat_batch, torch.from_numpy(np.array(1)))
        # loss_vae_all = loss_vae + self.k_z * KL_loss
        # y_0_hat_batch = z_sample
        y_T_mean = y_0_hat_batch
        e = torch.randn_like(batch_y).to(self.device)


        
        
        
        forward_noise = cal_forward_noise(self.model.betas_tilde, self.model.betas_bar, gx, gx, t)
        noise = e * torch.sqrt(forward_noise)
        sigma_tilde = cal_sigma_tilde(self.model.alphas, self.model.alphas_cumprod, self.model.alphas_cumprod_sum, 
                                      self.model.alphas_cumprod_prev, self.model.alphas_cumprod_sum_prev, 
                                      self.model.betas_tilde_m_1, self.model.betas_bar_m_1, gx, gx, t)

        y_t_batch = q_sample(batch_y, y_T_mean, self.model.alphas_bar_sqrt,
                                self.model.one_minus_alphas_bar_sqrt, t, noise=noise)
        
        output, sigma_theta = self.model(batch_x, batch_x_mark, y_t_batch, y_0_hat_batch, gx, t)
        sigma_theta = sigma_theta + EPS
        
        # loss = (e[:, -self.args.pred_len:, :] - output[:, -self.args.pred_len:, :]).square().mean()
        # kl_loss = ((e - output)/sigma_theta).square().mean() + (y_sigma/sigma_theta).mean() - torch.log(y_sigma/sigma_theta).mean()
        # kl_loss = ((e - output)).square().mean() + (y_sigma/sigma_theta).mean() - torch.log(y_sigma/sigma_theta).mean()
        # kl_loss = ((1/sigma_tilde)*(sigma_tilde.sqrt() * e - sigma_theta.sqrt()*output)).square().mean() + (sigma_tilde/sigma_theta).mean() - torch.log(sigma_tilde/sigma_theta).mean()
        # kl_loss = ((1/sigma_theta)*(sigma_tilde.sqrt() * e - sigma_theta.sqrt()*output)).square().mean() + (sigma_tilde/sigma_theta).mean() - torch.log(sigma_tilde/sigma_theta).mean()
        kl_loss = ((e -output)).square().mean() + (sigma_tilde/sigma_theta).mean() - torch.log(sigma_tilde/sigma_theta).mean()
        loss = kl_loss + loss1 + loss2 
        # print(kl_loss.item(), loss1.item() , loss2.item())
        # /sigma_theta
        return loss


    def _process_val_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        b = batch_x.shape[0]
        gen_y_by_batch_list = [[] for _ in range(self.diffusion_steps + 1)]
        y_se_by_batch_list = [[] for _ in range(self.diffusion_steps + 1)]
        minisample = self.diffusion_config.testing.minisample
        
        batch_y_mark_input = torch.concat([batch_x_mark[:, -self.label_len:, :], batch_y_mark], dim=1)

        dec_inp_pred = torch.zeros(
            [batch_x.size(0), self.pred_len, self.dataset.num_features]
        ).to(self.device)
        dec_inp_label = batch_x[:, -self.label_len :, :].to(self.device)
        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)


        def store_gen_y_at_step_t(config, config_diff, idx, y_tile_seq):
            """
            Store generated y from a mini-batch to the array of corresponding time step.
            """
            current_t = self.diffusion_steps - idx
            gen_y = y_tile_seq[idx].reshape(b,
                                            # int(config_diff.testing.n_z_samples / config_diff.testing.n_z_samples_depart),
                                            minisample,
                                            (config.pred_len),
                                            config.c_out).cpu()
            # directly modify the dict value by concat np.array instead of append np.array gen_y to list
            # reduces a huge amount of memory consumption
            if len(gen_y_by_batch_list[current_t]) == 0:
                gen_y_by_batch_list[current_t] = gen_y.detach().cpu()
            else:
                gen_y_by_batch_list[current_t] = torch.concat([gen_y_by_batch_list[current_t], gen_y], dim=0).detach().cpu()
            return gen_y



        n = batch_x.size(0)
        t = torch.randint(
            low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)
        ).to(self.device)
        t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n]
        
        y_0_hat_batch, _ = self.cond_pred_model(batch_x, batch_x_mark, dec_inp,batch_y_mark_input)
        gx = self.cond_pred_model_g(batch_x)
        # gx = torch.ones_like(gx).to(self.device)
        
        preds = []
        for i in range(self.diffusion_config.testing.n_z_samples //minisample):
            repeat_n = int(minisample)
            y_0_hat_tile = y_0_hat_batch.repeat(repeat_n, 1, 1, 1)
            y_0_hat_tile = y_0_hat_tile.transpose(0, 1).flatten(0, 1).to(self.device)
            y_T_mean_tile = y_0_hat_tile
            x_tile = batch_x.repeat(repeat_n, 1, 1, 1)
            x_tile = x_tile.transpose(0, 1).flatten(0, 1).to(self.device)

            x_mark_tile = batch_x_mark.repeat(repeat_n, 1, 1, 1)
            x_mark_tile = x_mark_tile.transpose(0, 1).flatten(0, 1).to(self.device)

            gx_tile = gx.repeat(repeat_n, 1, 1, 1)
            gx_tile = gx_tile.transpose(0, 1).flatten(0, 1).to(self.device)
            gen_y_box = []
            for _ in range(self.diffusion_config.testing.n_z_samples_depart):
                for _ in range(self.diffusion_config.testing.n_z_samples_depart):
                    y_tile_seq = p_sample_loop_pe(self.model, x_tile, x_mark_tile, y_0_hat_tile, gx_tile, y_T_mean_tile,
                                                self.model.num_timesteps,
                                                self.model.alphas, self.model.one_minus_alphas_bar_sqrt,
                                                self.model.alphas_cumprod, self.model.alphas_cumprod_sum,
                                                self.model.alphas_cumprod_prev, self.model.alphas_cumprod_sum_prev,
                                                self.model.betas_tilde, self.model.betas_bar,
                                                self.model.betas_tilde_m_1, self.model.betas_bar_m_1,
                                                )
                gen_y = store_gen_y_at_step_t(config=self.model.args,
                                                config_diff=self.diffusion_config,
                                                idx=self.model.num_timesteps, y_tile_seq=y_tile_seq)
                gen_y_box.append(gen_y.detach().cpu())
            outputs = torch.concat(gen_y_box, dim=1)

            f_dim = -1 if self.args.features == 'MS' else 0
            
            outputs = outputs[:, :, -self.pred_len:, f_dim:] # B, S, O, N

            pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()

            preds.append(pred.detach().cpu()) # numberof_testbatch,  B, S, O, N
            # trues.append(true) # numberof_testbatch, B, T, N
            
        preds = torch.concat(preds, dim=1)
        batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device) # B, T, N

        outs = preds.permute(0, 2, 3, 1)
        assert (outs.shape[1], outs.shape[2], outs.shape[3]) == (self.pred_len, self.dataset.num_features, self.diffusion_config.testing.n_z_samples)
        return outs, batch_y

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

        # print("Pretrain ....")
        # self.pretrain()
        
        
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
            self.early_stopper(val_result['crps'], model={'model':self.model, 'cond_pred_model':self.cond_pred_model, 'cond_pred_model_g':self.cond_pred_model_g})

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
    fire.Fire(NsDiffForecast)
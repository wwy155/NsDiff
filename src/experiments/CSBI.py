import argparse
from dataclasses import dataclass, field
import sys
from typing import List, Dict
import os

import torch
from dataclasses import dataclass, asdict, field
from torch_timeseries.nn.embedding import freq_map
from src.nn.csbi_net import build_transformerv5
from src.models.CSDI import CSDI_Forecasting
from src.experiments.prob_forecast import ProbForecastExp
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from tqdm import tqdm
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible
import time
# import multiprocessing
import torch.multiprocessing as mp
from ml_collections import config_dict
from torch_timeseries.utils.parse_type import parse_type
from torch_timeseries.dataloader import SlidingWindowTS, ETTHLoader, ETTMLoader

import numpy as np
import torch.distributed as dist
import torch
from tqdm import tqdm
import concurrent.futures
import wandb
import  src.nn.csbi_sde as sde
import  src.nn.csbi_policy as policy
import  src.nn.csbi_data as data
import  src.nn.csbi_util as util
import  src.nn.csbi_loss as sample_e
import yaml
from torch.optim.adam import Adam
from torch_timeseries.utils.early_stop import EarlyStopping


class CSBIEarlyStopping(EarlyStopping):
    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model['z_f'].state_dict(), os.path.join(self.path, 'z_f.pth'))
        torch.save(model['z_b'].state_dict(),os.path.join(self.path, 'z_b.pth'))
        self.val_loss_min = val_loss
        
        
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def update_metrics(preds, truths, metrics):
    """Function to update metrics in a separate process."""
    metrics.update(preds, truths)


def freeze_policy(policy):
    for p in policy.parameters():
        p.requires_grad = False
    policy.eval()
    return policy

def activate_policy(policy):
    for p in policy.parameters():
        p.requires_grad = True
    policy.train()
    return policy


def compute_div_gz_imputation(
        opt, dyn, ts, xs,
        obs_data, obs_mask, cond_mask, gt_mask,
        policy, return_zs=False):
    assert policy.direction == 'backward'

    if getattr(opt, policy.direction + '_net') in ['Transformerv2', 'Transformerv4', 'Transformerv5']:
        cond_obs =  obs_data * cond_mask
        noisy_target = (1-cond_mask) * xs  # no big difference if using  target_mask * x
        total_input = torch.cat([cond_obs, noisy_target], dim=1)
        diff_input = (total_input, cond_mask)

    elif getattr(opt, policy.direction + '_net') == 'Transformerv3':
        cond_obs = cond_mask * obs_data
        noisy_target = (1-cond_mask) * xs
        total_input = cond_obs + noisy_target
        diff_input = (total_input, cond_mask)

    else:
        diff_input = xs

    zs = policy(diff_input, ts)
    g_ts = dyn.g(ts)

    # g_ts = g_ts[:,None,None,None] if util.is_image_dataset(opt) else g_ts[:,None]
    if opt.problem_name in ['mnist','cifar10','celebA32','celebA64']:
        g_ts = g_ts[:,None,None,None]  # (B) (B,1,1,1)
    elif opt.problem_name in ['gmm','checkerboard', 'moon-to-spiral']:
        g_ts = g_ts[:,None]  # (B) (B,1)
    elif opt.problem_name in ['sinusoid', 'pm25', 'tba2017', 'physio',
        'exchange_rate_nips', 'solar_nips', 'electricity_nips', ]:
        g_ts = g_ts[:,None,None,None]  # (B) (B,1,1,1)
    else:
        raise NotImplementedError('New dataset.')

    gzs = g_ts * zs

    # From torch doc: create_graph if True, graph of the derivative will be constructed, allowing
    # to compute higher order derivative products. As the target equation involves the gradient,
    # so we need to compute the gradient (over model parameters) of gradient (over data x).
    if opt.num_hutchinson_samp == 1:
        e = sample_e(opt, xs)
        e_dzdx = torch.autograd.grad(gzs, xs, e, create_graph=True)[0]
        div_gz = e_dzdx * e
        # approx_div_gz = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    else:
        # Unit test see: Hutchinson-Test-Sinusoid.ipynb
        div_gz = 0
        for hut_id in range(opt.num_hutchinson_samp):
            e = sample_e(opt, xs)
            e_dzdx = torch.autograd.grad(gzs, xs, e, create_graph=True, retain_graph=True)[0]
            div_gz += e_dzdx * e
        div_gz = div_gz / opt.num_hutchinson_samp

    return [div_gz, zs] if return_zs else div_gz




def compute_sb_nll_alternate_imputation_train(
    opt, dyn, ts, xs, zs_impt,
    obs_data, obs_mask, cond_mask, gt_mask,
    policy_opt, return_z=False):
    """ Implementation of Eq (18,19) in our main paper.
    """
    assert opt.train_method in ['alternate', 'alternate_backward',
        'alternate_backward_imputation', 'alternate_backward_imputation_v2',
        'alternate_imputation', 'alternate_imputation_v2']
    assert xs.requires_grad
    assert not zs_impt.requires_grad

    batch_x = opt.train_bs_x
    batch_t = opt.train_bs_t

    with torch.enable_grad():
        div_gz, zs = compute_div_gz_imputation(opt, dyn, ts, xs,
            obs_data, obs_mask, cond_mask, gt_mask,
            policy_opt, return_zs=True)

        if opt.backward_net in ['Transformerv2', 'Transformerv3', 'Transformerv4', 'Transformerv5']:
            loss_mask = obs_mask - cond_mask
        else:
            loss_mask = obs_mask

        zs  = zs * loss_mask
        zs_impt = zs_impt * loss_mask
        div_gz = div_gz * loss_mask
        loss = zs*(0.5*zs + zs_impt) + div_gz
        # print('div_gz', torch.sum(div_gz * dyn.dt) / batch_x / batch_t)
        loss = torch.sum(loss * dyn.dt) / batch_x / batch_t  # sum over x_dim and T, mean over batch
    return loss, zs if return_z else loss

@dataclass
class CSBIParameters:
    zero_out_last_layer:bool = True
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
class CSBIForecast(ProbForecastExp, CSBIParameters):
    model_type: str = "CSBI"
    def _init_model(self):
        
        config = None
        with open('./configs/csbi.yaml', encoding="utf-8") as f:
            result =  yaml.load(f, Loader=yaml.UnsafeLoader)
            config = config_dict.ConfigDict(result)
            config = result
            

        opt = dict2namespace(config)
        
        
        def build_boundary_distribution(opt):
            print(util.magenta("build boundary distribution..."))

            opt.data_dim = [1, self.dataset.num_features, (self.windows + self.pred_len) ]
            opt.input_size = (self.dataset.num_features, self.windows + self.pred_len)
            prior = data.build_prior_sampler(opt, opt.samp_bs)
            pdata = self.build_data_sampler(self.windows, self.pred_len, self.dataset.num_features, self.train_loader, self.device)

            return pdata, prior
        
        
        opt.device = self.device
        self.ts = torch.linspace(opt.t0, opt.T, opt.interval).to(self.device)
        self.p, self.q = build_boundary_distribution(opt)  # p: data  q: prior.
        # opt.model_configs[self.net_name]['channels'] = self.dataset.num_features
        # net = build_transformerv5(opt.model_configs[self.net_name], opt.interval, self.zero_out_last_layer)
        
        # build dynamics, forward (z_f) and backward (z_b) policies
        self.dyn = sde.build(opt, self.p, self.q)
        self.z_f = policy.build(opt, self.dyn, 'forward').float()  # p -> q
        self.z_b = policy.build(opt, self.dyn, 'backward').float() # q -> p
        
        
        self.gt_mask = torch.concat([
                torch.ones(size=(self.windows, self.dataset.num_features)),
                torch.zeros(size=(self.pred_len, self.dataset.num_features)),
            ]).to(self.device).bool()
        
        self.observation_mask = ~self.gt_mask
        self.cond_mask = self.observation_mask

        self.opt = opt

        direction = 'backward'
        self.policy_opt, self.policy_impt = {
            'forward':  [self.z_f, self.z_b], # train forwad,   sample from backward
            'backward': [self.z_b, self.z_f], # train backward, sample from forward
        }.get(direction)

        # train_ts = self.ts
        # if self.opt.use_corrector and self.opt.backward_net in ['Transformerv2', 'Transformerv4',
        #     'Transformerv5']:
        #     def corrector(x, t):
        #         total_input = torch.cat([torch.zeros_like(x), x], dim=1)
        #         diff_input = (total_input, torch.zeros_like(x))
        #         return self.policy_opt(x,t) + self.policy_impt(diff_input, t)
        # elif self.opt.use_corrector and self.opt.backward_net in ['Transformerv3']:
        #     def corrector(x, t):
        #         diff_input = (x, torch.zeros_like(x))
        #         return self.policy_opt(x,t) + self.policy_impt(diff_input, t)
        # elif self.opt.use_corrector:
        #     def corrector(x, t):
        #         return self.policy_opt(x,t) + self.policy_impt(x,t)
        # else:
        #     corrector = None
        # with torch.no_grad():
        #     xs, zs, _ = self.dyn.sample_traj(train_ts, self.policy_impt, corrector=corrector)
        # import pdb;pdb.set_trace()
 



    def _init_optimizer(self):
        self.model_optim = parse_type(self.optm_type, globals=globals())(
            [{'params': self.z_f.parameters()}, {'params': self.z_b.parameters()}], 
            lr=self.lr, 
        )


    def build_data_sampler(self, windows, pred_len, num_features, train_loader, device):
        class DataSampler:
            def __init__(self):
            
                self.gt_mask = torch.concat([
                        torch.ones(size=(windows, num_features)),
                        torch.zeros(size=(pred_len, num_features)),
                    ]).to(device).bool()
                self.observation_mask = ~self.gt_mask
                self.cond_mask = self.observation_mask

            def sample(self, num_samples=None, return_mask=False, return_all_mask=False):
                batch_x, batch_y, _, _, batch_x_date_enc, batch_y_date_enc = next(iter(train_loader))
                batch_input = {
                    "observed_data": torch.concat([batch_x, batch_y], dim=1).float(),
                    "observed_mask": self.observation_mask.unsqueeze(0).expand(batch_x.shape[0], -1, -1),
                    "timepoints": torch.concat([batch_x_date_enc, batch_y_date_enc], dim=1)[:, :, 0],
                    "gt_mask": self.gt_mask.unsqueeze(0).expand(batch_x.shape[0], -1, -1),
                }
                if return_all_mask:
                    return (batch_input["observed_data"].float().transpose(1,2).unsqueeze(1).to(device),
                            batch_input["observed_mask"].transpose(1,2).unsqueeze(1).to(device),
                            batch_input["gt_mask"].transpose(1,2).unsqueeze(1).to(device))
                elif return_mask:
                    return (
                            batch_input["observed_mask"].transpose(1,2).unsqueeze(1).to(device),
                            batch_input["gt_mask"].transpose(1,2).unsqueeze(1).to(device))
                else:
                    return batch_input["observed_data"].float().transpose(1,2).unsqueeze(1).to(device)
                    
                    
        return DataSampler()


    def _process_train_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        pass
    #     # inputs:
    #     # batch_x: (B, T, N)
    #     # batch_y: (B, O, N)
    #     # ouputs:
    #     # - pred: (B, N)/(B, O, N)
    #     # - label: (B, N)/(B, O, N)
        
        
    #     policy_opt, policy_impt = self.z_b, self.z_f
        
    #     zs = policy_opt(xs, ts)
    #     g_ts = self.dyn.g(ts)
        

    #     observed_mask = self.observation_mask.unsqueeze(0).expand(batch_x.shape[0], -1, -1),
    #     gt_mask = self.gt_mask.unsqueeze(0).expand(batch_x.shape[0], -1, -1)
    #     cond_mask = gt_mask
    #     loss_mask = observed_mask
        
        
    #     gzs = g_ts * zs
        
        
    #     e = torch.rand_like(xs)
    #     e_dzdx = torch.autograd.grad(gzs, xs, e, create_graph=True)[0]
    #     div_gz = e_dzdx * e
        
    
    
    
    #     zs  = zs * loss_mask
    #     zs_impt = zs_impt * loss_mask
    #     div_gz = div_gz * loss_mask
    #     loss = zs*(0.5*zs + zs_impt) + div_gz
    #     # print('div_gz', torch.sum(div_gz * dyn.dt) / batch_x / batch_t)
    #     loss = torch.sum(loss * dyn.dt) / batch_x / batch_t  # sum over x_dim and T, mean over batch

    
    
    #     noise, pred_noise = self.model(batch_input, is_train=self.num_samples)
        
        
    
    
    
    #     return noise, pred_noise



    # def _init_data_loader(self, shuffle=True, fast_test=True, fast_val=True):
        
    #     self._init_dataset()
        
    #     self.scaler = parse_type(self.scaler_type, globals=globals())()
    #     if self.dataset_type[0:3] == "ETT":
    #         if self.dataset_type[0:4] == "ETTh":
    #             self.dataloader = ETTHLoader(
    #                 self.dataset,
    #                 self.scaler,
    #                 window=self.windows,
    #                 horizon=self.horizon,
    #                 steps=self.pred_len,
    #                 shuffle_train=shuffle,
    #                 freq=self.dataset.freq,
    #                 batch_size=self.batch_size,
    #                 num_worker=self.num_worker,
    #                 fast_test=fast_test,
    #                 fast_val=fast_val,
    #             )
    #         elif  self.dataset_type[0:4] == "ETTm":
    #             self.dataloader = ETTMLoader(
    #                 self.dataset,
    #                 self.scaler,
    #                 window=self.windows,
    #                 horizon=self.horizon,
    #                 steps=self.pred_len,
    #                 shuffle_train=shuffle,
    #                 freq=self.dataset.freq,
    #                 batch_size=self.batch_size,
    #                 num_worker=self.num_worker,
    #                 fast_test=fast_test,
    #                 fast_val=fast_val,
    #             )
    #     else:
    #         self.dataloader = SlidingWindowTS(
    #             self.dataset,
    #             self.scaler,
    #             window=self.windows,
    #             horizon=self.horizon,
    #             steps=self.pred_len,
    #             scale_in_train=True,
    #             shuffle_train=shuffle,
    #             freq=self.dataset.freq,
    #             batch_size=self.batch_size,
    #             train_ratio=self.train_ratio,
    #             test_ratio=self.test_ratio,
    #             num_worker=self.num_worker,
    #             fast_test=fast_test,
    #             fast_val=fast_val,
    #         )

    #     self.train_loader, self.val_loader, self.test_loader = (
    #         self.dataloader.train_loader,
    #         self.dataloader.val_loader,
    #         self.dataloader.test_loader,
    #     )
    #     self.train_steps = len(self.train_loader.dataset)
    #     self.val_steps = len(self.val_loader.dataset)
    #     self.test_steps = len(self.test_loader.dataset)

    #     print(f"train steps: {self.train_steps}")
    #     print(f"val steps: {self.val_steps}")
    #     print(f"test steps: {self.test_steps}")
        
    @torch.no_grad()
    def imputation(
            self,
            opt,
            x_cond,
            obs_mask,
            cond_mask,
            target_mask,
            num_samples=None):
        """Conditional inference."""
        # assert opt.backward_net in ['Transformerv2', 'Transformerv3']
        ts_reverse = torch.flip(self.ts, dims=[0])  # Backward diffusion.
        K, L = opt.input_size
        B = x_cond.shape[0]

        if opt.use_corrector and opt.backward_net in ['Transformerv2', 'Transformerv4',
            'Transformerv5']:
            def corrector(x, t, x_cond, cond_mask):
                cond_obs = cond_mask * x_cond
                noisy_target = (1-cond_mask) * x  # no big difference if using  target_mask * x
                total_input = torch.cat([cond_obs, noisy_target], dim=1)
                diff_input = (total_input, cond_mask)
                return self.z_f(x,t) + self.z_b(diff_input, t)
        elif opt.use_corrector and opt.backward_net in ['Transformerv3']:
            def corrector(x, t, x_cond, cond_mask):
                cond_obs = cond_mask * x_cond
                noisy_target = (1-cond_mask) * x  # no big difference if using  target_mask * x
                total_input = cond_obs + noisy_target
                diff_input = (total_input, cond_mask)
                return self.z_f(x,t) + self.z_b(diff_input, t)
        elif opt.use_corrector:
            def corrector(x, t, x_cond=None, cond_mask=None):
                return self.z_f(x,t) + self.z_b(x,t)
        else:
            corrector = None

        policy = freeze_policy(self.z_b)
        imputed_samples = torch.zeros(B, num_samples, K, L).to(opt.device)

        for i in tqdm(range(num_samples), ncols=100, file=sys.stdout):
            current_sample = torch.randn_like(x_cond)

            for idx, t in enumerate(ts_reverse):
                t_idx = len(ts_reverse)-idx-1  # backward ids.
                if opt.backward_net in ['Transformerv2', 'Transformerv4', 'Transformerv5']:
                    # Only need conditional mask, no need obs_mask or target_mask.
                    cond_obs = cond_mask * x_cond
                    noisy_target = (~cond_mask) * current_sample  # not target_mask here.
                    total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                    diff_input = (total_input, cond_mask)
                elif opt.backward_net == 'Transformerv3':
                    # Only need conditional mask, no need obs_mask or target_mask.
                    cond_obs = cond_mask * x_cond
                    noisy_target = (1 - cond_mask) * current_sample  # not target_mask here.
                    total_input = cond_obs + noisy_target  # (B,1,K,L)
                    diff_input = (total_input, cond_mask)
                else:
                    diff_input = cond_mask * x_cond + (1 - cond_mask) * current_sample

                z = policy(diff_input, t)
                g = self.dyn.g(t)
                f = self.dyn.f(current_sample,t,'backward')
                dt = self.dyn.dt
                dw = self.dyn.dw(current_sample,dt)
                current_sample = current_sample + (f + g * z)*dt
                if t > 0:
                    current_sample += g*dw

                # Apply corrector.
                if opt.use_corrector:
                    _t=t if idx==ts_reverse.shape[0]-1 else ts_reverse[idx+1]
                    current_sample = self.dyn.corrector_langevin_imputation_update(
                        _t, current_sample, corrector, x_cond, cond_mask, denoise_xT=False)

            imputed_samples[:, i] = current_sample.squeeze(1).detach()
        return imputed_samples



    def _process_val_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        
        
        samp_t_idx = torch.randint(self.opt.interval, (batch_x.shape[0], self.opt.train_bs_t,)) # (t)
        ts = self.ts[samp_t_idx].detach()
        ts = ts.reshape(batch_x.shape[0]*self.opt.train_bs_t) # B, t
        x0, obs_mask, gt_mask = torch.concat([batch_x, batch_y], dim=1).float().transpose(1,2).unsqueeze(1), \
                                    self.observation_mask.unsqueeze(0).expand(batch_x.shape[0], -1, -1).transpose(1,2).unsqueeze(1), \
                                        self.gt_mask.unsqueeze(0).expand(batch_x.shape[0], -1, -1).transpose(1,2).unsqueeze(1),
        cond_mask = ~obs_mask# self.get_randmask(obs_mask, miss_ratio=self.opt.rand_mask_miss_ratio, rank=self.opt.rand_mask_rank)  # Apply random mask here.
        samples = self.imputation(self.opt, x0, obs_mask, cond_mask, gt_mask, 100)
        return samples[:, :, :, -self.pred_len:].permute(0, 3, 2, 1), batch_y
    
    def _evaluate(self, dataloader):
        self.z_b.eval()
        self.z_f.eval()
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
                    
                # update_metrics(preds.contiguous().cpu().detach(), truths.contiguous().cpu().detach(), self.metrics)
                # if isinstance(preds, np.ndarray):
                #     results.append(self.task_pool.apply_async(update_metrics, (preds, truths, self.metrics)))
                # else:
                results.append(self.task_pool.apply_async(update_metrics, (preds.contiguous().cpu().detach(), truths.contiguous().cpu().detach(), self.metrics)))
                
                progress_bar.update(batch_x.shape[0])

        for result in results:
            result.get()  # Ensure the metric update is finished

        result = {name: float(metric.compute()) for name, metric in self.metrics.items()}
        return result
    


    # def _evaluate(self, dataloader):
    #     self.metrics.reset()
    #     results = []
    #     with tqdm(total=len(dataloader.dataset)) as progress_bar:
    #         for batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc in dataloader:
    #             batch_size = batch_x.size(0)
    #             origin_x = origin_x.to(self.device)
    #             origin_y = origin_y.to(self.device)
    #             batch_x = batch_x.to(self.device).float()
    #             batch_y = batch_y.to(self.device).float()
    #             batch_x_date_enc = batch_x_date_enc.to(self.device).float()
    #             batch_y_date_enc = batch_y_date_enc.to(self.device).float()
    #             preds, truths = self._process_val_batch(
    #                 batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
    #             )
    #             origin_y = origin_y.to(self.device)
    #             if self.invtrans_loss:
    #                 preds = self.scaler.inverse_transform(preds)
    #                 truths = origin_y
                    
    #             # update_metrics(preds.contiguous().cpu().detach(), truths.contiguous().cpu().detach(), self.metrics)
    #             # if isinstance(preds, np.ndarray):
    #             #     results.append(self.task_pool.apply_async(update_metrics, (preds, truths, self.metrics)))
    #             # else:
    #             results.append(self.task_pool.apply_async(update_metrics, (preds.contiguous().cpu().detach(), truths.contiguous().cpu().detach(), self.metrics)))
                
    #             progress_bar.update(batch_x.shape[0])

    #     for result in results:
    #         result.get()  # Ensure the metric update is finished

    #     result = {name: float(metric.compute()) for name, metric in self.metrics.items()}
    #     return result
    
    
    def _train(self):
        with torch.enable_grad(), tqdm(total=len(self.train_loader.dataset)) as progress_bar:
            self.z_b.train()
            self.z_f.eval()
            
            # train in backward
            policy_opt, policy_impt = self.z_b, self.z_f
            
            policy_impt = freeze_policy(policy_impt)
            policy_opt = activate_policy(policy_opt)

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
                
                batch_t = 5
                # prepare data
                origin_y = origin_y.to(self.device).float()
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()
                
                # batch_input = {
                #     "observed_data": torch.concat([batch_x, batch_y], dim=1).float(),
                #     "observed_mask": self.observation_mask.unsqueeze(0).expand(batch_x.shape[0], -1, -1),
                #     "timepoints": torch.concat([batch_x_date_enc, batch_y_date_enc], dim=1)[:, :, 0],
                #     "gt_mask": self.gt_mask.unsqueeze(0).expand(batch_x.shape[0], -1, -1),
                # }
                samp_t_idx = torch.randint(self.opt.interval, (batch_x.shape[0], self.opt.train_bs_t,)) # (t)
                ts = self.ts[samp_t_idx].detach()
                ts = ts.reshape(batch_x.shape[0]*self.opt.train_bs_t) # B, t
                x0, obs_mask, gt_mask = torch.concat([batch_x, batch_y], dim=1).float().transpose(1,2).unsqueeze(1), \
                                            self.observation_mask.unsqueeze(0).expand(batch_x.shape[0], -1, -1).transpose(1,2).unsqueeze(1), \
                                                self.gt_mask.unsqueeze(0).expand(batch_x.shape[0], -1, -1).transpose(1,2).unsqueeze(1),
                
                
                # x0 = x0.reshape(x0.shape[0], -1)
                # obs_mask = obs_mask.reshape(x0.shape[0], -1)
                # gt_mask = gt_mask.reshape(x0.shape[0], -1)
                
                
                
                
                policy = activate_policy(self.z_b)
                compute_xs_label = sde.get_xs_label_computer(self.opt, self.ts)
                samp_t_idx = torch.randint(self.opt.interval, (batch_x.shape[0], batch_t))
                ts = self.ts[samp_t_idx].detach()  # (batch_x*batch_t)
                ts = ts.reshape(batch_x.shape[0]*batch_t)
                xs, label, label_scale = compute_xs_label(x0=x0, samp_t_idx=samp_t_idx, return_scale=True)
                xs = util.flatten_dim01(xs)  # (batch, T, xdim) --> (batch*T, xdim)   (B*t, L*D)
                x0 = x0.unsqueeze(1).repeat(1,batch_t,1,1,1) # (B*t, L*D)
                x0 = util.flatten_dim01(x0)
                obs_mask = obs_mask.unsqueeze(1).repeat(1,batch_t,1,1,1)
                obs_mask = util.flatten_dim01(obs_mask)
                assert xs.shape[0] == ts.shape[0]

                cond_mask = ~obs_mask# self.get_randmask(obs_mask, miss_ratio=self.opt.rand_mask_miss_ratio, rank=self.opt.rand_mask_rank)  # Apply random mask here.
                target_mask = ~gt_mask #obs_mask - cond_mask
                
                cond_obs = cond_mask * x0
                noisy_target = (~cond_mask) * xs
                total_input = torch.cat([cond_obs, noisy_target], dim=1)
                diff_input = (total_input, cond_mask)
                loss_mask = target_mask  # don't use target_mask for fixed cond_mask.
                
                predicted = policy(diff_input, ts)
                label = label.reshape_as(predicted)
                residual = (label - predicted) * obs_mask
                num_eval = loss_mask.sum()
                loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
                loss.backward()
                
                
                # torch.randint(self.opt.interval, (opt.train_bs_x*opt.train_bs_t,))

                # import pdb;pdb.set_trace()
                # cond_mask = obs_mask
                # (xs, zs, x_term, obs_data, obs_mask, cond_mask, gt_mask
                #     ) = self.dyn.sample_traj_forecasting_forward(train_ts, policy_impt, obs_data, obs_mask, gt_mask, cond_mask, corrector=corrector)
                # train_xs = xs.detach()
                # train_zs = zs.detach()
                
                    
                # xs      = util.flatten_dim01(train_xs)
                # zs_impt = util.flatten_dim01(train_zs)
                # obs_data_ = util.flatten_dim01(obs_data_)
                # obs_mask_ = util.flatten_dim01(obs_mask_)
                # cond_mask_ = util.flatten_dim01(cond_mask_)
                # gt_mask_ = util.flatten_dim01(cond_mask_)

                
                
                # loss, zs = compute_sb_nll_alternate_imputation_train(self.opt, 
                #                                           self.dyn,
                #                                           self.ts,
                #                                           xs,
                #                                           zs_impt,
                #                                           obs_data_, obs_mask_, cond_mask_, gt_mask_,
                #                                             policy_opt, return_z=True
                #                                           )
                
                
                # # pred, true = self._process_train_batch(
                # #     batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                # # )
                
                # if self.invtrans_loss:
                #     pred = self.scaler.inverse_transform(pred)
                #     true = origin_y
                # loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.z_b.parameters(), self.max_grad_norm
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
            self.z_b.eval()
            self.z_f.eval()
            return train_loss

    def _resume_run(self, seed):
        # only train loader rshould be checkedpoint to keep the validation and test consistency
        check_point = torch.load(self.run_checkpoint_filepath, map_location=self.device)

        self.z_f.load_state_dict(check_point["z_f"])
        self.z_b.load_state_dict(check_point["z_b"])
        self.model_optim.load_state_dict(check_point["optimizer"])
        self.current_epoch = check_point["current_epoch"]

        self.early_stopper.set_state(check_point["early_stopping"])

    def _load_best_model(self):
        self.z_b.load_state_dict(
            torch.load(os.path.join(self.run_save_dir, 'z_b.pth'), map_location=self.device)
        )
        self.z_f.load_state_dict(
            torch.load(os.path.join(self.run_save_dir, 'z_f.pth'), map_location=self.device)
        )

    def _setup_early_stopper(self):
        self.best_zb_checkpoint_filepath = os.path.join(
            self.run_save_dir, "z_b.pth"
        )
        self.best_zf_checkpoint_filepath = os.path.join(
            self.run_save_dir, "z_f.pth"
        )
        self.early_stopper = CSBIEarlyStopping(
            self.patience, verbose=True, path=self.run_save_dir
        )


    def _save_run_check_point(self, seed):

        if not os.path.exists(self.run_save_dir):
            os.makedirs(self.run_save_dir)
        
        print(f"Saving run checkpoint to '{self.run_save_dir}'.")

        self.run_state = {
            "z_f": self.z_f.state_dict(),
            "z_b": self.z_b.state_dict(),
            "current_epoch": self.current_epoch,
            "optimizer": self.model_optim.state_dict(),
            "rng_state": torch.get_rng_state(),
            "early_stopping": self.early_stopper.get_state(),
        }

        torch.save(self.run_state, f"{self.run_checkpoint_filepath}")
        print("Run state saved ... ")


    def run(self, seed=42) -> Dict[str, float]:
        
        if self._use_wandb() and not self._init_wandb(self.project, seed): return {}
        
        self._setup_run(seed)
        if self._check_run_exist(seed):
            self._resume_run(seed)

        self._run_print(f"run : {self.current_run} in seed: {seed}")

        parameter_tables, model_parameters_num = count_parameters(self.policy_opt)
        parameter_tables, model_parameters_num = count_parameters(self.policy_impt)
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
            self.early_stopper(val_result['crps'], model={'z_f':self.z_f, 'z_b':self.z_b})

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
    fire.Fire(CSBIForecast)
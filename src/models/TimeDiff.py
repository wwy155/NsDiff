

from typing import List, Optional, Tuple, Union
from typing import Any, Dict
from functools import partial
from inspect import isfunction
from tqdm import tqdm
import math
import torch
import torch.nn as nn
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn.modules import loss
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from src.nn.DDPM_CNNNet import *
from src.nn.diffusion_worker import *

from src.nn.dpm_sampler import DPMSolverSampler


class TimeDiff(nn.Module):
    
    def __init__(self, args):
        super(TimeDiff, self).__init__()

        self.args = args
        
        print(self.args)
        self.device = args.device

        self.seq_len = args.seq_len
        self.pred_len = args.pred_len

        self.input_size = args.num_vars
        self.diff_steps = args.diff_steps

        if args.UNet_Type == "CNN":
            u_net = CNN_DiffusionUnet(args, self.input_size, self.seq_len+self.pred_len, self.pred_len, self.diff_steps)

        self.diffusion_worker = Diffusion_Worker(args, u_net, self.diff_steps)

        if args.type_sampler == "none":
            pass
        elif args.type_sampler == "dpm":
            assert self.args.parameterization == "x_start"
            self.sampler = DPMSolverSampler(u_net, self.diffusion_worker)

        self.short_term_range = args.label_len # args.seq_len # self.pred_len # args.seq_len
        self.dlinear_model = nn.Linear(self.short_term_range, self.pred_len)

        self.norm_len = args.label_len

    def pretrain_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        self.diffusion_worker.eval()
        self.dlinear_model.train()

        # print("x_enc", np.shape(x_enc))
        outs = self.dlinear_model(x_enc.permute(0,2,1)[:,:,-self.short_term_range:]).permute(0,2,1)

        flag_smooth_linear_target = 0

        target = x_dec[:,-self.pred_len:,:]

        f_dim = -1 if self.args.features == 'MS' else 0

        if flag_smooth_linear_target == 1:
            target_ft = torch.fft.rfft(target, dim=1)
            B, L, K = np.shape(target_ft)
            out_ft = torch.zeros(np.shape(target_ft),  device=target.device, dtype=torch.cfloat)
            out_ft[:, :5, :] = target_ft[:, :5, :]
            target_out = torch.fft.irfft(out_ft, n=self.pred_len, dim=1)
            # print(np.shape(target_out))
            loss = F.mse_loss(outs[:,:,f_dim:], target_out[:,:,f_dim:])
        else:
            loss = F.mse_loss(outs[:,:,f_dim:], target[:,:,f_dim:])
        
        return loss 

    def train_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        self.diffusion_worker.train()
        self.dlinear_model.eval()
        
        linear_outputs = self.dlinear_model(x_enc.permute(0,2,1)[:,:,-self.short_term_range:]).permute(0,2,1)

        if self.args.use_window_normalization:
            seq_len = np.shape(x_enc)[1]
            
            mean_ = torch.mean(x_enc[:,-self.norm_len:,:], dim=1).unsqueeze(1)
            std_ = torch.ones_like(torch.std(x_enc, dim=1).unsqueeze(1))

            x_enc_i = (x_enc-mean_.repeat(1,seq_len,1))/(std_.repeat(1,seq_len,1)+0.00001)

            seq_len = np.shape(x_dec)[1]
            x_dec_i = (x_dec-mean_.repeat(1,seq_len,1))/(std_.repeat(1,seq_len,1)+0.00001)
                
            seq_len = np.shape(linear_outputs)[1]
            linear_outputs_i = (linear_outputs-mean_.repeat(1,seq_len,1))/(std_.repeat(1,seq_len,1)+0.00001)
        else:
            x_enc_i = x_enc
            x_dec_i = x_dec
            linear_outputs_i = linear_outputs
        x_past = x_enc_i
        x_future = x_dec_i[:,-self.args.pred_len:,:] # - linear_outputs_i

        x_past = x_past.permute(0,2,1)     # torch.Size([64, 30, 24])
        x_future = x_future.permute(0,2,1) # [bsz, fea, seq_len]

        x_past = torch.cat([x_past, linear_outputs_i.permute(0,2,1)], dim=-1)
        
        f_dim = -1 if self.args.features in ['MS'] else 0
        loss = self.diffusion_worker(x_future[:,f_dim:,:], x_past)
        return loss

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, sample_times=5):

        self.diffusion_worker.eval()
        self.dlinear_model.eval()

        if self.args.vis_ar_part:
            saved_dict = {}
            W = self.dlinear_model.weight.data.cpu().numpy()
            B = self.dlinear_model.bias.data.cpu().numpy()

            saved_dict["W"] = W
            saved_dict["B"] = B

        # print(">>>>>", np.shape(W), np.shape(B))
        # (168, 168) (168,)

        linear_outputs = self.dlinear_model(x_enc.permute(0,2,1)[:,:,-self.short_term_range:]).permute(0,2,1)

        if self.args.use_window_normalization:
            seq_len = np.shape(x_enc)[1]
            
            mean_ = torch.mean(x_enc[:,-self.norm_len:,:], dim=1).unsqueeze(1)
            std_ = torch.ones_like(torch.std(x_enc, dim=1).unsqueeze(1))

            x_enc_i = (x_enc-mean_.repeat(1,seq_len,1))/(std_.repeat(1,seq_len,1)+0.00001)

            seq_len = np.shape(x_dec)[1]
            x_dec_i = (x_dec-mean_.repeat(1,seq_len,1))/(std_.repeat(1,seq_len,1)+0.00001)
                
            seq_len = np.shape(linear_outputs)[1]
            linear_outputs_i = (linear_outputs-mean_.repeat(1,seq_len,1))/(std_.repeat(1,seq_len,1)+0.00001)
        else:
            x_enc_i = x_enc
            x_dec_i = x_dec
            linear_outputs_i = linear_outputs

        x_past = x_enc_i
        x_future = x_dec_i[:,-self.args.pred_len:,:] # - linear_outputs_i

        x_past = x_past.permute(0,2,1)     # torch.Size([64, 30, 24])
        x_future = x_future.permute(0,2,1) # [bsz, fea, seq_len]

        x_past = torch.cat([x_past, linear_outputs_i.permute(0,2,1)], dim=-1)

        B, nF, nL = np.shape(x_past)[0], self.input_size, self.pred_len
        if self.args.features in ['MS']:
            nF = 1
        shape = [nF, nL]
        
        all_outs = []
        for i in range(sample_times):
            start_code = torch.randn((B, nF, nL), device=self.device)
            
            if self.args.type_sampler == "none":
                f_dim = -1 if self.args.features in ['MS'] else 0
                outs_i = self.diffusion_worker.sample(x_future[:,f_dim:,:], x_past)
            else:
                samples_ddim, _ = self.sampler.sample(S=20,
                                             conditioning=x_past,
                                             batch_size=B,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=1.0,
                                             unconditional_conditioning=None,
                                             eta=0.,
                                             x_T=start_code)
                outs_i = samples_ddim.permute(0,2,1)

            if self.args.use_window_normalization:
                out_len = np.shape(outs_i)[1]
                outs_i = outs_i * std_.repeat(1,out_len,1) + mean_.repeat(1,out_len,1)

            all_outs.append(outs_i)
        all_outs = torch.stack(all_outs, dim=0)

        flag_return_all = True

        if flag_return_all:
            outs = all_outs.permute(1,0,2,3)
            f_dim = -1 if self.args.features in ['MS'] else 0
            outs = outs[:, :, -self.pred_len:, f_dim:] # - 0.4
        else:
            outs = all_outs.mean(0)
            f_dim = -1 if self.args.features == ['MS'] else 0
            outs = outs[:, -self.pred_len:, f_dim:] # - 0.4

        if self.args.use_window_normalization:
            
            out_len = np.shape(x_enc_i)[1]
            x_enc_i = x_enc_i * std_.repeat(1,out_len,1) + mean_.repeat(1,out_len,1)

            out_len = np.shape(x_dec_i)[1]
            x_dec_i = x_dec_i * std_.repeat(1,out_len,1) + mean_.repeat(1,out_len,1)

            # out_len = np.shape(linear_outputs_i)[1]
            # linear_outputs_i = linear_outputs_i * std_.repeat(1,out_len,1) + mean_.repeat(1,out_len,1)

        if self.args.vis_ar_part:
            # inp, predictied, output, linear output
            saved_dict["predicted"] = outs.detach().cpu().numpy()
            saved_dict["predicted_linear"] = linear_outputs.detach().cpu().numpy()
            saved_dict["history"] = x_enc.cpu().numpy()
            saved_dict["ground_truth"] = x_dec[:, -self.args.pred_len:, :].cpu().numpy()

            import pickle

            with open('AR_{}.pickle'.format(self.args.dataset_name), 'wb') as handle:
                pickle.dump(saved_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            raise Exception("Save the AR visualization.")

        return outs, x_enc[:,:,f_dim:], x_dec[:, -self.args.pred_len:, f_dim:], None, None





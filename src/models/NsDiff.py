import torch
import torch.nn as nn
from torch_timeseries.nn.embedding import DataEmbedding
import yaml
import argparse
from src.nn.tmdm_diffusion_utils import *
from src.layer.denoise import ConditionalGuidedModel
from src.utils.sigma import wv_sigma


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def compute_tilde_alpha(alpha: torch.Tensor) -> torch.Tensor:
    alpha = alpha.float()
    n = alpha.shape[0]
    tilde_alpha = torch.zeros_like(alpha)  
    
    for t in range(n):
        slice_t = alpha[:t+1].flip(dims=[0])
        cprod = torch.cumprod(slice_t, dim=0)
        tilde_alpha[t] = cprod.sum()
    return tilde_alpha


# def compute_tilde_alpha(alpha):
#     n = len(alpha)
#     tilde_alpha = np.zeros(n, dtype=float)
#     for t in range(n):
#         import pdb;pdb.set_trace()
#         slice_t = alpha[t::-1] 
#         cprod = np.cumprod(slice_t)
#         tilde_alpha[t] = cprod.sum()
    
#     return tilde_alpha

class NsDiff(nn.Module):
    """
    Vanilla Transformer
    """

    def __init__(self, configs, device):
        super(NsDiff, self).__init__()

        with open(configs.diffusion_config_dir, "r") as f:
            config = yaml.unsafe_load(f)
            diffusion_config = dict2namespace(config)

        diffusion_config.diffusion.timesteps = configs.timesteps
        
        self.args = configs
        self.device = device
        self.diffusion_config = diffusion_config

        self.model_var_type = diffusion_config.model.var_type
        self.num_timesteps = diffusion_config.diffusion.timesteps
        self.vis_step = diffusion_config.diffusion.vis_step
        self.num_figs = diffusion_config.diffusion.num_figs
        self.dataset_object = None

        betas = make_beta_schedule(schedule=diffusion_config.diffusion.beta_schedule, num_timesteps=self.num_timesteps,
                                   start=diffusion_config.diffusion.beta_start, end=diffusion_config.diffusion.beta_end)
        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        alphas_cumprod = alphas.to('cpu').cumprod(dim=0).to(self.device)
        self.alphas_cumprod = alphas_cumprod
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        
        # self.alphas_cumprod_sum = torch.cumsum(alphas_cumprod.flip(0), dim=0).flip(0)
        self.alphas_cumprod_sum = compute_tilde_alpha(alphas)
        
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if diffusion_config.diffusion.beta_schedule == "cosine":
            self.one_minus_alphas_bar_sqrt *= 0.9999  # avoid division by 0 for 1/sqrt(alpha_bar_t) during inference
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=self.device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_sum_prev = torch.cat(
            [torch.ones(1, device=self.device), self.alphas_cumprod_sum[:-1]], dim=0
        )

        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_mean_coeff_1 = (
                betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coeff_2 = (
                torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = posterior_variance
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        self.tau = None  # precision fo test NLL computation

        # CATE MLP
        self.diffussion_model = ConditionalGuidedModel(diffusion_config.diffusion.timesteps, configs.enc_in)

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.CART_input_x_embed_dim, configs.embed, configs.freq,
                                           configs.dropout)


    def forward(self, x, x_mark, y_t, y_0_hat, gx, t):
        enc_out = self.enc_embedding(x, x_mark) #  B, T, d_model
        dec_out, sigma = self.diffussion_model(enc_out, y_t, y_0_hat, gx, t)

        return dec_out, sigma

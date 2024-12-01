# -*-Encoding: utf-8 -*-

import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from src.nn.resnet import Res12_Quadratic
from src.nn.d3vae_diffusion import GaussianDiffusion, get_beta_schedule
from src.nn.d3vae_encoder import Encoder
from torch_timeseries.nn.embedding import DataEmbedding


class diffusion_generate(nn.Module):
    def __init__(self,sequence_length, beta_end, beta_schedule, scale, diff_steps ,  mult,  channel_mult,  prediction_length,  num_preprocess_blocks,  num_preprocess_cells,
                  num_channels_enc,  arch_instance,  num_latent_per_group,  num_channels_dec, 
                   num_postprocess_blocks,  num_postprocess_cells,  embedding_dimension,  hidden_size,  target_dim,  num_layers,  dropout_rate, groups_per_scale):
        super().__init__()
        """
        Two main parts are included, the coupled diffusion process is included in the GaussianDiffusion Module, and the bidirection model.
        """
        self.target_dim = target_dim
        self.input_size = embedding_dimension
        self.prediction_length = prediction_length
        self.seq_length = sequence_length
        self.scale = scale
        self.rnn = nn.GRU(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate,
            #batch_first=True,
        )
        self.generative = Encoder(sequence_length,  mult,  channel_mult,  prediction_length,  num_preprocess_blocks,  num_preprocess_cells,
                  num_channels_enc,  arch_instance,  num_latent_per_group,  num_channels_dec, 
                   num_postprocess_blocks,  num_postprocess_cells,  embedding_dimension,  hidden_size,  target_dim,  num_layers,dropout_rate ,groups_per_scale           )
        self.diffusion = GaussianDiffusion(
            self.generative,
            diff_steps=diff_steps,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            scale = scale,
        )
        self.projection = nn.Linear(embedding_dimension+hidden_size, embedding_dimension)
    
    def forward(self, past_time_feat, future_time_feat, t):
        """
        Output the generative results.
        """
        time_feat, _ = self.rnn(past_time_feat)
        input = torch.concat([time_feat, past_time_feat], axis=-1)
        output, y_noisy, total_c = self.diffusion.log_prob(input, future_time_feat, t)
        return output, y_noisy, total_c


class D3VAE(nn.Module):
    def __init__(self, input_dim, embedding_dimension, freq, dropout_rate, beta_schedule, beta_start, beta_end, diff_steps,
                 sequence_length, scale ,  mult,  channel_mult,  prediction_length,  num_preprocess_blocks,  num_preprocess_cells,
                  num_channels_enc,  arch_instance,  num_latent_per_group,  num_channels_dec, 
                   num_postprocess_blocks,  num_postprocess_cells, hidden_size,  target_dim,  num_layers, groups_per_scale, num_samples=100):
        super().__init__()
        """
        The whole model architecture consists of three main parts, the coupled diffusion process and the generative model are 
         included in diffusion_generate module, an resnet is used to calculate the scores.
        """
        
        
        # ResNet that used to calculate the scores.
        self.score_net = Res12_Quadratic(1, 64, 32, normalize=False, AF=nn.ELU())

        self.diff_steps = diff_steps
        self.sequence_length = sequence_length
        self.pred_len = prediction_length
        # Generate the diffusion schedule.
        self.num_samples = num_samples
        sigmas = get_beta_schedule(beta_schedule, beta_start, beta_end, diff_steps)
        alphas = 1.0 - sigmas*0.5
        self.register_buffer("alphas_cumprod", torch.tensor(np.cumprod(alphas, axis=0)))
        self.register_buffer("sqrt_alphas_cumprod", torch.tensor(np.sqrt(np.cumprod(alphas, axis=0))))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.tensor(np.sqrt(1-np.cumprod(alphas, axis=0))))
        self.register_buffer("sigmas", torch.tensor(1. - self.alphas_cumprod))
        # self.alphas_cumprod = torch.tensor(np.cumprod(alphas, axis=0))
        # self.sqrt_alphas_cumprod = torch.tensor(np.sqrt(np.cumprod(alphas, axis=0)))
        # self.sqrt_one_minus_alphas_cumprod = torch.tensor(np.sqrt(1-np.cumprod(alphas, axis=0)))
        # self.sigmas = torch.tensor(1. - self.alphas_cumprod)
        
        # The generative bvae model.
        self.diffusion_gen = diffusion_generate(sequence_length, beta_end, beta_schedule, scale, diff_steps ,  mult,  channel_mult,  prediction_length,  num_preprocess_blocks,  num_preprocess_cells,
                  num_channels_enc,  arch_instance,  num_latent_per_group,  num_channels_dec, 
                   num_postprocess_blocks,  num_postprocess_cells,  embedding_dimension,  hidden_size,  target_dim,  num_layers,  dropout_rate, groups_per_scale)
        
        # Input data embedding module.
        self.embedding = DataEmbedding(input_dim, embedding_dimension, 'fixed', freq,
                                           dropout_rate)
    
    def extract(self, a, t, x_shape):
        """ extract the t-th element from a"""
        b, *_ = t.shape
        out = torch.gather(a, dim=0, index=t)  # Use torch.gather instead of fluid layers
        out = out.reshape((b, *((1,) * (len(x_shape) - 1))))  # Reshape to match the target shape
        return out

    def forward(self, past_time_feat, past_time_feat_mark, future_time_feat):
        """
        Params:
           past_time_feat: Tensor   [B, T, *]
               the input time series.
           mark: Tensor  [B, T, *]
               the time feature mark.
           future_time_feat: Tensor  [B, O, *]
               the target time series.
        -------------
        return:
           output: Tensor
               The gauaaian distribution of the generative results.
           y_noisy: Tensor
               The diffused target.
           total_c: Float
               Total correlation of all the latent variables in the BVAE, used for disentangling.
           loss: Float
               The loss of score matching.
        """
        
        # Embed the original time series.
        t = torch.randint(low=0, high=self.diff_steps, size=(past_time_feat.shape[0],)).long().to(past_time_feat.device)
        
        input = self.embedding(past_time_feat,  past_time_feat_mark)  # [B, T, *]

        # Output the distribution of the generative results, the sampled generative results and the total correlations of the generative model.
        output, y_noisy, total_c = self.diffusion_gen(input, future_time_feat, t) 
        
        # Score matching.
        sigmas_t = self.extract(self.sigmas, t, y_noisy.shape)
        y = future_time_feat.unsqueeze(1).float()
        y_noisy1 = output.sample().float()  # Sample from the generative distribution to obtain generative results.
        # y_noisy1.requires_grad = False
        E = self.score_net(y_noisy1).sum()  
        # grad_x = paddle.grad(E, y_noisy1)[0]  # Calculate the gradient.
        grad_x = torch.autograd.grad(E, y_noisy1, create_graph=True)[0]  # Calculate the gradient
        
        # The Loss of multi-scale score mathching.
        loss = torch.mean(torch.sum(((y-y_noisy1.detach())+grad_x*0.001)**2*sigmas_t, [1,2,3])).float()
        # loss.requires_grad = False
        return  output, y_noisy, total_c, loss
    
    def pred(self, x, mark):
        """
        generate the prediction by the trained model.
        Return:
            y: The noisy generative results
            out: Denoised results, remove the noise from y through score matching.
            tc: Total correlations, indicator of extent of disentangling.
        """
        with torch.no_grad():
            input = self.embedding(x, mark)
            x_t, _ = self.diffusion_gen.rnn(input)
            input = torch.concat([x_t, input], axis=-1)
            input = input.unsqueeze(1)

            logits, tc = self.diffusion_gen.generative(input)
            output = self.diffusion_gen.generative.decoder_output(logits)
            
        # The noisy generative results.
        y = output.mu.float()
        y.requires_grad_(True)
        
        # Denoising the generatice results
        E = self.score_net(y).sum()
        # grad_x = torch.grad(E, y)[0]
        E.requires_grad_(True)
        grad_x = torch.autograd.grad(E, y, create_graph=False,  allow_unused=True)[0]
        out = y - grad_x*0.001
        return y, out, output, tc


    def prob_pred(self, x, mark):
        """
        generate the prediction by the trained model.
        Return:
            y: The noisy generative results
            out: Denoised results, remove the noise from y through score matching.
            tc: Total correlations, indicator of extent of disentangling.
        """
        B, T, N = x.shape
        with torch.no_grad():
            input = self.embedding(x, mark)
            x_t, _ = self.diffusion_gen.rnn(input)
            input = torch.concat([x_t, input], axis=-1)
            input = input.unsqueeze(1)

            logits, tc = self.diffusion_gen.generative(input)
            output = self.diffusion_gen.generative.decoder_output(logits)
            
        # The noisy generative results.
        # y = output.mu + torch.rand((B, 100, T, N)).to(x.device) * output.sigma
        y = output.mu.float()
        y.requires_grad_(True)
        
        # Denoising the generatice results
        E = self.score_net(y).sum()
        # grad_x = torch.grad(E, y)[0]
        E.requires_grad_(True)
        grad_x = torch.autograd.grad(E, y, create_graph=False,  allow_unused=True)[0]
        # out = y - grad_x*0.001
        out = y -  torch.rand((B, self.num_samples, T, N)).to(x.device) * grad_x * output.sigma#*0.001
        return y, out, output, tc


class Discriminator(nn.Module):
    def __init__(self, neg_slope=0.2, latent_dim=10, hidden_units=1000, out_units=2):
        """Discriminator proposed in [1].
        Params:
        neg_slope: float
            Hyperparameter for the Leaky ReLu
        latent_dim : int
            Dimensionality of latent variables.
        hidden_units: int
            Number of hidden units in the MLP
        Model Architecture
        ------------
        - 6 layer multi-layer perceptron, each with 1000 hidden units
        - Leaky ReLu activations
        - Output 2 logits
        References:
            [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
            arXiv preprint arXiv:1802.05983 (2018).
        """
        super(Discriminator, self).__init__()

        # Activation parameters
        self.neg_slope = neg_slope
        self.leaky_relu = nn.LeakyReLU(self.neg_slope, True)

        # Layer parameters
        self.z_dim = latent_dim
        self.hidden_units = hidden_units
        # theoretically 1 with sigmoid but gives bad results => use 2 and softmax
        out_units = out_units

        # Fully connected layers
        self.lin1 = nn.Linear(self.z_dim, hidden_units)
        self.lin2 = nn.Linear(hidden_units, hidden_units)
        self.lin3 = nn.Linear(hidden_units, hidden_units)
        self.lin4 = nn.Linear(hidden_units, hidden_units)
        self.lin5 = nn.Linear(hidden_units, hidden_units)
        self.lin6 = nn.Linear(hidden_units, out_units)
        self.softmax = nn.Softmax()

    def forward(self, z):

        # Fully connected layers with leaky ReLu activations
        z = self.leaky_relu(self.lin1(z))
        z = self.leaky_relu(self.lin2(z))
        z = self.leaky_relu(self.lin3(z))
        z = self.leaky_relu(self.lin4(z))
        z = self.leaky_relu(self.lin5(z))
        z = self.lin6(z)

        return z
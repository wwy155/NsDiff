import torch
import torch.nn as nn
from torch_timeseries.nn.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from torch_timeseries.nn.SelfAttention_Family import DSAttention, AttentionLayer
from torch_timeseries.nn.embedding import DataEmbedding
import torch.nn.functional as F
from src.utils.sigma import wv_sigma, wv_sigma_trailing

# class MLP(nn.Module):

#     def __init__(self, seq_len, pred_len, enc_in, hidden_size):
#         super(SigmaEstimation, self).__init__()
#         self.pred_len = pred_len
#         self.seq_len = seq_len
#         self.enc_in = enc_in
#         self.hidden_size = hidden_size


class SigmaEstimation(nn.Module):
    """
    Non-stationary Transformer
    """

    def __init__(self, seq_len, pred_len, enc_in, hidden_size, kernel_size=24):
        super(SigmaEstimation, self).__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.hidden_size = hidden_size
        
        
        # Define 2-layer MLP for predicting future sigmas
        self.mlp = nn.Sequential(
            nn.Linear(seq_len -  kernel_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, pred_len)  # Output size should match enc_in
        )
        
        # Moving average kernel (can adjust the window size if needed)
        self.kernel_size = kernel_size  # You can modify this to adjust the window size
        self.padding = self.kernel_size // 2  # To ensure the output length matches the input length

        
        
    def forward(self, x_enc):
        # x_enc: B, T, N
        # return sigmas: B, O, N
        B, T, N = x_enc.shape
        
        # # 1. Compute the rolling standard deviation (sigma) along the time dimension
        # x_enc_padded = F.pad(x_enc, (self.padding, self.padding), mode='constant', value=0)  # Padding for edges
        # # Standard deviation in a sliding window along the time dimension
        # squared = x_enc_padded ** 2
        # mean_sq = F.conv1d(squared.permute(0, 2, 1), weight=torch.ones(1, N, self.window_size).to(x_enc.device) / self.window_size, stride=1)
        # mean = F.conv1d(x_enc_padded.permute(0, 2, 1), weight=torch.ones(1, N, self.window_size).to(x_enc.device) / self.window_size, stride=1)
        # std_dev = torch.sqrt(mean_sq - mean ** 2)
        
        sigma = wv_sigma_trailing(x_enc, self.kernel_size, discard_rep=True)        
        # 2. Use MLP to predict future sigma values
        sigma = sigma[:, -(T - self.kernel_size):, :] + 10e-8
        pred_sigma = self.mlp(sigma.permute(0, 2, 1))  # (B, T, N) -> (B, N, T)
        
        # 3. Extract the last `pred_len` time steps for prediction
        pred_sigma = F.softplus(pred_sigma).permute(0, 2, 1)  # (B, O, N) where O = pred_len
        # stdevs = torch.sqrt(sigmas)
        return pred_sigma[:, -self.pred_len:, :]

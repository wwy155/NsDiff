
from torch.nn.modules import loss
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

# from gluonts.core.component import validated

from pts.model import weighted_average
from src.utils.gaussian_diffusion import GaussianDiffusion
from pts.model.time_grad import EpsilonTheta

import time

def get_lagged_subsequences(
    sequence: torch.Tensor,
    sequence_length: int,
    indices: List[int],
    subsequences_length: int = 1,
) -> torch.Tensor:
    """
    Returns lagged subsequences of a given sequence.
    Parameters
    ----------
    sequence
        the sequence from which lagged subsequences should be extracted.
        Shape: (N, T, C).
    sequence_length
        length of sequence in the T (time) dimension (axis = 1).
    indices
        list of lag indices to be used.
    subsequences_length
        length of the subsequences to be extracted.
    Returns
    --------
    lagged : Tensor
        a tensor of shape (N, S, C, I),
        where S = subsequences_length and I = len(indices),
        containing lagged subsequences.
        Specifically, lagged[i, :, j, k] = sequence[i, -indices[k]-S+j, :].
    """
    # we must have: history_length + begin_index >= 0
    # that is: history_length - lag_index - sequence_length >= 0
    # hence the following assert
    assert max(indices) + subsequences_length <= sequence_length, (
        f"lags cannot go further than history length, found lag "
        f"{max(indices)} while history length is only {sequence_length}"
    )
    assert all(lag_index >= 0 for lag_index in indices)

    lagged_values = []
    for lag_index in indices:
        begin_index = -lag_index - subsequences_length
        end_index = -lag_index if lag_index > 0 else None
        lagged_values.append(sequence[:, begin_index:end_index, ...].unsqueeze(1))
    return torch.cat(lagged_values, dim=1).permute(0, 2, 3, 1)




class TimeGrad(nn.Module):
    def __init__(
        self,
        pred_len : int ,
        sequence_length : int , 
        enc_in : int ,
        lags : List[int]=  [1, 24],
        context_length : int = 24,
        TimeF : int = 4,
        rnn_hidden_size: int =40,
        rnn_type : str = 'LSTM',
        num_layers: int = 2,
        dropout_rate : float = 0.1,
        scale : bool = True,
        covariate_hidden_size : int = 128,
        conditioning_length: int = 100,
        diff_steps: int = 50,
        beta_end: float = 0.1,
        beta_schedule: str = 'linear',
        residual_layers: int = 8,
        residual_channels: int = 8,
        num_samples:int = 100,
        dilation_cycle_length: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        
        self.context_length = context_length
        self.conditioning_length = conditioning_length
        self.pred_len = pred_len
        self.lags = lags
        self.enc_in = enc_in
        self.covariate_static_embed = torch.nn.Parameter(torch.randn(enc_in, covariate_hidden_size), requires_grad=True)        
        self.scale = scale
        self.seq_len = sequence_length
        self.num_timesteps = diff_steps
        self.num_samples = num_samples
        self.rnn_type =rnn_type
        
        
        self.shifted_lags = [l - 1 for l in self.lags]
        I= len(self.lags)
   
   
        rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_type]
        input_size = self.enc_in*I + self.enc_in*covariate_hidden_size +  TimeF 
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=rnn_hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
        )
        
        self.dist_args_proj = nn.Linear(rnn_hidden_size, conditioning_length)
        
        self.denoise_fn = EpsilonTheta(
            target_dim=enc_in,
            cond_length=conditioning_length,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
            dilation_cycle_length=dilation_cycle_length,
        )

        self.diffusion = GaussianDiffusion(
            self.denoise_fn,
            input_size=enc_in,
            diff_steps=diff_steps,
            loss_type='l2',
            beta_end=beta_end,
            beta_schedule=beta_schedule,
        )

        # self.distr_output = DiffusionOutput(
        #     self.diffusion, input_size=target_dim, cond_size=conditioning_length
        # )
    
    def encoder(self, x, x_date, y_date, y=None):
        """
            input
                x: (B, T, N)
                y: (B, O, N)
                x_date: (B, T, TimeF) 
                y_date: (B, T, TimeF) 
            output:
                if y: B, T_c+O, H
                else: B, T_c, H
        """
        T_c = self.context_length

        if y is None:
            time_feat = x_date[:, -T_c :, ...]
            rnn_input_length = self.seq_len
            rnn_input = x
            S = T_c
        else:
            time_feat = torch.cat(
                (x_date[:, -T_c :, ...], y_date), 
                dim=1,
            )
            rnn_input_length = self.pred_len + self.seq_len
            rnn_input = sequence = torch.cat((x, y), dim=1)
            S = T_c + self.pred_len

        # lags: (B, S, N, I)
        lags_x = get_lagged_subsequences(
            sequence=rnn_input,
            sequence_length=rnn_input_length,
            indices=self.lags,
            subsequences_length=S,
        )
        
        outputs, states, inputs = self.rnn_output(
            lags_x,
            time_feat,
            S,
        )
        
        return outputs, states, inputs
        
    
    def rnn_output(
        self, 
        lags_x: torch.Tensor,
        time_feat: torch.Tensor,
        unroll_length: int,
        begin_state: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        ):
        # lags_x: lagged input (B, S, I, N)
        # time_feat: input (B, S, TimF)
        # unroll_length: S
        # begin_state: (B, N)
        
        
        B = lags_x.shape[0]
        input_lags = lags_x.reshape(
            (-1, unroll_length, len(self.lags) * self.enc_in)
        ) # B, U, I*N
        
        index_embeddings = (
            self.covariate_static_embed.unsqueeze(0).unsqueeze(0) #1, 1, N, 128
            .expand(B, unroll_length, -1, -1) # B, U, N, 128
            .reshape((-1, unroll_length, self.enc_in * 128))  # , U, N*128
        ).to(lags_x.device) # B, S, 128*N
        
        
        # inputs: (B, S, dimension= N*I + N*128 + (TimF) )
        inputs = torch.cat((input_lags, index_embeddings, time_feat), dim=-1)

        outputs, state = self.rnn(inputs, begin_state) 
        
        return outputs, state, inputs


    # def rnn_condition_output(self, x, x_date, y_date, begin_state=None):
    #     # input rnn
    #     lags_x = get_lagged_subsequences(x, self.seq_len, self.lags, subseq_length) # (B, S, N, len(self.lags)),
    #     lags_x = lags_x.reshape(-1, subseq_length, lags_x.shape[-1] * lags_x.shape[-2]) # (B, S, N*len(self.lags))

    #     # lags_x_date = get_lagged_subsequences(x_date, self.seq_len, self.lags, subseq_length) # (B, S, TimF, len(self.lags)),
    #     # lags_x_date = lags_x_date.reshape(-1, subseq_length, lags_x_date.shape[-1] * lags_x_date.shape[-2]) # (B, S, TimF*len(self.lags))
    #     # lags_y_date = lags_y.reshape(-1, subseq_length, lags_y.shape[-1] * lags_y.shape[-2]) # (B, S, N*len(self.lags))
        
    #     if self.pred_len > S:
    #         dates = torch.concat([lags_x_date[:, -S:, :], lags_y_date[:, :S, :]])
    #     else:
    #         dates = torch.concat([lags_x_date[:, -(S + S - self.pred_len):, :], lags_y_date]) # (B, S, 2*TimF)
        
    #     self.covariate_static_embed = self.covariate_static_embed.unsqueeze(0).unsqueeze(0).repeat(B, S, 1) # (N, 128) -> B, S, N, 128
    #     self.covariate_static_embed = self.covariate_static_embed.reshape(-1, subseq_length, self.covariate_static_embed.shape[-1] * self.covariate_static_embed.shape[-2]) # (B, S, N*len(self.lags))
        
    #     rnn_outputs, states = self.rnn(torch.concat([lags_x, self.covariate_static_embed, dates], dim=2)) # input: (B, S, dimension= N*I + N*128 + (TimF*2) )

    #     return rnn_outputs, states

    
    def sample_decoder(self, x, y_date, begin_states):
        """
            x: (B, T, N)
            y_date: (B, O, TimeF)
            begin_states: (B, 1, H)
            
            produce numer of  self.num_parallel_samples diffusion samples
        """
        def repeat(tensor, dim=0):
            return tensor.repeat_interleave(repeats=self.num_samples, dim=dim)

        # blows-up the dimension of each tensor to
        # batch_size * self.num_sample_paths for increasing parallelism
        
        repeated_x_new_y= repeat(x)
        repeated_y_date = repeat(y_date)
        

        if self.rnn_type == "LSTM":
            repeated_states = [repeat(s, dim=1) for s in begin_states]
        else:
            repeated_states = repeat(begin_states, dim=1)

        future_samples = []
        
        
        for k in range(self.pred_len):
            lags_x = get_lagged_subsequences(
                sequence=repeated_x_new_y,
                sequence_length=self.seq_len + k,
                indices=self.shifted_lags,
                subsequences_length=1,
            )
            
            rnn_outputs, repeated_states, _ = self.rnn_output(
                lags_x,
                repeated_y_date[:, k:k+1, ...],
                1,
                repeated_states
            )
            
            distr_args = self.dist_args_proj(rnn_outputs) # B*samples, 1 , 100

            new_samples = self.diffusion.sample(cond=distr_args) # B*samples, 1 , 100

            # (batch_size, seq_len, target_dim)
            future_samples.append(new_samples)
            repeated_x_new_y = torch.cat(
                (repeated_x_new_y, new_samples), dim=1
            )
        # (batch_size * num_samples, O, N)
        samples = torch.cat(future_samples, dim=1)

        # (batch_size, num_samples, prediction_length, target_dim)
        return samples.reshape(
            (
                -1,
                self.num_samples,
                self.pred_len,
                self.enc_in,
            )
        )




    def inference(self, x, x_date, y_date):
        # x: (B, T, N)
        # x_date: (B, T, TimF)
        # y_date: (B, O, TimF)

        # ouputs: (B, T_c, N), states: (B, 1, H) 
        # get hidden states
        _, h, _ = self.encoder(x, x_date, y_date)
        
         # (batch_size, num_samples, prediction_length, target_dim)
        output = self.sample_decoder(x, y_date, h)

        if self.scale:
            output.reshape(-1, self.pred_len, self.enc_in)

        return output
        
        
        
        

        # def repeat(tensor, dim=0):
        #     return tensor.repeat_interleave(repeats=self.num_parallel_samples, dim=dim)

        # if self.cell_type == "LSTM":
        #     repeated_states = [repeat(s, dim=1) for s in begin_states]
        # else:
        #     repeated_states = repeat(begin_states, dim=1)

        
        
        
        # _, begin_states = self.rnn_condition_output(x, x_date, y_date)
        
        
        # # for each future time-units, draw new samples for this time-unit
        # # and update the state
        # for k in range(self.prediction_length):
        #     lags = self.get_lagged_subsequences(
        #         sequence=repeated_past_target_cdf,
        #         sequence_length=self.history_length + k,
        #         indices=self.shifted_lags,
        #         subsequences_length=1,
        #     )

        #     rnn_outputs, repeated_states, _, _ = self.unroll(
        #         begin_state=repeated_states,
        #         lags=lags,
        #         scale=repeated_scale,
        #         time_feat=repeated_time_feat[:, k : k + 1, ...],
        #         target_dimension_indicator=repeated_target_dimension_indicator,
        #         unroll_length=1,
        #     )

        #     distr_args = self.distr_args(rnn_outputs=rnn_outputs) # B, T_C + O, 100

        #     # (batch_size, 1, target_dim)
        #     new_samples = self.diffusion.sample(cond=distr_args)

        #     # (batch_size, seq_len, target_dim)
        #     future_samples.append(new_samples)
        #     repeated_past_target_cdf = torch.cat(
        #         (repeated_past_target_cdf, new_samples), dim=1
        #     )

        # # (batch_size * num_samples, prediction_length, target_dim)
        # samples = torch.cat(future_samples, dim=1)


        
        

    def train_forward(self, x, y, x_date, y_date):
        # x: (B, T, N)
        # y: (B, O, N)
        # x_date: (B, T, TimF)
        # y_date: (B, O, TimF)
        
        B, T, N = x.shape
        T_c = self.context_length
        _, O, _ = y.shape
        
        subseq_length = 24
        S = self.context_length
        
        # scale in Time Grad
        if self.scale:
            mean = x.mean(dim=1, keepdim=True)
            mean = torch.where(
                mean == 0,
                mean,
                1
            )
            x = x/(mean)
        
        # RNN encode 
        outputs, states, inputs = self.encoder(x, x_date, y_date, y=y)
        # # input rnn
        # rnn_outputs, _ =  self.rnn_condition_output(x, x_date, y_date)
        # lags_x = get_lagged_subsequences(x, self.seq_len, self.lags, subseq_length) # (B, S, N, len(self.lags)),
        # lags_x = lags_x.reshape(-1, subseq_length, lags_x.shape[-1] * lags_x.shape[-2]) # (B, S, N*len(self.lags))

        # # lags_x_date = get_lagged_subsequences(x_date, self.seq_len, self.lags, subseq_length) # (B, S, TimF, len(self.lags)),
        # # lags_x_date = lags_x_date.reshape(-1, subseq_length, lags_x_date.shape[-1] * lags_x_date.shape[-2]) # (B, S, TimF*len(self.lags))
        # # lags_y_date = lags_y.reshape(-1, subseq_length, lags_y.shape[-1] * lags_y.shape[-2]) # (B, S, N*len(self.lags))
        
        # if self.pred_len > S:
        #     dates = torch.concat([lags_x_date[:, -S:, :], lags_y_date[:, :S, :]])
        # else:
        #     dates = torch.concat([lags_x_date[:, -(S + S - self.pred_len):, :], lags_y_date]) # (B, S, 2*TimF)
        
        # self.covariate_static_embed = self.covariate_static_embed.unsqueeze(0).unsqueeze(0).repeat(B, S, 1) # (N, 128) -> B, S, N, 128
        # self.covariate_static_embed = self.covariate_static_embed.reshape(-1, subseq_length, self.covariate_static_embed.shape[-1] * self.covariate_static_embed.shape[-2]) # (B, S, N*len(self.lags))
        
        # rnn_outputs, _ = self.rnn(torch.concat([lags_x, self.covariate_static_embed, dates], dim=2)) # input: (B, S, dimension= N*I + N*128 + (TimF*2) )
        
        # diffusion forward process
        dist_args = self.dist_args_proj(outputs) # B, T_C + O, conditioning_length(100)
        
        target = torch.cat(
            (x[:, -self.context_length :, ...], y),
            dim=1,
        ) # B, T_c+O, N

        time = torch.randint(0, self.num_timesteps, (B * (T_c+ O),), device=x.device).long()
        noise, x_noisy, pred_noise = self.diffusion.q_sample_denoise(target.reshape(B * (T_c+ O), 1, -1), dist_args.reshape(B * (T_c+ O), 1, -1), time)

        return noise, pred_noise          
    
    def forward(self, x, y, x_date, y_date, train=True):
        # x: (B, T, N)
        # x_date: (B, T, N)
        # y_date: (B, T)
        if self.scale:
            mean = x.mean(dim=1, keepdim=True)
            mean = torch.where(
                mean == 0,
                mean,
                1
            )
            x = x/mean

        if train:
            if self.scale:
                y = y/mean
            noise, pred_noise = self.train_forward(x, y, x_date, y_date) # (B, )
            return noise, pred_noise
            
        else:
            output = self.inference(x, x_date, y_date) # B, samples, O, N  
            if self.scale:
                output *= mean.unsqueeze(1)
            return output
            
        
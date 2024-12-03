

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange



def noise_mask(X, masking_ratio=0.15, lm=3, mode='separate', distribution='geometric', exclude_feats=None):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        mode: whether each variable should be masked separately ('separate'), or all variables at a certain positions
            should be masked concurrently ('concurrent')
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)

    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        if mode == 'separate':  # each variable (feature) is independent
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), X.shape[1])

    return mask


def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked

    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask

class Conv1dWithInitialization(nn.Module):
    def __init__(self, **kwargs):
        super(Conv1dWithInitialization, self).__init__()
        self.conv1d = torch.nn.Conv1d(**kwargs)
        torch.nn.init.orthogonal_(self.conv1d.weight.data, gain=1)

    def forward(self, x):
        return self.conv1d(x)


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):

        x = self.embedding[diffusion_step]
        # print("1", np.shape(x))
        x = self.projection1(x)
        # print("2", np.shape(x))
        x = F.silu(x)
        x = self.projection2(x)
        # print("3", np.shape(x))
        x = F.silu(x)
        # 1 torch.Size([64, 128])
        # 2 torch.Size([64, 128])
        # 3 torch.Size([64, 128])
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class InputConvNetwork(nn.Module):

    def __init__(self, args, inp_num_channel, out_num_channel, num_layers=3, ddpm_channels_conv=None):
        super(InputConvNetwork, self).__init__()

        self.args = args

        self.inp_num_channel = inp_num_channel
        self.out_num_channel = out_num_channel

        kernel_size = 3
        padding = 1
        if ddpm_channels_conv is None:
            self.channels = args.ddpm_channels_conv
        else:
            self.channels = ddpm_channels_conv
        self.num_layers = num_layers

        self.net = nn.ModuleList()

        if num_layers == 1:
            self.net.append(Conv1dWithInitialization(
                                            in_channels=self.inp_num_channel,
                                            out_channels=self.out_num_channel,
                                            kernel_size=kernel_size,
                                            stride=1,
                                            padding=padding, bias=True
                                        )
                                    )
        else:
            for i in range(self.num_layers-1):
                if i == 0:
                    dim_inp = self.inp_num_channel
                else:
                    dim_inp = self.channels
                self.net.append(Conv1dWithInitialization(
                                            in_channels=dim_inp,
                                            out_channels=self.channels,
                                            kernel_size=kernel_size,
                                            stride=1,
                                            padding=padding, bias=True
                                        ))
                self.net.append(torch.nn.BatchNorm1d(self.channels)), 
                self.net.append(torch.nn.LeakyReLU(0.1)),
                self.net.append(torch.nn.Dropout(0.1, inplace = True))

            self.net.append(Conv1dWithInitialization(
                                            in_channels=self.channels,
                                            out_channels=self.out_num_channel,
                                            kernel_size=kernel_size,
                                            stride=1,
                                            padding=padding, bias=True
                                        )
                                    )

    def forward(self, x=None):

        out = x
        for m in self.net:
            out = m(out)

        return out


class CNN_DiffusionUnet(nn.Module):

    def __init__(self, args, num_vars, seq_len, pred_len, diff_steps):
        super(CNN_DiffusionUnet, self).__init__()

        self.args = args

        self.num_vars = num_vars
        self.ori_seq_len = args.seq_len
        self.seq_len = seq_len
        self.label_len = args.label_len
        self.pred_len = pred_len

        kernel_size = 3
        padding = 1
        self.channels = args.ddpm_inp_embed
        if self.args.features in ['MS']:
            self.input_projection = InputConvNetwork(args, 1, self.channels)
        else:
            self.input_projection = InputConvNetwork(args, self.num_vars, self.channels, num_layers=args.ddpm_layers_inp)

        self.dim_diff_steps = args.ddpm_dim_diff_steps
        
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=diff_steps,
            embedding_dim=self.dim_diff_steps,
        )

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

        kernel_size = 3
        padding = 1
        self.dim_intermediate_enc = args.ddpm_channels_fusion_I

        self.enc_conv = InputConvNetwork(args, self.channels+self.dim_diff_steps, self.dim_intermediate_enc, num_layers=args.ddpm_layers_I)

        self.cond_projections = nn.ModuleList()

        if args.ablation_study_F_type == "Linear":
            for i in range(self.num_vars):
                self.cond_projections.append(nn.Linear(self.ori_seq_len,self.pred_len))
                self.cond_projections[i].weight = nn.Parameter((1/self.ori_seq_len)*torch.ones([self.pred_len,self.ori_seq_len]))
        elif args.ablation_study_F_type == "CNN":
            for i in range(self.num_vars):
                self.cond_projections.append(nn.Linear(self.ori_seq_len,self.pred_len))
                self.cond_projections[i].weight = nn.Parameter((1/self.ori_seq_len)*torch.ones([self.pred_len,self.ori_seq_len]))

            self.cnn_cond_projections = InputConvNetwork(args, self.num_vars, self.pred_len, num_layers=args.cond_ddpm_num_layers, ddpm_channels_conv=args.cond_ddpm_channels_conv)
            self.cnn_linear = nn.Linear(self.ori_seq_len, self.num_vars)

        if self.args.ablation_study_case  in ["mix_1", "mix_ar_0"]:
            self.combine_conv = InputConvNetwork(args, self.dim_intermediate_enc+self.num_vars, self.num_vars, num_layers=args.ddpm_layers_II)
        else:
            self.combine_conv = InputConvNetwork(args, self.dim_intermediate_enc+self.num_vars+self.num_vars, self.num_vars, num_layers=args.ddpm_layers_II)

    def forward(self,  yn=None, diffusion_step=None, cond_info=None, y_clean=None):

        x = yn
        x = self.input_projection(x)  
        diffusion_emb = self.diffusion_embedding(diffusion_step.long())
        # diffusion_emb = self.act(self.diffusion_embedding(diffusion_step)) 
        diffusion_emb = self.act(diffusion_emb)
        diffusion_emb = diffusion_emb.unsqueeze(-1).repeat(1,1,np.shape(x)[-1])

        # print(">>>>", np.shape(diffusion_emb), np.shape(x))
        # torch.Size([64, 1024, 168]) torch.Size([64, 1024, 168])

        # print(np.shape(diffusion_emb), np.shape(x))
        # torch.Size([64, 128, 168]) torch.Size([64, 256, 168])

        # print(">>>", np.shape(diffusion_emb))
        x = self.enc_conv(torch.cat([diffusion_emb, x], dim=1))
        # print(np.shape(x))
        
        pred_out = torch.zeros([yn.size(0),self.num_vars,self.pred_len],dtype=yn.dtype).to(yn.device)
        for i in range(self.num_vars):
            pred_out[:,i,:] = self.cond_projections[i](cond_info[:,i,:self.ori_seq_len])
        
        if self.args.ablation_study_F_type == "CNN":
            # cnn with residual links
            temp_out = self.cnn_cond_projections(cond_info[:,:,:self.ori_seq_len])
            pred_out += self.cnn_linear(temp_out).permute(0,2,1)
            
        return_pred_out = pred_out

        if y_clean is not None:
            y_clean = y_clean[:,:,-self.pred_len:]

            rand_for_mask = torch.rand_like(y_clean).to(x.device)
                
        # ==================================================================================
        if y_clean is not None:
            pred_out = rand_for_mask * pred_out + (1-rand_for_mask) * y_clean
        inp = torch.cat([x, pred_out, cond_info[:,:,self.ori_seq_len:]], dim=1)

        out = self.combine_conv(inp)

        if y_clean is not None:
            return out, return_pred_out
        return out



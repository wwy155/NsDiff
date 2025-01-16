import math
import torch
import numpy as np

EPS = 10e-8
def make_beta_schedule(schedule="linear", num_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == "linear":
        betas = torch.linspace(start, end, num_timesteps)
    elif schedule == "const":
        betas = end * torch.ones(num_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, num_timesteps) ** 2
    elif schedule == "jsd":
        betas = 1.0 / torch.linspace(num_timesteps, 1, num_timesteps)
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == "cosine" or schedule == "cosine_reverse":
        max_beta = 0.999
        cosine_s = 0.008
        betas = torch.tensor(
            [min(1 - (math.cos(((i + 1) / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2) / (
                    math.cos((i / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2), max_beta) for i in
             range(num_timesteps)])
        if schedule == "cosine_reverse":
            betas = betas.flip(0)  # starts at max_beta then decreases fast
    elif schedule == "cosine_anneal":
        betas = torch.tensor(
            [start + 0.5 * (end - start) * (1 - math.cos(t / (num_timesteps - 1) * math.pi)) for t in
             range(num_timesteps)])
    return betas


def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

def cal_sigma12(alphas,alphas_cumprod,alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled_m_1, betas_bar_m_1, gx, y_sigma, t):
    
    at = extract(alphas, t, gx)
    at_bar = extract(alphas_cumprod, t, gx)
    # at_bar_prev = extract(alpha_bar_prev, t, gx)
    at_tilde = extract(alphas_cumprod_sum, t, gx)
    b_tilde_m_1 = extract(betas_tiled_m_1, t, gx)
    b_bar_m_1 = extract(betas_bar_m_1, t, gx)
    # at_tilde_prev = extract(alphas_cumprod_sum_prev, t, gx)

    Sigma_1 = (1 - at)**2*gx + at*(1 - at)*y_sigma 
    Sigma_2 = (b_bar_m_1 - b_tilde_m_1)*gx + b_tilde_m_1*y_sigma
    # sigma_tilde = (Sigma_1*Sigma_2)/(at * Sigma_2 + Sigma_1)
    # # mu_tilde = (Sigma_1*Sigma_2)/(at * Sigma_2 + Sigma_1)
    # Sigma_1 = 1 - at
    # Sigma_2 = 1 - at_bar_prev
    return at, at_bar, at_tilde, Sigma_1, Sigma_2

def cal_forward_noise(betas_tiled, betas_bar, gx, y_sigma, t):
    b_bar_t =  extract(betas_bar, t, gx)
    b_tilded_t =  extract(betas_tiled, t, gx)
    
    noise = (b_bar_t - b_tilded_t)*gx + b_tilded_t*y_sigma
    assert (noise > 0).all()
    # if not (noise > 0).all():
    #     import pdb;pdb.set_trace()
    return noise


def cal_sigma_tilde(alphas,alphas_cumprod,alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled_m_1, betas_bar_m_1, gx, y_sigma, t):
    at, at_bar, at_tilde, Sigma_1, Sigma_2 = cal_sigma12(alphas,alphas_cumprod,alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled_m_1, betas_bar_m_1, gx, y_sigma, t)
    sigma_tilde = (Sigma_1*Sigma_2)/(at * Sigma_2 + Sigma_1)
    return sigma_tilde

def calc_gammas(alphas,alphas_cumprod,alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled_m_1, betas_bar_m_1, gx, y_sigma, t):
    at, at_bar, at_tilde, Sigma_1, Sigma_2 = cal_sigma12(alphas,alphas_cumprod,alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled_m_1, betas_bar_m_1, gx, y_sigma, t)
    
    alpha_bar_t_m_1 = extract(alpha_bar_prev, t, gx)
    sqrt_alpha_t = at.sqrt()
    sqrt_alpha_bar_t_m_1 = alpha_bar_t_m_1.sqrt()
    
    at_s1_s2 = at*Sigma_2 + Sigma_1
    
    gamma_0 = sqrt_alpha_bar_t_m_1*Sigma_1/at_s1_s2
    gamma_1 = sqrt_alpha_t*Sigma_2/at_s1_s2
    gamma_2 = ((sqrt_alpha_t*(at - 1))*Sigma_2 + (1 - sqrt_alpha_bar_t_m_1)*Sigma_1)/at_s1_s2
    return gamma_0, gamma_1, gamma_2


# Forward functions
def q_sample(y, y_0_hat, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, t, noise=None):
    """
    y_0_hat: prediction of pre-trained guidance model; can be extended to represent
        any prior mean setting at timestep T.
    """
    if noise is None:
        noise = torch.randn_like(y).to(y.device)
    sqrt_alpha_bar_t = extract(alphas_bar_sqrt, t, y)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    # q(y_t | y_0, x)
    y_t = sqrt_alpha_bar_t * y + (1 - sqrt_alpha_bar_t) * y_0_hat + noise
    return y_t


# Reverse function -- sample y_{t-1} given y_t
def p_sample(model, x, x_mark, y, y_0_hat, gx, y_T_mean, t, alphas, one_minus_alphas_bar_sqrt, alphas_cumprod,alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled_all, betas_bar_all, betas_tiled_m_1_all, betas_bar_m_1_all):
    """
    Reverse diffusion process sampling -- one time step.

    y: sampled y at time step t, y_t.
    y_0_hat: prediction of pre-trained guidance model.
    y_T_mean: mean of prior distribution at timestep T.
    We replace y_0_hat with y_T_mean in the forward process posterior mean computation, emphasizing that 
        guidance model prediction y_0_hat = f_phi(x) is part of the input to eps_theta network, while 
        in paper we also choose to set the prior mean at timestep T y_T_mean = f_phi(x).
    """
    device = next(model.parameters()).device
    t = torch.tensor([t]).to(device)
    eps_theta, sigma_theta = model(x, x_mark, y, y_0_hat, gx, t)
    
    eps_theta = eps_theta.to(device).detach()
    sigma_theta = sigma_theta.to(device).detach()
    
    z =  torch.randn_like(y)  # if t > 1 else torch.zeros_like(y)
    alpha_t = extract(alphas, t, y)
    
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    sqrt_one_minus_alpha_bar_t_m_1 = extract(one_minus_alphas_bar_sqrt, t - 1, y)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    sqrt_alpha_bar_t_m_1 = (1 - sqrt_one_minus_alpha_bar_t_m_1.square()).sqrt()
    
    betas_tiled_m_1 = extract(betas_tiled_m_1_all, t, y)
    betas_bar_m_1 = extract(betas_bar_m_1_all, t, y)
    betas_tiled = extract(betas_tiled_all, t, y)
    betas_bar = extract(betas_bar_all, t, y)
    # estimate Sigma Y0
    # lambda_0 = alpha_t*(1 - alpha_t)*betas_tiled_m_1
    # lambda_1 = (1 - alpha_t)**2*betas_tiled_m_1 + alpha_t*(1 - alpha_t)*(betas_bar_m_1 - betas_tiled_m_1)*gx - sigma_theta*(alpha_t*betas_tiled_m_1 + alpha_t*(1 - alpha_t))
    # lambda_2 = gx**2*(1 - alpha_t)**2*(betas_bar_m_1 - betas_tiled_m_1) - sigma_theta*gx*(alpha_t*betas_bar_m_1 - alpha_t*betas_tiled_m_1 + (1 - alpha_t)**2)
    # sigma_y0_hat = (-lambda_1 + ((lambda_1)**2 - 4*lambda_0*lambda_2).sqrt()  )/(2*lambda_0)
    # noise = (betas_bar - betas_tiled)*gx + betas_tiled*sigma_y0_hat
    # # assert (noise > 0).all()
    # noise[noise<=0] = gx[noise<=0]
    noise = (betas_bar)*gx

    
    # y_t_m_1 posterior mean component coefficients, when inference, use gx to replace \Sigma_{Y_0}
    # at, at_bar, at_tilde, Sigma_1, Sigma_2 = cal_sigma12(alphas, alphas_cumprod, alphas_cumprod_sum,alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled_m_1_all, betas_bar_m_1_all, gx, sigma_y0_hat, t)
    # gamma_0 = (1 - alpha_t) * sqrt_alpha_bar_t_m_1 / (sqrt_one_minus_alpha_bar_t.square())
    # gamma_1 = (sqrt_one_minus_alpha_bar_t_m_1.square()) * (alpha_t.sqrt()) / (sqrt_one_minus_alpha_bar_t.square())
    # gamma_2 = 1 + (sqrt_alpha_bar_t - 1) * (alpha_t.sqrt() + sqrt_alpha_bar_t_m_1) / (
    #     sqrt_one_minus_alpha_bar_t.square())
    
    
    # y_0 reparameterization
    y_0_reparam = 1 / sqrt_alpha_bar_t * (
            y - (1 - sqrt_alpha_bar_t) * y_T_mean - eps_theta*torch.sqrt(noise))
    # posterior mean
    gamma_0, gamma_1, gamma_2 = calc_gammas(alphas, alphas_cumprod, alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled_m_1_all, betas_bar_m_1_all, gx, gx, t)
    y_t_m_1_hat = gamma_0 * y_0_reparam + gamma_1 * y + gamma_2 * y_T_mean
    # posterior variance
    y_t_m_1 = y_t_m_1_hat.to(device) + torch.sqrt(sigma_theta) *z.to(device)
    return y_t_m_1


# Reverse function -- sample y_0 given y_1
def p_sample_t_1to0(model, x, x_mark, y, y_0_hat, gx, y_T_mean, one_minus_alphas_bar_sqrt,alphas,alphas_cumprod,alphas_cumprod_sum,alpha_bar_prev, alphas_cumprod_sum_prev,betas_tiled_all, betas_bar_all, betas_tiled_m_1_all, betas_bar_m_1_all):
    device = next(model.parameters()).device
    t = torch.tensor([0]).to(device)  # corresponding to timestep 1 (i.e., t=1 in diffusion models)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    eps_theta, sigma_theta = model(x, x_mark, y, y_0_hat, gx, t)
    
    # at_tilde = extract(alphas_cumprod_sum, t, gx)
    
    eps_theta = eps_theta.to(device).detach()
    sigma_theta = sigma_theta.to(device).detach()
    alpha_t = extract(alphas, t, y)
    
    betas_tiled_m_1 = extract(betas_tiled_m_1_all, t, y)
    betas_bar_m_1 = extract(betas_bar_m_1_all, t, y)
    betas_tiled = extract(betas_tiled_all, t, y)
    betas_bar = extract(betas_bar_all, t, y)

    # estimate Sigma Y0
    # lambda_0 = alpha_t*(1 - alpha_t)*betas_tiled_m_1
    # lambda_1 = (1 - alpha_t)**2*betas_tiled_m_1 + alpha_t*(1 - alpha_t)*(betas_bar_m_1 - betas_tiled_m_1)*gx - sigma_theta*(alpha_t*betas_tiled_m_1 + alpha_t*(1 - alpha_t))
    # lambda_2 = gx**2*(1 - alpha_t)**2*(betas_bar_m_1 - betas_tiled_m_1) - sigma_theta*gx*(alpha_t*betas_bar_m_1 - alpha_t*betas_tiled_m_1 + (1 - alpha_t)**2)
    # sigma_y0_hat = (-lambda_1 + ((lambda_1)**2 - 4*lambda_0*lambda_2).sqrt()  )/(2*lambda_0)
    # noise = (betas_bar - betas_tiled)*gx + betas_tiled*sigma_y0_hat
    noise = (betas_bar - betas_tiled)*gx + betas_tiled*gx
    
    # assert (noise > 0).all()
    # noise[noise<=0] = gx[noise<=0]
    
    # at, at_bar, at_tilde, Sigma_1, Sigma_2 = cal_sigma12(alphas, alphas_cumprod,alphas_cumprod_sum,alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled_m_1_all, betas_bar_m_1_all, gx, gx, t)
    
    
    # y_0 reparameterization
    y_0_reparam = 1 / sqrt_alpha_bar_t * (
            y - (1 - sqrt_alpha_bar_t) * y_T_mean - eps_theta * torch.sqrt(noise))
    y_t_m_1 = y_0_reparam.to(device)
    return y_t_m_1


def p_sample_loop(model, x, x_mark, y_0_hat, gx, y_T_mean, n_steps, alphas, one_minus_alphas_bar_sqrt, alphas_cumprod,alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled, betas_bar, betas_tiled_m_1, betas_bar_m_1):
    device = next(model.parameters()).device
    z = torch.randn_like(y_T_mean).to(device) # sample 
    cur_y = torch.sqrt(gx) * z + y_T_mean  # sample y_T
    
    y_p_seq = [cur_y]
    for t in reversed(range(1, n_steps)):  # t from T to 2
        y_t = cur_y
        cur_y = p_sample(model, x, x_mark, y_t, y_0_hat, gx, y_T_mean, t, alphas, one_minus_alphas_bar_sqrt,alphas_cumprod,alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled, betas_bar, betas_tiled_m_1, betas_bar_m_1)  # y_{t-1}
        y_p_seq.append(cur_y)
    assert len(y_p_seq) == n_steps
    y_0 = p_sample_t_1to0(model, x, x_mark, y_p_seq[-1], y_0_hat, gx, y_T_mean, one_minus_alphas_bar_sqrt,alphas,alphas_cumprod,alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled, betas_bar, betas_tiled_m_1, betas_bar_m_1)
    y_p_seq.append(y_0)
    return y_p_seq


# Evaluation with KLD
def kld(y1, y2, grid=(-20, 20), num_grid=400):
    y1, y2 = y1.numpy().flatten(), y2.numpy().flatten()
    p_y1, _ = np.histogram(y1, bins=num_grid, range=[grid[0], grid[1]], density=True)
    p_y1 += 1e-7
    p_y2, _ = np.histogram(y2, bins=num_grid, range=[grid[0], grid[1]], density=True)
    p_y2 += 1e-7
    return (p_y1 * np.log(p_y1 / p_y2)).sum()

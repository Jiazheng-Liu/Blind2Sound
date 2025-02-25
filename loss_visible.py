import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal


def calc_covariance(noisy, dn_out, exp_out):
    # calculate sigma_x
    n, c, h, w = noisy.shape
    A_c1 = dn_out  # Components ot triangular A.
    A_c2 = exp_out  # Components ot triangular A.
    if c == 1:
        sigma_x = A_c1*A_c2  # N1HW
    elif c == 3:
        A_c1 = A_c1.permute(0, 2, 3, 1)  # NHWC
        A_c2 = A_c2.permute(0, 2, 3, 1)  # NHWC
        A_m1 = torch.zeros(size=(n, h, w, c, c), dtype=A_c1.dtype, device=A_c1.device)
        A_m2 = torch.zeros(size=(n, h, w, c, c), dtype=A_c2.dtype, device=A_c2.device)
        # Calculate A^T * A
        A_m1[..., [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]] = A_c1
        A_m2[..., [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]] = A_c2
        sigma_x = torch.matmul(A_m1, A_m2.transpose(-1, -2))  # NHWCC
    elif c == 4:
        A_c1 = A_c1.permute(0, 2, 3, 1)  # NHWC
        A_c2 = A_c2.permute(0, 2, 3, 1)  # NHWC
        A_m1 = torch.zeros(size=(n, h, w, c, c), dtype=A_c1.dtype, device=A_c1.device)
        A_m2 = torch.zeros(size=(n, h, w, c, c), dtype=A_c2.dtype, device=A_c2.device)
        # Calculate A^T * A
        A_m1[..., [0, 0, 0, 0, 1, 1, 1, 2, 2, 3], [0, 1, 2, 3, 1, 2, 3, 2, 3, 3]] = A_c1
        A_m2[..., [0, 0, 0, 0, 1, 1, 1, 2, 2, 3], [0, 1, 2, 3, 1, 2, 3, 2, 3, 3]] = A_c2
        sigma_x = torch.matmul(A_m1, A_m2.transpose(-1, -2))  # NHWCC
    return sigma_x

def calc_wb_pme(noisy, mu_x, mu_e, sigma_x, sigma_e, est_out, beta, noise_type):
    device = noisy.device
    inc = noisy.shape[1]    
    I = torch.eye(inc, device=device).reshape(1, 1, 1, inc, inc)
    Ieps = I * 1e-6

    noisy2 = noisy.permute(0, 2, 3, 1)  # NHWC
    mu_x2 = mu_x.permute(0, 2, 3, 1)  # NHWC
    mu_e2 = mu_e.permute(0, 2, 3, 1)  # NHWC

    noise_var_x = calc_noise_var(est_out, mu_x, noise_type) # NHWCC
    sigma_xn = noise_var_x.permute(0, 2, 3, 1).unsqueeze(
        -1
    ) * I  # NHWC1 * NHWCC = NHWCC
    noise_var_e = calc_noise_var(est_out, mu_e, noise_type) # NHWCC
    sigma_en = noise_var_e.permute(0, 2, 3, 1).unsqueeze(
        -1
    ) * I  # NHWC1 * NHWCC = NHWCC
    nx_w = 1 / (1 + beta)
    ne_w = beta / (1 + beta)
    sigma_yx = (sigma_x + sigma_xn) * nx_w**2 # NHWCC
    sigma_ye = (sigma_e + sigma_en) * ne_w**2 # NHWCC
    MG_X = MultivariateNormal(mu_x2, sigma_yx + Ieps) 
    MG_E = MultivariateNormal(mu_e2, sigma_ye + Ieps) 
    g_x = MG_X.log_prob(noisy2).exp() * nx_w # NHW
    g_e = MG_E.log_prob(noisy2).exp() * ne_w # NHW
    wb_x = g_x / (g_x + g_e) # NHW
    wb_e = 1 - wb_x # NHW
    mu_f = wb_x.unsqueeze(1) * mu_x + wb_e.unsqueeze(1) * mu_e
    return mu_f

def calc_sigma(noisy, cov_map):
    # calculate sigma_x
    n, c, h, w = noisy.shape
    A_c = cov_map  # Components ot triangular A.
    if c == 1:
        sigma_x = A_c**2  # N1HW
    elif c == 3:
        A_c = A_c.permute(0, 2, 3, 1)  # NHWC
        A_m = torch.zeros(size=(n, h, w, c, c), dtype=A_c.dtype, device=A_c.device)
        # Calculate A^T * A
        A_m[..., [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]] = A_c
        sigma_x = torch.matmul(A_m, A_m.transpose(-1, -2))  # NHWCC
    elif c == 4:
        A_c = A_c.permute(0, 2, 3, 1)  # NHWC
        A_m = torch.zeros(size=(n, h, w, c, c), dtype=A_c.dtype, device=A_c.device)
        # Calculate A^T * A
        A_m[..., [0, 0, 0, 0, 1, 1, 1, 2, 2, 3], [0, 1, 2, 3, 1, 2, 3, 2, 3, 3]] = A_c
        sigma_x = torch.matmul(A_m, A_m.transpose(-1, -2))  # NHWCC
    return sigma_x


def calc_mu_sigma(noisy, dn_out):
    inc = noisy.shape[1]
    diag_cov = True if dn_out.shape[1] == 6 else False
    # calculate mu_x and sigma_x
    mu_x = dn_out[:, :inc, :, :]  # Means (NCHW)
    A_c = dn_out[:, inc:, :, :]  # Components ot triangular A.
    n, c, h, w = mu_x.shape
    if inc == 1:
        sigma_x = A_c**2  # N1HW
    elif inc == 3:
        A_c = A_c.permute(0, 2, 3, 1)  # NHWC
        A_m = torch.zeros(size=(n, h, w, c, c), dtype=A_c.dtype, device=A_c.device)
        if diag_cov:  # diagonal_covariance: True
            A_m[..., [0, 1, 2], [0, 1, 2]] = A_c
        else:
            # Calculate A^T * A
            A_m[..., [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]] = A_c
        sigma_x = torch.matmul(A_m, A_m.transpose(-1, -2))  # NHWCC
    elif inc == 4:
        A_c = A_c.permute(0, 2, 3, 1)  # NHWC
        A_m = torch.zeros(size=(n, h, w, c, c), dtype=A_c.dtype, device=A_c.device)
        if diag_cov:  # diagonal_covariance: True
            A_m[..., [0, 1, 2, 3], [0, 1, 2, 3]] = A_c
        else:
            # Calculate A^T * A
            A_m[..., [0, 0, 0, 0, 1, 1, 1, 2, 2, 3], [0, 1, 2, 3, 1, 2, 3, 2, 3, 3]] = A_c
        sigma_x = torch.matmul(A_m, A_m.transpose(-1, -2))  # NHWCC
    return mu_x, sigma_x


def calc_noise_var(est_out, mu_x, noise_type):
    device = est_out.device
    eps = torch.tensor(1e-3, dtype=torch.float32).to(device)

    # Distill noise parameters from learned/known data.
    if noise_type == "gauss":
        noise_var = est_out**2
    elif (
        noise_type == "poisson"
    ):  # Simple signal-dependent Poisson approximation [Hasinoff 2012].
        noise_var = torch.maximum(mu_x, eps) * est_out  # NCHW
    elif noise_type == "pg":
        noise_var = est_out[:, 0:1, :, :]**2 + torch.maximum(mu_x, eps) * est_out[:, 1:2, :, :]  # NCHW

    return noise_var 


def likelihood_loss(noisy, mu_x, sigma_x, sigma_n, factor=1, train=True):
    device = noisy.device
    inc = noisy.shape[1]

    if not factor == 1:
        noisy = noisy * factor
        mu_x = mu_x * factor
        sigma_x = sigma_x * factor
        sigma_n = sigma_n * factor

    # # MG
    # sigma_n = (1 + beta**2) / (1 + beta)**2 * noise_var

    I = torch.eye(inc, device=device).reshape(1, 1, 1, inc, inc)
    Ieps = I * 1e-6
    zero64 = torch.tensor(1e-6, dtype=torch.float32).to(device)
    # Helpers
    def batch_mvmul(m, v):  # Batched (M * v)
        # return torch.sum(m * v.unsqueeze(-2), dim=-1)
        return torch.matmul(m, v.unsqueeze(-1)).squeeze(-1)

    def batch_vtmv(v, m):  # Batched (v^T * M * v)
        # return torch.sum(v.unsqueeze(-1) * v.unsqueeze(-2) * m, dim=[-2, -1])
        return (
            torch.matmul(torch.matmul(v.unsqueeze(-2), m), v.unsqueeze(-1))
            .squeeze(-1)
            .squeeze(-1)
        )

    # Negative log-likelihood loss and posterior mean estimation.
    if inc == 1:
        if train:
            sigma_n = sigma_n  # N111 / N1HW
            sigma_y = sigma_x + sigma_n  # N1HW. Total variance.
            loss_out = ((noisy - mu_x) ** 2) / sigma_y + torch.log(sigma_y)  # N1HW
        else:
            sigma_n = sigma_n  # N111 / N1HW
            pme_out = (noisy * sigma_x + mu_x * sigma_n) / (
                sigma_x + sigma_n
            )  # N1HW
    else:
        mu_x2 = mu_x.permute(0, 2, 3, 1)  # NHWC
        noisy_in2 = noisy.permute(0, 2, 3, 1)  # NHWC
        if train:
            # Training loss.
            sigma_n = sigma_n.permute(0, 2, 3, 1).unsqueeze(
                -1
            ) * I  # NHWC1 * NHWCC = NHWCC
            sigma_y = (
                sigma_x + sigma_n
            )  # NHWCC, total covariance matrix. Cannot be singular because sigma_n is at least a small diagonal.
            sigma_y_inv = torch.inverse(sigma_y)  # NHWCC
            diff = noisy_in2 - mu_x2  # NHWC
            diff = -0.5 * batch_vtmv(diff, sigma_y_inv)  # NHW
            dets = torch.linalg.det(sigma_y)  # NHW
            dets = torch.maximum(
                zero64, dets
            )  # NHW. Avoid division by zero and negative square roots.
            loss_out = 0.5 * torch.log(dets) - diff  # NHW
        else:
            # Posterior mean estimate.
            sigma_n = sigma_n.permute(0, 2, 3, 1).unsqueeze(
                -1
            ) * I  # NHWC1 * NHWCC = NHWCC
            sigma_x_inv = torch.inverse(sigma_x + Ieps)  # NHWCC
            sigma_n_inv = torch.inverse(sigma_n + Ieps)  # NHWCC
            pme_c1 = torch.inverse(sigma_x_inv + sigma_n_inv + Ieps)  # NHWCC
            pme_c2 = batch_mvmul(sigma_x_inv, mu_x2)  # NHWCC * NHWC -> NHWC
            pme_c2 = pme_c2 + batch_mvmul(sigma_n_inv, noisy_in2)  # NHWC
            pme_out = batch_mvmul(pme_c1, pme_c2)  # NHWC
            pme_out = pme_out.permute(0, 3, 1, 2)  # NCHW

    if train:
        return loss_out
    else:
        return pme_out / factor

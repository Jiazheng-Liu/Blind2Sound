import sys
import numpy as np
import math
import torch
from skimage import measure


def get_PSNR(X, X_hat):
    mse = np.mean((X - X_hat) ** 2)
    test_PSNR = 10 * math.log10(1 / mse)
    return test_PSNR


def get_SSIM(X, X_hat):
    test_SSIM = measure.compare_ssim(
        np.transpose(X, (1, 2, 0)),
        np.transpose(X_hat, (1, 2, 0)),
        data_range=X.max() - X.min(),
        multichannel=True,
    )
    return test_SSIM
    

def im2patch(x, psize=8, stride=4):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, psize, stride=stride)
    patches = unfolded_x.view(n, c, psize**2, -1)
    return patches


def chen_estimate(im, psize=8, stride=4):
    device = im.device
    pch = im2patch(im, psize, stride)
    n, c, d, n_pch = pch.shape
    mu = torch.mean(pch, dim=-1, keepdim=True)
    umat = torch.eye(d, device=device)
    x = pch - mu
    sig_x = torch.matmul(x, x.permute(0,1,3,2)) / n_pch
    e_value = torch.linalg.eigvalsh(sig_x, UPLO='U')
    ## old version
    # e_value, _ = torch.symeig(sig_x, eigenvectors=True)

    triangle = torch.ones((n, c, d, d), device=device)
    triangle = torch.tril(triangle)
    e_value_d = e_value.unsqueeze(-1).repeat(1,1,1,d)*umat
    sig_matrix = torch.matmul(triangle, e_value_d)

    # calculate whole threshold value at a single time
    num_vec = torch.arange(d, device=device) + 1
    num_vec = num_vec.to(dtype=torch.float32)
    sum_arr = torch.sum(sig_matrix, dim=-1)
    tau_arr = sum_arr / num_vec
    tau_arr_d = tau_arr.unsqueeze(-1).repeat(1,1,1,d)*umat
    tau_mat = torch.matmul(tau_arr_d, triangle)

    # find median value with masking scheme:
    big_bool = torch.sum(sig_matrix > tau_mat, axis=-1)
    small_bool = torch.sum(sig_matrix < tau_mat, axis=-1)
    mask = (big_bool == small_bool).to(dtype=torch.float32).to(device)
    tau_chen = torch.max(mask * tau_arr, dim=-1).values # NC
    # tau_chen = torch.mean(tau_chen, dim=1)
    sigma = torch.sqrt(tau_chen)
    return sigma


def anscombe(z, alpha):
    if len(alpha.shape)==1:
        alpha = alpha.view(*(alpha.shape),1,1,1)
    elif len(alpha.shape)==2:
        alpha = alpha.view(*(alpha.shape),1,1)
    alpha = torch.ones_like(z) * alpha
    f = (2.0) * torch.sqrt(
        torch.max(z + 3.0 / 8.0, torch.zeros_like(z))
    )
    return f


def gat(z, sigma, alpha):
    if len(sigma.shape)==1:
        sigma = sigma.view(*(sigma.shape),1,1,1)
        alpha = alpha.view(*(alpha.shape),1,1,1)
    elif len(sigma.shape)==2:
        sigma = sigma.view(*(sigma.shape),1,1)
        alpha = alpha.view(*(alpha.shape),1,1)
    alpha = torch.ones_like(z) * alpha
    sigma = torch.ones_like(z) * sigma
    z = z / alpha
    _sigma = sigma / alpha
    f = (2.0) * torch.sqrt(
        torch.max(z + (3.0 / 8.0) + _sigma**2, torch.zeros_like(z))
    )
    return f


def inv_gat(z, sigma, alpha, method="closed_form"):
    if len(sigma.shape)==1:
        sigma = sigma.view(*(sigma.shape),1,1,1)
        alpha = alpha.view(*(alpha.shape),1,1,1)
    elif len(sigma.shape)==2:
        sigma = sigma.view(*(sigma.shape),1,1)
        alpha = alpha.view(*(alpha.shape),1,1)
    _sigma = sigma / alpha
    if method == "closed_form":
        # # eps bias
        # eps = torch.tensor(3e-1, device=z.device)
        # z = torch.max(z, eps)
        exact_inverse = (
            (z / 2.0) ** 2.0
            + 0.25 * math.sqrt(1.5) * z ** (-1.0)
            - 11.0 / 8.0 * z ** (-2.0)
            + 5.0 / 8.0 * math.sqrt(1.5) * z ** (-3.0)
            - 1.0 / 8.0
            - _sigma ** 2
        )        
        exact_inverse = torch.max(exact_inverse, torch.zeros_like(exact_inverse))
    elif method == "asym":
        exact_inverse = (z / 2.0) ** 2 - 1.0 / 8.0 - _sigma ** 2
    else:
        raise NotImplementedError("Only supports the closed-form")
    exact_inverse *= alpha
    # exact_inverse = torch.clip(exact_inverse, 0., 1.)
    return exact_inverse


def norm_tensor(z):
    batch = z.shape[0]
    z_min = torch.min(z.view(batch,-1), dim=-1).values
    z_max = torch.max(z.view(batch,-1), dim=-1).values
    z_min = z_min.view(batch, 1, 1, 1)
    z_max = z_max.view(batch, 1, 1 ,1)

    z_nm = (z - z_min) / (z_max - z_min)
    sigma = 1 / (z_max - z_min)
    return z_nm, sigma, z_min, z_max


def denorm_tensor(z_nm, z_min, z_max):
    z = z_nm * (z_max - z_min) + z_min
    return z


def inv_gat_numpy(z, sigma, alpha, method="closed_form"):
    _sigma = sigma / alpha
    if method == "closed_form":
        exact_inverse = (
            np.power(z / 2.0, 2.0)
            + 0.25 * np.sqrt(1.5) * np.power(z, -1.0)
            - 11.0 / 8.0 * np.power(z, -2.0)
            + 5.0 / 8.0 * np.sqrt(1.5) * np.power(z, -3.0)
            - 1.0 / 8.0
            - _sigma**2
        )
        exact_inverse = np.maximum(0.0, exact_inverse)
    elif method == "asym":
        exact_inverse = (z / 2.0) ** 2 - 1.0 / 8.0 - _sigma**2
    else:
        raise NotImplementedError("Only supports the closed-form")
    exact_inverse *= alpha
    return exact_inverse


def normalize_after_gat_torch(transformed):
    min_transform = torch.min(transformed)
    max_transform = torch.max(transformed)

    transformed = (transformed - min_transform) / (max_transform - min_transform)
    transformed_sigma = 1 / (max_transform - min_transform)
    transformed_sigma = torch.ones_like(transformed) * (transformed_sigma)
    return transformed, transformed_sigma, min_transform, max_transform

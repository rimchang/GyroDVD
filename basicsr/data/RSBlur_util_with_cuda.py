import numpy as np
import torch
import random
import pickle as pkl


def rgb2lin_pt(x):
    return torch.pow(x, 2.2)

def lin2rgb_pt(x):
    return torch.pow(x, 1/2.2)

def apply_cmatrix(img, matrix):
    # img : (b, h, w, c)
    # matrix : (b, 3, 3)

    """
    same results below code
    img_reshape = img.reshape(1, h*w, 3)
    out2 = torch.matmul(img_reshape, matrix.permute(0, 2, 1))
    out2 = out2.reshape(1, h, w, 3)
    """

    images = img[:, :, :, None, :]  # (b, h, w, 1, c)
    ccms = matrix[:, None, None, :, :]  # (1, 1, 1, 3, 3)
    out = torch.sum(images * ccms, -1)  # (h, w, 3)

    return out


def mosaic_bayer(image, pattern):
    """Extracts RGGB Bayer planes from an RGB image."""
    shape = image.shape

    if pattern == 'RGGB':
        red = image[:, 0::2, 0::2, 0]  # (b, h/2, w/2)
        green_red = image[:, 0::2, 1::2, 1]
        green_blue = image[:, 1::2, 0::2, 1]
        blue = image[:, 1::2, 1::2, 2]
    elif pattern == 'BGGR':
        red = image[:, 0::2, 0::2, 2]  # (b, h/2, w/2)
        green_red = image[:, 0::2, 1::2, 1]
        green_blue = image[:, 1::2, 0::2, 1]
        blue = image[:, 1::2, 1::2, 0]
    elif pattern == 'GRBG':
        red = image[:, 0::2, 0::2, 1]  # (b, h/2, w/2)
        green_red = image[:, 0::2, 1::2, 0]
        green_blue = image[:, 1::2, 0::2, 2]
        blue = image[:, 1::2, 1::2, 1]
    elif pattern == 'GBRG':
        red = image[:, 0::2, 0::2, 1]  # (b, h/2, w/2)
        green_red = image[:, 0::2, 1::2, 2]
        green_blue = image[:, 1::2, 0::2, 0]
        blue = image[:, 1::2, 1::2, 1]

    image = torch.stack((red, green_red, green_blue, blue), dim=3)  # (b, h/2, w/2, 4)
    image = image.view(-1, shape[1] // 2, shape[2] // 2, 4)

    return image


def add_Poisson_noise_random(img, beta1, beta2):

    random_K_v = beta1.view(-1, 1, 1, 1).to(img.device)

    noisy_img = torch.poisson(img / random_K_v)
    noisy_img = noisy_img * random_K_v

    random_other = beta2.view(-1, 1, 1, 1).to(img.device)
    noisy_img = noisy_img + (torch.normal(torch.zeros_like(noisy_img), std=1) * torch.sqrt(random_other))

    return noisy_img


def WB_img(img, pattern, fr_now, fb_now):
    red_gains = fr_now
    blue_gains = fb_now
    green_gains = torch.ones_like(red_gains)

    if pattern == 'RGGB':
        gains = torch.cat([red_gains, green_gains, green_gains, blue_gains], dim=1)
    elif pattern == 'BGGR':
        gains = torch.cat([blue_gains, green_gains, green_gains, red_gains], dim=1)
    elif pattern == 'GRBG':
        gains = torch.cat([green_gains, red_gains, blue_gains, green_gains], dim=1)
    elif pattern == 'GBRG':
        gains = torch.cat([green_gains, blue_gains, red_gains, green_gains], dim=1)

    gains = gains[:, None, None, :]
    img = img * gains

    return img
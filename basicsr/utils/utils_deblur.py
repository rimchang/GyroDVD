import torch
import basicsr.utils.median_pool as median_pool

# --------------------------------
def get_uperleft_denominator(img, kernel, scale: float = 8.0 / 3.0 / 10.0):
    ker_f = convert_psf2otf(kernel, img.size())  # DFT of kernel
    nsr = wiener_filter_para(img, scale)
    denominator = inv_fft_kernel_est(ker_f, nsr)
    img1 = img

    # 구식: numerator = torch.rfft(img1, 3, onesided=False)
    numerator = torch.fft.fftn(img1, dim=(2, 3))  # full complex FFT

    #print(denominator.shape, numerator.shape, nsr.shape, ker_f.shape, nsr)
    deblur = deconv(denominator, numerator)
    #print(deblur.shape)
    return deblur


def deconv_logfft(x, k, fftshift=False, eps = 1e-12):

    x_fft = torch.fft.fft2(x, dim=(-2, -1))
    if fftshift:
        x_fft = torch.fft.fftshift(x_fft, dim=(-2, -1))

    x_fft_mag = torch.abs(x_fft)
    x_fft_mag_log = torch.log(x_fft_mag + eps)
    x_fft_ang = torch.angle(x_fft)

    # PSF -> OTF (중심을 (0,0)로 이동 후 FFT)
    k_fft = convert_psf2otf(k, x.size(), rot180=True)
    if fftshift:
        k_fft = torch.fft.fftshift(k_fft, dim=(-2, -1))

    k_fft_mag = torch.abs(k_fft)
    k_fft_mag_log = torch.log(k_fft_mag + eps)
    k_fft_ang = torch.angle(k_fft)

    out_fft_mag_log = x_fft_mag_log - k_fft_mag_log
    out_fft_mag = torch.exp(out_fft_mag_log)
    out_fft_ang = x_fft_ang - k_fft_ang

    out_fft = torch.polar(out_fft_mag, out_fft_ang)
    if fftshift:
        out_fft = torch.fft.ifftshift(out_fft, dim=(-2, -1))
    out = torch.fft.ifft2(out_fft, dim=(-2, -1)).real

    return out


def wiener_filter_para(_input_blur, eps: float = 1e-12, scale: float = 8.0 / 3.0 / 10.0):
    """
    NSR ≈ Var(noise) / Var(signal)
    - noise ≈ median_pool(x) - x
    - signal ≈ x
    반환 shape: [B, C, 1, 1]
    """
    # median filter (채널별/이미지 경계 보정)
    median_filter = median_pool.MedianPool2d(kernel_size=3, padding=3 // 2)(_input_blur)
    diff = median_filter - _input_blur  # noise 추정

    B, C, H, W = _input_blur.shape
    num = H * W

    # 분산(variance) 계산: 채널별 [B,C,1,1]
    mean_n = diff.mean(dim=(2, 3), keepdim=True)
    var_n = ((diff - mean_n) ** 2).sum(dim=(2, 3), keepdim=True) / max(num - 1, 1)

    mean_x = _input_blur.mean(dim=(2, 3), keepdim=True)
    var_x = ((_input_blur - mean_x) ** 2).sum(dim=(2, 3), keepdim=True) / max(num - 1, 1)

    # NSR: Var(noise) / Var(signal) (+ 안정화 항)
    NSR = (var_n / (var_x + eps)) * scale

    # (옵션) 과도한 값 방지
    NSR = NSR.clamp(min=0.0)

    return NSR


# --------------------------------
def inv_fft_kernel_est(ker_f, NSR):
    inv_denominator = ker_f.real * ker_f.real + ker_f.imag * ker_f.imag + NSR
    inv_ker_f = (ker_f.conj() / inv_denominator)
    return inv_ker_f


# --------------------------------
def deconv(inv_ker_f, fft_input_blur):
    # element-wise multiplication (complex)
    deblur_f = inv_ker_f * fft_input_blur
    # 구식: torch.irfft(..., onesided=False)
    deblur = torch.fft.ifftn(deblur_f, dim=(2, 3)).real
    return deblur


# --------------------------------
def convert_psf2otf(ker, size, rot180=False):
    psf = torch.zeros(size, device=ker.device)
    centre = ker.shape[2] // 2 + 1
    psf[:, :, :centre, :centre] = ker[:, :, (centre - 1):, (centre - 1):]
    psf[:, :, :centre, -(centre - 1):] = ker[:, :, (centre - 1):, :(centre - 1)]
    psf[:, :, -(centre - 1):, :centre] = ker[:, :, :(centre - 1), (centre - 1):]
    psf[:, :, -(centre - 1):, -(centre - 1):] = ker[:, :, :(centre - 1), :(centre - 1)]

    if rot180:
        psf = torch.rot90(psf, k=2, dims=[-2, -1])

    # 구식: otf = torch.rfft(psf, 3, onesided=False)
    otf = torch.fft.fftn(psf, dim=(2, 3))  # complex tensor
    return otf


def trajectories2kernel(blur_kernels, ker_size, bins=129):
    # blur_kernels: (num_patches, T, 2)
    device = blur_kernels.device
    dtype = blur_kernels.dtype

    num_p, t, _ = blur_kernels.shape

    x, y = blur_kernels[:, :, 0], blur_kernels[:, :, 1]
    x, y = x + ker_size // 2, y + ker_size // 2

    x_left, y_upper = torch.floor(x).int(), torch.floor(y).int()
    x_right = x_left + 1
    y_bottom = y_upper + 1

    w1, w2 = x - x_left, x_right - x
    h1, h2 = y_bottom - y, y - y_upper

    a = h1 / (h1 + h2)
    b = h2 / (h1 + h2)
    p = w1 / (w1 + w2)
    q = w2 / (w1 + w2)

    weight_left_upper = (q * a).to(dtype)
    weight_left_bottom = (q * b).to(dtype)
    weight_right_upper = (p * a).to(dtype)
    weight_right_bottom = (p * b).to(dtype)

    # int -> long, 경계 clamp 추가 (중요)
    current_x_left = x_left.long().clamp_(0, ker_size - 1)
    current_x_right = x_right.long().clamp_(0, ker_size - 1)
    current_y_upper = y_upper.long().clamp_(0, ker_size - 1)
    current_y_bottom = y_bottom.long().clamp_(0, ker_size - 1)

    # 2D 인덱스를 1D 선형 인덱스로 변환해서 scatter_add_
    # shape: (T, H, W) -> (-1,)
    lin_left_bottom = (current_y_bottom * ker_size + current_x_left)
    lin_right_bottom = (current_y_bottom * ker_size + current_x_right)
    lin_left_upper = (current_y_upper * ker_size + current_x_left)
    lin_right_upper = (current_y_upper * ker_size + current_x_right)

    vals_left_bottom = weight_left_bottom  # .view(-1)
    vals_right_bottom = weight_right_bottom  # .view(-1)
    vals_left_upper = weight_left_upper  # .view(-1)
    vals_right_upper = weight_right_upper  # .view(-1)

    # 평면 커널을 만든 뒤 scatter_add_로 누적
    kernels_flat = torch.zeros(num_p, ker_size * ker_size, dtype=dtype, device=device)
    kernels_flat.scatter_add_(1, lin_left_bottom, vals_left_bottom)
    kernels_flat.scatter_add_(1, lin_right_bottom, vals_right_bottom)
    kernels_flat.scatter_add_(1, lin_left_upper, vals_left_upper)
    kernels_flat.scatter_add_(1, lin_right_upper, vals_right_upper)

    kernels = kernels_flat.view(num_p, ker_size, ker_size)
    nomalize_kernels = kernels / kernels.sum(axis=[1, 2], keepdim=True)
    return nomalize_kernels

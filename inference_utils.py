import os
import torch
import numpy as np
import torch.nn.functional as F

from gtsam import PreintegratedImuMeasurements
from scipy.ndimage import distance_transform_edt



# ---------------------------------------------------------
# Camera intrinsic conversion
# ---------------------------------------------------------
# Construct per-frame intrinsics for 1080x1920 resolution
def constructIntrinsic(fx, fy, cx, cy, H, W):
    ori_H, ori_W = 4080, 3072
    ori_fx = ori_fy = fx
    ori_cx, ori_cy = cx, cy

    # 1) Center crop to 16:9 aspect ratio
    crop_W = round(ori_H * 9 / 16)
    crop_H = ori_H

    # Offset of the cropped region in the original image coordinates
    off_x = (ori_W - crop_W) / 2.0
    off_y = (ori_H - crop_H) / 2.0

    # 2) Resize scale from cropped image to inference resolution
    sx = W / crop_W
    sy = H / crop_H

    # 3) Convert intrinsics to resized image coordinates
    fx = ori_fx * sx
    fy = ori_fy * sy

    cx_crop = ori_cx - off_x
    cy_crop = ori_cy - off_y

    cx = cx_crop * sx
    cy = cy_crop * sy

    K_list = []
    K_inv_list = []
    for fx, fy, cx, cy in zip(fx, fy, cx, cy):
        K_np = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ]).astype('float32')

        K = torch.from_numpy(K_np)
        K_inv = torch.inverse(K)

        K_list.append(K)
        K_inv_list.append(K_inv)

    K = torch.stack(K_list, dim=0)
    K_inv = torch.stack(K_inv_list, dim=0)

    return K, K_inv

# ---------------------------------------------------------
# Optical flow utility
# ---------------------------------------------------------
def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear' or 'nearest4'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.


    Returns:
        Tensor: Warped image or feature map.
    """
    n, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, dtype=x.dtype, device=x.device),
                                    torch.arange(0, w, dtype=x.dtype, device=x.device))
    grid = torch.stack((grid_x, grid_y), 2)  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)

    output = F.grid_sample(x, vgrid_scaled , mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    return output


# ---------------------------------------------------------
# Optical flow consistency check
# ---------------------------------------------------------
def fbConsistencyCheck(flow_fw, flow_bw):
    flow_fw_warped = flow_warp(flow_fw, flow_bw.permute(0, 2, 3, 1), padding_mode='border')  # wb(wf(x))
    flow_diff_fw = flow_bw + flow_fw_warped  # wf + wb(wf(x))

    abs_sum = torch.norm(flow_diff_fw, dim=1, keepdim=True)

    return abs_sum


# ---------------------------------------------------------
# Timestamp parser
# ---------------------------------------------------------
def getCenterStamp(path):
    path_split = os.path.basename(path).split('_')
    start_stamp = int(path_split[1])
    end_stamp = int(path_split[3])
    center_stamp = start_stamp + (end_stamp - start_stamp) // 2

    return center_stamp


# ---------------------------------------------------------
# Time conversion
# ---------------------------------------------------------
def nano2sec(nanoseconds):
    """
    Convert nanoseconds to seconds.

    Args:
        nanoseconds (int or float or np.ndarray): Time in nanoseconds.

    Returns:
        float or np.ndarray: Time in seconds.
    """
    return nanoseconds * 1e-9


def sec2nano(sec):
    return sec * 1e9


# ---------------------------------------------------------
# Gyro-only preintegration between two timestamps
# ---------------------------------------------------------
# This integrates angular velocity samples from ts_a to ts_b using GTSAM's
# PreintegratedImuMeasurements. Acceleration is set to zero because only the
# relative rotation deltaRij() is needed for rotational blur trajectories.
def integrate_only_gyro(preint, ts_a, ts_b, gyro_ts, gyro_w):
    # 1) idx_start < ts_a < indx_end < ts_b 인 구간 인덱스 찾기
    idx_start = np.searchsorted(gyro_ts, ts_a, side='right') - 1  # a 포함 전 샘플
    idx_end = np.searchsorted(gyro_ts, ts_b, side='right') - 1  # b 직전 샘플

    # 2) ts_a -  gyro_ts[idx_start+1] 사이 부분 적분
    t0, t1 = ts_a, gyro_ts[idx_start + 1]
    if t1 <= t0:  # 만약 같은 샘플이면 skip
        dt = 0
    else:
        dt = t1 - t0
        omega = gyro_w[idx_start]  # 이전 샘플 각속도를 dt 만큼 적분
        preint.integrateMeasurement(np.zeros(3), omega.reshape(-1), dt)

    # 3) 중간 샘플 간 적분
    for k in range(idx_start + 1, idx_end):
        t0, t1 = gyro_ts[k], gyro_ts[k + 1]
        dt = t1 - t0
        omega = gyro_w[k]  # 구간 내 각속도
        preint.integrateMeasurement(np.zeros(3), omega.reshape(-1), dt)

    # 4) 마지막 샘플–b 사이 부분 적분
    t0, t1 = gyro_ts[idx_end], ts_b
    if t1 <= t0:
        dt = 0
    else:
        dt = t1 - t0
        omega = gyro_w[idx_end]  # 마지막 샘플 속도로 보간
        preint.integrateMeasurement(np.zeros(3), omega.reshape(-1), dt)

    return preint.deltaRij()


# ---------------------------------------------------------
# Convert rotation matrices into pixel-wise motion trajectories
# ---------------------------------------------------------
# For each pixel:
#   1. Back-project pixel coordinates to camera rays using K^{-1}.
#   2. Rotate rays using gyro-derived rotations.
#   3. Project rotated rays back using K.
#   4. Subtract original pixel coordinates to obtain 2D flow/trajectory.
# The output is used as the rotational blur kernel input to GyroDVD.
def compute_traj_from_rots(R, coordinate, K, K_inv):
    # coordinate : cropped coordinates (b, 1, 3, h, w)
    # Rot : rotation matrices computed from gyo (b, N, 8, 3, 3)
    b, _, _, h, w = coordinate.shape
    b, n_seq, _, _, _ = R.shape

    pix = coordinate.view(b, 3, h*w)

    # 2. Compute normalized rays in camera coordinates
    rays = K_inv @ pix  # (b, 3, 3)x(b, 3, hxw) => (b, 3, H*W)

    R = R.permute(0, 1, 2, 4, 3) # transform R_wc to R_cw
    rot_rays = R @ rays.view(b, -1, 1, 3, h*w) # (1, N, 8, 3, 3)x(1, 1, 1, 3, hxw) => (b, N, 8, 3, hxw)
    rot_rays = rot_rays / rot_rays[:,:,:,2:3, :]
    proj = K.view(b, -1, 1, 3, 3) @ rot_rays # (1, 1, 1, 3, 3)x(b, N, 8, 3, hxw) => (b, N, 8, 3, hxw)
    proj_xy = proj[:,:,:,:2,:] # (b, N, 8, 2, hxw)

    trajectories = proj_xy - coordinate[:,:,:2,:,:].view(1, 1, 1, 2, -1)
    trajectories = trajectories.reshape(b, n_seq, -1, h, w) # (b, n, 8x2, h, w)

    return trajectories


# ---------------------------------------------------------
# Pixel coordinate grid construction
# ---------------------------------------------------------
# Produces either [x, y] or homogeneous [x, y, 1] coordinates.
# The homogeneous form is required for projection with intrinsic matrices.
def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid



# ---------------------------------------------------------
# Fill invalid tau values using nearest valid neighbors
# ---------------------------------------------------------
# Optical flow can be unreliable around occlusions or inconsistent regions.
# The mask marks reliable pixels. Invalid pixels are filled by copying the
# nearest valid tau value using scipy.ndimage.distance_transform_edt().
def nearest_fill_with_scipy(img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:


    device = img.device
    dtype  = img.dtype
    N, C, H, W = img.shape

    # CPU 변환
    img_np  = img.detach().numpy()          # (N,2,H,W)
    valid   = (mask.detach().numpy() > 0)[0, 0]  # (H,W) bool

    out_np = np.empty_like(img_np)

    if valid.all():
        # 전부 유효: 그대로 반환
        out_np[...] = img_np
    elif (~valid).all():
        # 전부 결측: 최근접 정의 불가 → 여기선 원본 유지(필요시 0 채움 등으로 변경)
        out_np[...] = img_np
    else:
        # 결측 픽셀에서 가장 가까운 유효 픽셀의 인덱스(y,x) 맵
        # inds.shape = (2,H,W)
        edt, inds = distance_transform_edt(~valid, return_indices=True)  # (y_map, x_map)

        # 배치/채널 전체에 동일 인덱스 적용
        # img_np[:, c, y, x] <- img_np[:, c, inds_y[y,x], inds_x[y,x]]
        out_np = img_np[:, :, inds[0], inds[1]]

    return torch.as_tensor(out_np, device=device, dtype=dtype)


# ---------------------------------------------------------
# Patch-wise GyroDVD inference
# ---------------------------------------------------------
# To save memory, we compute final blur kernels for each patches in this fuction
def test_by_patch(net_g, lq, coordinate, rot_mat, masked_forward_tau, maksed_backward_tau, dT, K_cu, K_inv_cu):
    with torch.no_grad():
        size_patch_testing = 512
        overlap_size = 128
        b, t, c, h, w = lq.shape
        stride = size_patch_testing - overlap_size
        h_idx_list = list(range(0, h - size_patch_testing, stride)) + [max(0, h - size_patch_testing)]
        w_idx_list = list(range(0, w - size_patch_testing, stride)) + [max(0, w - size_patch_testing)]

        E = torch.zeros(b, t - 2, c, h, w)
        W = torch.zeros_like(E)
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                
                assert (h_idx % 2) == 0 and (w_idx % 2) == 0 and (size_patch_testing % 2) == 0

                h_idx_tau = h_idx//2
                w_idx_tau = w_idx//2
                size_patch_testing_tau = size_patch_testing // 2
                in_masked_forward_tau = masked_forward_tau[..., h_idx_tau:h_idx_tau + size_patch_testing_tau, w_idx_tau:w_idx_tau + size_patch_testing_tau]
                in_maksed_backward_tau = maksed_backward_tau[..., h_idx_tau:h_idx_tau + size_patch_testing_tau, w_idx_tau:w_idx_tau + size_patch_testing_tau]

                in_masked_forward_tau = F.interpolate(in_masked_forward_tau, scale_factor=2, mode='bilinear', align_corners=True)
                in_maksed_backward_tau = F.interpolate(in_maksed_backward_tau, scale_factor=2, mode='bilinear', align_corners=True)

                # scaling tau with relative exposure difference from center => now these are trajectories from tau 
                center = 4
                all_forward_tau = (in_masked_forward_tau.unsqueeze(0).unsqueeze(2) * torch.abs(dT[:, :center]).reshape(1, t, -1, 1, 1, 1)).float()
                all_backward_tau = (in_maksed_backward_tau.unsqueeze(0).unsqueeze(2) * dT[:, center + 1:].reshape(1, t, -1, 1, 1, 1)).float()
                ker_tran = torch.cat([all_forward_tau, all_backward_tau], dim=2).reshape(1, t, -1, size_patch_testing, size_patch_testing)

                # inputs for network
                in_patch = lq[..., h_idx:h_idx + size_patch_testing, w_idx:w_idx + size_patch_testing].cuda()
                in_coordinate = coordinate[..., h_idx:h_idx + size_patch_testing, w_idx:w_idx + size_patch_testing].cuda()
                in_ker_tran = ker_tran

                # trajectories from gyro
                in_ker_rot = compute_traj_from_rots(rot_mat.cuda(), in_coordinate, K_cu, K_inv_cu)

                with torch.cuda.amp.autocast():
                    out_patch = net_g(in_patch, in_ker_rot, in_ker_tran.cuda())

                out_patch = out_patch.detach().cpu().reshape(b, t - 2, c, size_patch_testing, size_patch_testing)
                print(h_idx, w_idx, out_patch.mean())

                out_patch_mask = torch.ones_like(out_patch)

                if True:
                    if h_idx < h_idx_list[-1]:
                        out_patch[..., -overlap_size // 2:, :] *= 0
                        out_patch_mask[..., -overlap_size // 2:, :] *= 0
                    if w_idx < w_idx_list[-1]:
                        out_patch[..., :, -overlap_size // 2:] *= 0
                        out_patch_mask[..., :, -overlap_size // 2:] *= 0
                    if h_idx > h_idx_list[0]:
                        out_patch[..., :overlap_size // 2, :] *= 0
                        out_patch_mask[..., :overlap_size // 2, :] *= 0
                    if w_idx > w_idx_list[0]:
                        out_patch[..., :, :overlap_size // 2] *= 0
                        out_patch_mask[..., :, :overlap_size // 2] *= 0

                E[..., h_idx:(h_idx + size_patch_testing), w_idx:(w_idx + size_patch_testing)].add_(out_patch)
                W[..., h_idx:(h_idx + size_patch_testing), w_idx:(w_idx + size_patch_testing)].add_(out_patch_mask)
        output = E.div_(W)
    output = output[:, :, :, :, :]

    return output
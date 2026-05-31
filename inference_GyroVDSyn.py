import glob
import os
import torch
import cv2
import tqdm
import numpy as np
import gc
import torch.nn.functional as F
import gtsam
import importlib
from copy import deepcopy
from gtsam import PreintegrationParams, PreintegratedImuMeasurements, Rot3
from time import perf_counter
from scipy.ndimage import distance_transform_edt
import argparse
from basicsr.archs.RAFT.raft_small import RAFT_small
from basicsr.archs.RAFT.utils.utils import InputPadder, image2torch
from basicsr.utils.img_util import tensor2img, img2tensor, imwrite
from inference_utils import *

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_size',
    type=int,
    choices=[48, 64, 96, 128],
    default=48,
    help='GyroDVD model size'
)
parser.add_argument(
    '--dataset_root',
    help="dataset root",
    default='dataset/GyroReal'
)
parser.add_argument(
    '--out_path',
    help="output path"
)
args = parser.parse_args()
viz_path = args.out_path 
source_dataset = args.dataset_root 


# Build GyroDVD model
module_name = "basicsr.archs.GyroDVD_arch"
model_name = f"GyroDVD_{args.model_size}"
net = getattr(
    importlib.import_module(module_name),
    model_name
)

# Create model and Load pretrained weights
model = net()
model.eval()
model.cuda()
weight_path = f"model_zoos/GyroDVD_{args.model_size}.pth"
load_net = torch.load(
    weight_path, map_location=lambda storage, loc: storage)
load_net = load_net['params']
model.load_state_dict(load_net, strict=True)


# Preintegration for gyro integration
imu_params = PreintegrationParams.MakeSharedU(9.81)
imu_params.setAccelerometerCovariance(np.identity(3))
imu_params.setGyroscopeCovariance(np.identity(3))
imu_params.setIntegrationCovariance(np.identity(3))
bias = gtsam.imuBias.ConstantBias()


# Image resolution and IMU-to-camera coordinate transform
H = 1920
W = 1080
R_imu_cam = Rot3(np.diag([1, -1, -1]))
R_cam_imu = R_imu_cam.inverse()

# acquired from Android API LENS_INTRINSIC_CALIBRATION
fx, fy = 1356.7, 1356.7
cx, cy = 541.6, 965.49

# instrinsic
K_np = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
]).astype('float32')
K = torch.from_numpy(K_np).unsqueeze(0)
K_inv = torch.inverse(K).unsqueeze(0)


# RAFT optical flow model
raft_small = RAFT_small()
raft_small.cuda()
raft_small.eval()
raft_small.max_batch = 32


dir_list = glob.glob(source_dataset + '/**/*')

with open('datalist/GyroVD_Syn_test.txt', 'rt') as f:
    test_video_list = f.readlines()
test_video_list = [line.strip().split('/')[-1] for line in test_video_list]

dir_list = [path for path in dir_list if path.split('/')[-1] in test_video_list]
dir_list = sorted(dir_list)
print(len(dir_list))
assert len(dir_list) == 77

for dir_path in tqdm.tqdm(dir_list):
    img_list = glob.glob(os.path.join(dir_path, 'blur/*.png'))
    img_list = sorted(img_list)

    # Load gyro and camera metadata
    gyro = np.loadtxt(os.path.join(dir_path, 'meta_info/gyro.csv'), delimiter=',')
    gyro[:, -1] = nano2sec(gyro[:, -1])

    # Load input images
    imgs = [cv2.imread(path)[:, :, ::-1].astype('float32')/255.0 for path in img_list]
    # The gyro data assumes portrait orientation, so we rotate the images.
    imgs = [cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) for img in imgs]
    imgs_torch = img2tensor(imgs, bgr2rgb=False, float32=True)
    imgs_torch = torch.stack(imgs_torch, dim=0).unsqueeze(0)


    # Compute center timestamps
    center_timestamps = []
    for path in img_list:
        timestamp = getCenterStamp(path)
        center_timestamps.append(nano2sec(float(timestamp)))


    #####################################  GYRO integration ##################################################

    # Runtime measurement
    t0 = perf_counter()
    n_sample = 8
    rot_mat_list = []
    dT_list = []
    for ith, img_path in enumerate(img_list):
        # Parse timestamps from filename
        split_name = os.path.basename(img_path).split('_')
        frame_num, start_stamp, end_stamp = int(split_name[0]), int(split_name[1]), int(split_name[3])
        exposure = int(split_name[3]) - int(split_name[2])

        start_stamp = nano2sec(start_stamp)
        end_stamp = nano2sec(end_stamp)
        exposure = nano2sec(exposure)

        # Sample timestamps during exposure of each frame
        timestamp_kers = np.linspace(start_stamp, end_stamp, n_sample + 1)
        dT_list.append(timestamp_kers - timestamp_kers[4:5])

        # Timestamps for rotation-induced flow
        if ith == 0:
            prev_stamp = center_timestamps[ith] - (center_timestamps[ith+1] - center_timestamps[ith])
            next_stamp = center_timestamps[ith+1]
        elif ith == (len(img_list)-1):
            prev_stamp = center_timestamps[ith-1]
            next_stamp = center_timestamps[ith] + (center_timestamps[ith] - center_timestamps[ith-1])
        else:
            prev_stamp = center_timestamps[ith-1]
            next_stamp = center_timestamps[ith+1]
        all_timestamp = np.concatenate([np.array([prev_stamp]), timestamp_kers, np.array([next_stamp])])

        # Forward gyro integration (center -> next frame)
        center = len(all_timestamp) // 2
        forward_rotations = []
        preint = PreintegratedImuMeasurements(imu_params, bias)
        for i in range(center, len(all_timestamp) - 1):
            # rotation is accumulated by preint
            Ra = integrate_only_gyro(preint, all_timestamp[i], all_timestamp[i + 1], gyro[:, -1], gyro[:, :3])
            Ra = R_imu_cam * Ra * R_cam_imu
            forward_rotations.append(Ra.matrix())


        # Backward gyro integration (center -> previous frame)
        backward_rotations = []
        for i in range(center - 1, -1, -1):
            # Backward rotations are computed from scratch, It's not accumulated.
            # This implementation can be improved by accumulating.
            preint = PreintegratedImuMeasurements(imu_params, bias)
            Rb = integrate_only_gyro(preint, all_timestamp[i], all_timestamp[center], gyro[:, -1], gyro[:, :3])
            Rb = R_imu_cam * Rb * R_cam_imu
            backward_rotations.append(Rb.inverse().matrix())

        # Compute rotated pixels from the center frame
        rot_list = list(reversed(backward_rotations)) + forward_rotations
        rot_np = np.stack(rot_list, 0)
        rot_mat_list.append(rot_np)


    # Per-frame rotation matrices
    rot_mat = np.stack(rot_mat_list, axis=0).astype('float32') # (t, 10, 3, 3)
    dt = (perf_counter() - t0) / len(img_list)
    print(f"Total time for gyro integration: {dt:.4f} sec.")


    # Split rotations for kernels and rotation-induced flow
    rot_mat_split = np.split(rot_mat, 10, axis=1)
    rot_mat_kers = np.concatenate(rot_mat_split[1:-1], axis=1)
    rot_mat_backward_flows = rot_mat_split[-1]
    rot_mat_forward_flows = rot_mat_split[0]

    ###################################### compute optical flows ############################################
    t0 = perf_counter()
    with torch.no_grad():
        b, t, c, h, w = imgs_torch.shape


        # Downsample images before RAFT inference.
        # RAFT can be less stable on high-resolution inputs,
        # so optical flow is estimated at half resolution and later rescaled.
        imgs_torch_resize = torch.clamp(F.interpolate(imgs_torch.view(b*t, c, h, w), scale_factor=0.5, mode='bicubic', align_corners=True), 0, 1).cuda()

        # Consecutive frame pairs
        img1_torch = imgs_torch_resize[:-1, ...]
        img2_torch = imgs_torch_resize[1:, ...]

        # Pad inputs for RAFT
        padder = InputPadder(img1_torch.shape)
        img1_torch, img2_torch = padder.pad(img1_torch, img2_torch)

        forward_flow_up = raft_small(img2_torch, img1_torch, iters=20, test_mode=True)
        flows_forwards_torch = padder.unpad(forward_flow_up)

        backward_flow_up = raft_small(img1_torch, img2_torch, iters=20, test_mode=True)
        flows_backwards_torch = padder.unpad(backward_flow_up)

        # Build reliable flow masks
        flows_backwards_cmap = fbConsistencyCheck(flows_forwards_torch, flows_backwards_torch)
        flows_forwards_cmap = fbConsistencyCheck(flows_backwards_torch, flows_forwards_torch)

        cycle_th = 8 / 2
        flows_backwards_mask = (flows_backwards_cmap < cycle_th).float()
        flows_forwards_mask = (flows_forwards_cmap < cycle_th).float()

        # Restore the original flow scale.
        # The flow is estimated at half resolution, so the magnitude is scaled by 2.
        # Spatial upsampling is performed later after tau computation to reduce
        # post-processing computation.
        flows_forwards_torch = flows_forwards_torch * 2
        flows_backwards_torch = flows_backwards_torch * 2

    dt = (perf_counter() - t0) / len(img_list)
    print(f"Total time for optical flows: {dt:.4f} sec.")


    # Clear temporary tensors and GPU memory
    flows_forwards_torch = flows_forwards_torch.detach().cpu()#.contiguous()
    flows_backwards_torch = flows_backwards_torch.detach().cpu()#.contiguous()
    flows_forwards_mask  = flows_forwards_mask.detach().cpu()#.contiguous()
    flows_backwards_mask = flows_backwards_mask.detach().cpu()#.contiguous()
    imgs_torch = imgs_torch.detach().cpu()

    del imgs_torch_resize, img1_torch, img2_torch
    del forward_flow_up, backward_flow_up
    del flows_forwards_cmap, flows_backwards_cmap

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    ###################################### compute tau ############################################

    coordinate = coords_grid(1, H, W, homogeneous=True)  # [B, 2, H, W]

    rot_mat_forward_flows_torch = torch.from_numpy(rot_mat_forward_flows)
    rot_mat_backward_flows_torch = torch.from_numpy(rot_mat_backward_flows)

    # Runtime measurement
    t0 = perf_counter()

    # Compute rotation-induced flow
    coordinate_resize = F.interpolate(coordinate, scale_factor=0.5, mode='bilinear', align_corners=True)
    forward_flows_gyro = compute_traj_from_rots(rot_mat_forward_flows_torch.unsqueeze(0), coordinate_resize.unsqueeze(0), K, K_inv)[0]
    backward_flows_gyro = compute_traj_from_rots(rot_mat_backward_flows_torch.unsqueeze(0), coordinate_resize.unsqueeze(0), K, K_inv)[0]

    center_timestamps_torch = torch.from_numpy(np.array(center_timestamps)) # (t)

    # backward_tau, from current frame to next frame
    backward_tau = (flows_backwards_torch - backward_flows_gyro[:-1]) / torch.abs(center_timestamps_torch[1:] - center_timestamps_torch[:-1]).view(-1, 1, 1, 1)

    # forward_tau, from current from to previous frame
    forward_tau = (flows_forwards_torch - forward_flows_gyro[1:]) / torch.abs(center_timestamps_torch[1:] - center_timestamps_torch[:-1]).view(-1, 1, 1, 1)

    # for handling the first and last frames
    backward_tau = torch.cat([backward_tau, forward_tau[-1:]*-1], dim=0)
    forward_tau = torch.cat([backward_tau[0:1]*-1, forward_tau], dim=0)
    tau = torch.cat([forward_tau, backward_tau], dim=1)

    # Combine flow-consistency masks
    mask = flows_backwards_mask[1:] * flows_forwards_mask[:-1]
    mask = torch.cat([flows_backwards_mask[0:1], mask, flows_forwards_mask[-1:]])

    dt = (perf_counter() - t0) / (t)
    print(f"Total time for tau: {dt:.4f} sec.")

    # Runtime measurement
    t0 = perf_counter()
    # Filterting tau using the flow-consistency masks
    results = []
    for ith, img_path in enumerate(img_list):
        masked_tau = nearest_fill_with_scipy(tau[ith:ith+1], mask[ith:ith+1])  # (2, 2, 1920, 1080)
        results.append(masked_tau)
    masked_tau = torch.cat(results, dim=0)  # (t, 4, H, W)
    masked_forward_tau, maksed_backward_tau = masked_tau[:, 0:2, :, :], masked_tau[:, 2:4, :, :]  # (b, n_seq, 1, 2, H, W)

    dt = (perf_counter() - t0) / len(img_list)
    print(f"Total time for masking: {dt:.4f} sec.")

    # Center exposure index
    center = 4

    dT = torch.from_numpy(np.stack(dT_list, axis=0))
    n_seq = dT.shape[0]

    # Rotational blur kernels
    rot_mat_kers_torch = torch.from_numpy(rot_mat_kers).unsqueeze(0)

    # Runtime measurement
    t0 = perf_counter()

    K_cu, K_inv_cu = K.cuda(), K_inv.cuda()
    coordinate = coordinate.unsqueeze(0) 
    n_total = n_seq

    # Sliding-window inference due to memory limiation
    if (n_total - 4) % 48 == 0:
        window = 52
        discard = 2
        overlap = discard * 2
        step = window - overlap

        results = []
        for start in range(0, n_total, step):
            end = min(start + window, n_total)

            in_lq = imgs_torch[:, start:end]  # (1, 48, 3, 1920, 1080)
            in_rot_mat = rot_mat_kers_torch[:, start:end]

            in_masked_forward_tau = masked_forward_tau[start:end]
            in_maksed_backward_tau = maksed_backward_tau[start:end]

            in_dT = dT[start:end]
            in_K_cu = K_cu
            in_K_inv_cu = K_inv_cu

            # Generate the final blur kernels in test_by_patch reducing the memory
            result = test_by_patch(model, in_lq, coordinate, in_rot_mat, in_masked_forward_tau, in_maksed_backward_tau, in_dT, in_K_cu, in_K_inv_cu)
            # Ignore boundary frames
            results.append(result[0, 1:-1])

        result = torch.cat(results, dim=0)

    dt = (perf_counter() - t0) / len(img_list)
    print(f"Total time for deblurring: {dt:.4f} sec.")

    # Ignore boundary frames as done in previous video deblurring methods
    path_list = img_list[2:-2]

    # Convert tensors to images
    result_list = torch.split(result, 1, 0)
    result_list = [tensor.squeeze(0) for tensor in result_list]
    sr_img_list = tensor2img(result_list, rgb2bgr=True)

    # Save restored frames
    for i, (sr_img, lq_path) in enumerate(zip(sr_img_list, path_list)):
        video_name = lq_path.split('/')[-3]
        img_name = os.path.basename(lq_path).replace('.jpg', '.png')
        save_img_path = os.path.join(
            viz_path, video_name,
            f'{img_name}')
        imwrite(sr_img, save_img_path)
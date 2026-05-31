from torch.utils import data as data
from torchvision.transforms.functional import normalize
import random


from basicsr.data.data_util import paired_videos_from_list
from basicsr.data.transforms import video_random_crop_with_grid
from basicsr.utils import img2tensor
import cv2
import glob
import numpy as np
import torch
import scipy.io
import os
import torch.nn.functional as F
import math
from basicsr.utils.registry import DATASET_REGISTRY

def random_noise(iso):
    """Generates random noise levels from a log-log linear distribution."""

    # noise profile from g channel
    iso2shot = lambda x: 7.2001e-07 * x + 1.2589e-05
    logshot2logread = lambda x: 1.4141 * x + -2.0269

    shot_noise = iso2shot(iso)
    log_shot_noise = math.log(shot_noise)
    log_read_noise = logshot2logread(log_shot_noise)
    read_noise = math.exp(log_read_noise)

    read_noise = read_noise + random.gauss(mu=0.0, sigma=2.2937725324349756e-07)  # randomness of ISO 100

    while read_noise <= 0 or shot_noise <= 0:
        shot_noise = iso2shot(iso)
        log_shot_noise = math.log(shot_noise)
        log_read_noise = logshot2logread(log_shot_noise)
        read_noise = math.exp(log_read_noise)

        read_noise = read_noise + random.gauss(mu=0.0, sigma=2.2937725324349756e-07)  # randomness of ISO 100


    return shot_noise, read_noise

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

@DATASET_REGISTRY.register()
class GyroVD_dataset(data.Dataset):
    def __init__(self, opt):
        super(GyroVD_dataset, self).__init__()
        self.opt = opt

        self.dataset_folder = opt['dataset_root']
        self.dataset_folder_tau = opt['dataset_root_tau']
        self.videos = paired_videos_from_list(self.dataset_folder , opt['datalist'], replicate=1)
        self.video2frame = self.getVideo2frame()

        self.n_seq = opt['n_sequence']

        # coordinate of images for warping pixels
        self.ori_grid = coords_grid(1, 1920, 1080,
                                      homogeneous=True)  # [B, 2, H, W]


    # for video frames loading
    def getVideo2frame(self):
        video2frame = {}
        video_list = glob.glob(os.path.join(self.dataset_folder, '**', '*'))
        video_list = sorted(video_list)

        for video_path in video_list:
            video_path_split = video_path.split(os.path.sep)

            video_name = video_path_split[-1]
            day_name = video_path_split[-2]
            frame_list = glob.glob(os.path.join(self.dataset_folder, day_name, video_name, 'blur', '*.png'))
            frame_list = sorted(frame_list)

            gtframe_list = [path.replace('blur'+os.path.sep, 'gt'+os.path.sep).replace('_blur.png', '_gt.png') for path in frame_list]
            video2frame["%s/%s" % (day_name, video_name)] = [frame_list, gtframe_list]

        return video2frame

    def __getitem__(self, index):

        video_name = self.videos[index]

        all_blur_frames_list, all_gt_frames_list = self.video2frame[video_name]
        video_length = len(all_blur_frames_list)

        # find proper center frame index containing n_seqs nearby frames
        start_frames = self.n_seq // 2
        end_frames = video_length - self.n_seq // 2 - (self.n_seq % 2)

        if self.opt['phase'] == 'train':
            center_index = random.randint(start_frames, end_frames)
            blur_frames_list = all_blur_frames_list[center_index - (self.n_seq // 2):center_index + (self.n_seq // 2) + (self.n_seq % 2)]
            gt_frames_list = all_gt_frames_list[center_index - (self.n_seq // 2):center_index + (self.n_seq // 2) + (self.n_seq % 2)]

        elif self.opt['phase'] == 'val':
            center_index = len(all_gt_frames_list)//2

            blur_frames_list = all_blur_frames_list[center_index - (self.n_seq // 2):center_index + (self.n_seq // 2) + (self.n_seq % 2)]
            gt_frames_list = all_gt_frames_list[center_index - (self.n_seq // 2):center_index + (self.n_seq // 2) + (self.n_seq % 2)]

        elif self.opt['phase'] == 'test':
            center_index = len(all_gt_frames_list)//2

            blur_frames_list = all_blur_frames_list
            gt_frames_list = all_gt_frames_list

        # Load gt and lq images.
        gt_videoclip_list = []
        for gt_path in gt_frames_list:
            img_gt = cv2.imread(gt_path).astype('float32') / 255.0
            img_gt = cv2.rotate(img_gt, cv2.ROTATE_90_CLOCKWISE)
            img_gt = img_gt[:, :, ::-1]  # BGR2RGB
            gt_videoclip_list.append(img_gt)

        lq_videoclip_list = []
        for lq_path in blur_frames_list:
            img_lq = cv2.imread(lq_path).astype('float32') / 255.0
            img_lq = cv2.rotate(img_lq, cv2.ROTATE_90_CLOCKWISE)
            img_lq = img_lq[:, :, ::-1]  # BGR2RGB
            lq_videoclip_list.append(img_lq)

        # Load Rot matrix computed from gyro
        n_sample = 8
        rot_videoclip_list = []
        for gt_path in gt_frames_list:
            rot_path = gt_path.replace('/gt/', '/rot/').replace('_gt.png', '_blur.npy')
            rot_np = np.load(rot_path)
            rot_videoclip_list.append(rot_np)

        # Load dK computed from gyro and flows
        if self.opt['load_tau']:
            tau_videoclip_list = []
            dTime_videoclip_list = []
            for gt_path in gt_frames_list:
                dK_path = gt_path.replace(self.dataset_folder, self.dataset_folder_tau).replace('/gt/', '/tau/').replace('_gt.png', '_blur_dK.npz')
                result = np.load(dK_path)
                tau_videoclip_list.append(result['dK'] * 50) # (2, 2, H, W)
                dTime_videoclip_list.append(result['dT'])
        else:
            tau_videoclip_list = []
            dTime_videoclip_list = []
            for gt_path in gt_frames_list:
                tau_videoclip_list.append(np.zeros([2, 2, img_lq.shape[0], img_lq.shape[1]]).astype('float32')) # (2, 2, H, W)
                dTime_videoclip_list.append(np.ones(9).astype('float32'))


        if self.opt['RSBlur']:
            # load saturation mask
            sat_mask_videoclip_list = []
            for lq_path in blur_frames_list:
                sat_mask_path = lq_path.replace('_blur.png', '_satmask.png').replace('blur' + os.path.sep, 'sat_mask' + os.path.sep)
                sat_mask = cv2.imread(sat_mask_path).astype('float32') / 255.0
                sat_mask = cv2.rotate(sat_mask, cv2.ROTATE_90_CLOCKWISE)
                sat_mask = sat_mask[:, :, ::-1]  # BGR2RGB
                sat_mask_videoclip_list.append(sat_mask)

            # generate noise, wb, paparmeters
            video_name = lq_path.split(os.path.sep)[-3]
            day_name = lq_path.split(os.path.sep)[-4]

            iso = random.uniform(50, 1600)
            beta1, beta2 = random_noise(iso)
            alpha_saturation = random.uniform(0.25, 2.75)
            alpha_saturation = torch.tensor(alpha_saturation)

            CamInfo_path = os.path.join(self.dataset_folder, day_name, video_name, 'meta_info', 'cam_metainfo.csv')
            camInfo = np.loadtxt(CamInfo_path, delimiter=',')

            redGain = float(camInfo[2])
            blueGain = float(camInfo[5])

            red_gain = torch.tensor([redGain]).float()
            blue_gain = torch.tensor([blueGain]).float()

            beta1 = torch.tensor(beta1).float()
            beta2 = torch.tensor(beta2).float()

            ccm_RAW2RGB = camInfo[6:].astype('float32').reshape(3, 3)
            ccm_RGB2RAW = np.linalg.inv(ccm_RAW2RGB)

            ccm_RAW2RGB = torch.tensor(ccm_RAW2RGB)
            ccm_RGB2RAW = torch.tensor(ccm_RGB2RAW)

        else:
            sat_mask_videoclip_list = [np.zeros_like(img_lq) for i in range(len(lq_videoclip_list))]

            alpha_saturation = torch.tensor(0)
            red_gain = torch.tensor([0]).float()
            blue_gain = torch.tensor([0]).float()
            beta1 = torch.tensor(0).float()
            beta2 = torch.tensor(0).float()

            ccm_RAW2RGB = torch.zeros(3,3).float()
            ccm_RGB2RAW = torch.zeros(3, 3).float()

        ori_grid = self.ori_grid

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size'] + self.opt['boundary_size']
            scale = 1

            tau_list = [tau.transpose(0, 2, 3, 1) for tau in tau_videoclip_list]
            grid_and_tau = [ori_grid.permute(0, 2, 3, 1)] + tau_list

            gt_videoclip_list, lq_videoclip_list, sat_mask_videoclip_list, grid_and_tau = video_random_crop_with_grid(
                gt_videoclip_list, \
                lq_videoclip_list, \
                sat_mask_videoclip_list, \
                grid_and_tau,
                gt_size, scale,
                gt_path, center_crop=False)

            grid = grid_and_tau.pop(0)
            ori_grid = grid.permute(0, 3, 1, 2)

            tau_videoclip_list = [tau.transpose(0, 3, 1, 2) for tau in grid_and_tau]

        elif self.opt['phase'] == 'val':
            gt_size = 1024
            scale = 1
            # random crop
            tau_list = [tau.transpose(0, 2, 3, 1) for tau in tau_videoclip_list]
            grid_and_tau = [ori_grid.permute(0, 2, 3, 1)] + tau_list

            gt_videoclip_list, lq_videoclip_list, sat_mask_videoclip_list, grid_and_tau = video_random_crop_with_grid(
                    gt_videoclip_list, \
                    lq_videoclip_list, \
                    sat_mask_videoclip_list, \
                    grid_and_tau,
                    gt_size, scale,
                    gt_path, center_crop=True)

            grid = grid_and_tau.pop(0)
            ori_grid = grid.permute(0, 3, 1, 2)

            tau_videoclip_list = [tau.transpose(0, 3, 1, 2) for tau in grid_and_tau]

        hflip, vflip, rot90 = False, False, False
        # augmentation for training
        if self.opt['phase'] == 'train':
            
            # Blur kernels and images are augmented together later in GyroDVD_model.py
            hflip = self.opt['use_flip'] and random.random() < 0.5
            if self.opt['use_rot']:
                vflip = random.random() < 0.5
            rot90 = self.opt['use_rot'] and random.random() < 0.5


        gt_videoclip_list = [img_np.copy() for img_np in gt_videoclip_list]
        lq_videoclip_list = [img_np.copy() for img_np in lq_videoclip_list]

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        gt_videoclip_list_pt = img2tensor(gt_videoclip_list,
                                          bgr2rgb=False,
                                          float32=True)

        lq_videoclip_list_pt = img2tensor(lq_videoclip_list,
                                          bgr2rgb=False,
                                          float32=True)

        # final video clip
        gt_videoclip = torch.stack(gt_videoclip_list_pt)
        lq_videoclip = torch.stack(lq_videoclip_list_pt)

        img_sat_np = np.stack([img_sat_mask.copy().transpose(2, 0, 1) for img_sat_mask in sat_mask_videoclip_list], 0).astype('float32')
        img_sat_mask = torch.from_numpy(img_sat_np)

        rot_mat_np = np.stack(rot_videoclip_list, 0).astype('float32') # (N_seq, 8, 3, 3)
        rot_mat = torch.from_numpy(rot_mat_np)

        tau_np = np.stack(tau_videoclip_list, 0).astype('float32') # (N_seq, 8, 3, 3)
        tau = torch.from_numpy(tau_np)

        dTime_np = np.stack(dTime_videoclip_list, 0).astype('float32')  # (N_seq, 8, 3, 3)
        dTime = torch.from_numpy(dTime_np)

        return {
            'lq': lq_videoclip,
            'gt': gt_videoclip,
            'rot_mat': rot_mat,
            'tau': tau,
            'dTime': dTime,
            'lq_path': blur_frames_list,
            'gt_path': gt_frames_list,
            'center_index': center_index,

            # training augmentation parameters
            'hflip': hflip,
            'vflip': vflip,
            'rot90': rot90,
            'coordinate': ori_grid,

            # RSBlur parameters
            'img_sat_mask': img_sat_mask,
            'red_gain': red_gain,
            'blue_gain': blue_gain,
            'beta1': beta1,
            'beta2': beta2,
            'alpha_saturation': alpha_saturation,
            'ccm_RAW2RGB': ccm_RAW2RGB,
            'ccm_RGB2RAW': ccm_RGB2RAW
        }


    def __len__(self):
        return len(self.videos)


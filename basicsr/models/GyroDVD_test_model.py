# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
from torch import distributed as dist
from collections import OrderedDict
from tqdm import tqdm
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.nn import functional as F
from basicsr.models.base_model import BaseModel
from basicsr.archs import build_network
from basicsr.utils import get_root_logger, tensor2img, imwrite
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.logger import AverageMeter
from copy import deepcopy


from basicsr.data.RSBlur_util_with_cuda import *
from debayer import Debayer5x5, Layout
import scipy.io
import os

def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

@MODEL_REGISTRY.register()
class GyroDVD_test_model(BaseModel):
    """Base Deblur model for single image deblur."""
    def __init__(self, opt):
        super(GyroDVD_test_model, self).__init__(opt)

        self.net_g = build_network(opt["network_g"])
        self.net_g = self.model10_to_device(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))


        # intrinsic for computing trajectories from gyro
        fx, fy = 1356.7, 1356.7
        cx, cy = 541.6, 965.49

        K_np = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ]).astype('float32')
        self.K = torch.from_numpy(K_np).cuda()
        self.K_inv = torch.inverse(self.K).cuda()


    def model10_to_device(self, net):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """

        net = net.to(self.device)
        if self.opt['dist']:
            net = DistributedDataParallel(
                net,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=False
                )
            net._set_static_graph()
            
        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        return net

    def compute_traj_from_rots(self, R, coordinate, K, K_inv):
        # coordinate : cropped coordinates (b, 1, 3, h, w)
        # Rot : rotation matrices computed from gyo (b, N, 8, 3, 3)
        b, _, _, h, w = coordinate.shape
        b, n_seq, _, _, _ = R.shape

        pix = coordinate.view(b, 3, h*w)

        # 2. Compute normalized rays in camera coordinates
        rays = K_inv.unsqueeze(0) @ pix  # (1, 3, 3)x(b, 3, hxw) => (b, 3, H*W)

        #print(R.shape, R.permute(0, 1, 2, 4, 3).shape)
        R = R.permute(0, 1, 2, 4, 3) # transform R_wc to R_cw
        rot_rays = R @ rays.view(b, 1, 1, 3, h*w) # (b, N, 8, 3, 3)x(b, 1, 1, 3, hxw) => (b, N, 8, 3, hxw)
        rot_rays = rot_rays / rot_rays[:,:,:,2:3, :]
        proj = K.view(1, 1, 1, 3, 3) @ rot_rays # (1, 1, 1, 3, 3)x(b, N, 8, 3, hxw) => (b, N, 8, 3, hxw)
        proj_xy = proj[:,:,:,:2,:] # (b, N, 8, 2, hxw)

        trajectories = proj_xy - coordinate[:,:,:2,:,:].view(1, 1, 1, 2, -1)
        trajectories = trajectories.reshape(b, n_seq, -1, h, w) # (b, n, 8x2, h, w)

        return trajectories

    def feed_data_test(self,data):
        # To save GPU memory, inputs and kernels are kept on CPU and moved to GPU
        # only right before network inference.

        lq, gt = data['lq'],data['gt']
        self.lq = lq#.to(self.device)
        self.gt = gt#.to(self.device)

        # rotation matrixes computed from gyro
        self.rot_mat = data['rot_mat']#.to(self.device) # (b, n, 8, 3, 3)
        self.coordinate = data['coordinate']#.to(self.device) # (b, 1, 2, H, W)

        self.tau = data['tau']#.to(self.device) # (b, n_seq, 2, 2, H, W)
        self.dTime = data['dTime']#.to(self.device)  # (b, n_seq, 9)

        split_rot_mat = torch.split(self.rot_mat, 25, 1)
        results = []
        for rot_mat in split_rot_mat:
            kernels_rot = self.compute_traj_from_rots(rot_mat, self.coordinate, self.K.cpu(), self.K_inv.cpu())#.cpu() # (b, n, 16, h, w)
            results.append(kernels_rot)
        self.ker_rot = torch.cat(results, dim=1)

        b, n_seq, c, h, w = self.ker_rot.size()

        forward_tran, backward_tran = self.tau[:, :, 0:1, :, :, :], self.tau[:, :, 1:,:,:,:] # (b, n_seq, 1, 2, H, W)
        center = self.dTime.shape[-1]//2
        forward_tran = forward_tran * torch.abs(self.dTime[:, :, :center]).reshape(b, n_seq, -1, 1, 1, 1)
        backward_tran = backward_tran * self.dTime[:, :, center+1:].reshape(b, n_seq, -1, 1, 1, 1)

        self.ker_tran = torch.cat([forward_tran, backward_tran], dim=2).reshape(b, n_seq, -1, h, w)

    def test_by_patch(self, lq, ker_rot, ker_tran):

        self.net_g.eval()
        with torch.no_grad():
            size_patch_testing = 512
            overlap_size = 128
            b, t, c, h, w = lq.shape
            stride = size_patch_testing - overlap_size
            h_idx_list = list(range(0, h - size_patch_testing, stride)) + [max(0, h - size_patch_testing)]
            w_idx_list = list(range(0, w - size_patch_testing, stride)) + [max(0, w - size_patch_testing)]

            E = torch.zeros(b, t-2, c, h, w)
            W = torch.zeros_like(E)
            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    with torch.cuda.amp.autocast():
                        in_patch = lq[..., h_idx:h_idx + size_patch_testing, w_idx:w_idx + size_patch_testing]
                        in_ker_rot = ker_rot[..., h_idx:h_idx + size_patch_testing, w_idx:w_idx + size_patch_testing]
                        in_ker_tran = ker_tran[..., h_idx:h_idx + size_patch_testing, w_idx:w_idx + size_patch_testing]
                        out_patch = self.net_g(in_patch.to(self.device), in_ker_rot.to(self.device), in_ker_tran.to(self.device))

                    out_patch = out_patch.detach().cpu().reshape(b, t-2, c, size_patch_testing, size_patch_testing)

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
        self.output = output[:, :, :, :, :]

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                self.output = self.net_g(self.lq, self.ker_rot, self.ker_tran)


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image=True):
        logger = get_root_logger()
        # logger.info('Only support single GPU validation.')
        import os
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image=True):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        torch.cuda.empty_cache()
        metric_data = dict()
        for idx, val_data in enumerate(dataloader):
            torch.cuda.empty_cache()
            self.feed_data_test(val_data)
            self.test_by_patch(self.lq[:, :52], self.ker_rot[:, :52], self.ker_tran[:, :52])  # 0:52
            visuals = self.get_current_visuals()

            result1 = visuals['result'][0, 1:-1]  # 2:50
            torch.cuda.empty_cache()

            self.test_by_patch(self.lq[:, 48:], self.ker_rot[:, 48:], self.ker_tran[:, 48:])  # 48:100
            visuals = self.get_current_visuals()
            result2 = visuals['result'][0, 1:-1]  # 50:98

            torch.cuda.empty_cache()

            result = torch.cat([result1, result2], dim=0)
            gt = visuals['gt'][0, 2:-2]

            result_list = torch.split(result, 1, 0)
            gt_list = torch.split(gt, 1, 0)

            result_list = [tensor.squeeze(0) for tensor in result_list]
            gt_list = [tensor.squeeze(0) for tensor in gt_list]

            sr_img_list = tensor2img(result_list, rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img_list = tensor2img(gt_list, rgb2bgr=rgb2bgr)

            if self.opt['val']['save_img']:
                for i, (sr_img, gt_img) in enumerate(zip(sr_img_list, gt_img_list)):
                    lq_path = val_data['lq_path'][i + 2][0]
                    video_name = lq_path.split('/')[-3]
                    img_name = os.path.basename(lq_path)
                    save_img_path = os.path.join(
                        self.opt['path']['visualization'], dataset_name, video_name,
                        f'{img_name}')
                    imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:

                    for sr_img, gt_img in zip(sr_img_list, gt_img_list):

                        metric_data['img'] = sr_img
                        metric_data['img2'] = gt_img
                        for metric_idx, (name, opt_) in enumerate(self.opt['val']['metrics'].items()):
                            result = calculate_metric(metric_data, opt_)
                            self.metric_results[name] += result

                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            print(self.metric_results)
            pbar.update(1)
            # pbar.set_description(f'Test {img_name}')
            cnt += len(sr_img_list)

        pbar.close()

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)

        torch.cuda.empty_cache()

        return current_metric


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()  # [0,2:-2,:,:,:]
        out_dict['result'] = self.output.detach().cpu()  # [0,2:-2,:,:,:]
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()  # [0,2:-2,:,:,:]
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

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
class GyroDVD_model(BaseModel):
    """Base Deblur model for single image deblur."""
    def __init__(self, opt):
        super(GyroDVD_model, self).__init__(opt)

        self.net_g = build_network(opt["network_g"])
        self.net_g = self.model10_to_device(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()
        self.scaler = torch.cuda.amp.GradScaler()

        self.demosaic = [Debayer5x5(layout=Layout.RGGB).cuda(),\
                         Debayer5x5(layout=Layout.BGGR).cuda(),\
                         Debayer5x5(layout=Layout.GRBG).cuda(),\
                         Debayer5x5(layout=Layout.GBRG).cuda()]

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

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.log_dict = OrderedDict()
        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
            self.log_dict['l_pix'] = AverageMeter()
        else:
            self.cri_pix = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')
        
        self.setup_optimizers()
        self.setup_schedulers()

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

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                
                logger.warning(f'Params {k} will not be optimized.')
        logger = get_root_logger()

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                **train_opt['optim_g'])

        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)


    def compute_traj_from_rots(self, R, coordinate):
        # coordinate : cropped coordinates (b, 1, 3, h, w)
        # Rot : rotation matrices computed from gyo (b, N, 8, 3, 3)
        b, _, _, h, w = coordinate.shape
        b, n_seq, _, _, _ = R.shape

        pix = coordinate.view(b, 3, h*w)

        # 2. Compute normalized rays in camera coordinates
        rays = self.K_inv.unsqueeze(0) @ pix  # (1, 3, 3)x(b, 3, hxw) => (b, 3, H*W)

        #print(R.shape, R.permute(0, 1, 2, 4, 3).shape)
        R = R.permute(0, 1, 2, 4, 3) # transform R_wc to R_cw
        rot_rays = R @ rays.view(b, 1, 1, 3, h*w) # (b, N, 8, 3, 3)x(b, 1, 1, 3, hxw) => (b, N, 8, 3, hxw)
        rot_rays = rot_rays / rot_rays[:,:,:,2:3, :]
        proj = self.K.view(1, 1, 1, 3, 3) @ rot_rays # (1, 1, 1, 3, 3)x(b, N, 8, 3, hxw) => (b, N, 8, 3, hxw)
        proj_xy = proj[:,:,:,:2,:] # (b, N, 8, 2, hxw)

        trajectories = proj_xy - coordinate[:,:,:2,:,:].view(1, 1, 1, 2, -1)
        trajectories = trajectories.reshape(b, n_seq, -1, h, w) # (b, n, 8x2, h, w)

        return trajectories

    def _augment(self, img, hflip, vflip, rot90):
        # img : (n_seq, 3, h, w)
        if hflip:  # horizontal
            img = torch.flip(img, dims=[3])  # flip width

        if vflip:  # vertical
            img = torch.flip(img, dims=[2])  # flip height

        if rot90:
            img = img.clone().permute(0, 1, 3, 2)

        return img

    def _augment_kers(self, kernel, hflip, vflip, rot90):
        n_seq, c, h, w = kernel.shape
        kernel = kernel.view(n_seq, -1, 2, h, w)  # (n_seq, 8, 2, h, w)

        if hflip:  # horizontal
            kernel = torch.flip(kernel, dims=[4])  # flip width
            kernel[:, :, 0, :, :] *= -1

        if vflip:  # vertical
            kernel = torch.flip(kernel, dims=[3])  # flip height
            kernel[:, :, 1, :, :] *= -1

        if rot90: # -rot90
            kernel = kernel.clone().permute(0, 1, 2, 4, 3)
            kernel = kernel[:, :, [1, 0], :, :]

        kernel = kernel.view(n_seq, c, h, w)

        return kernel

    def feed_data_test(self,data):
        lq, gt = data['lq'],data['gt']
        self.lq = lq.to(self.device)
        self.gt = gt.to(self.device)

        # rotation matrixes computed from gyro
        self.rot_mat = data['rot_mat'].to(self.device) # (b, n, 8, 3, 3)
        self.coordinate = data['coordinate'].to(self.device) # (b, 1, 2, H, W)

        self.tau = data['tau'].to(self.device) # (b, n_seq, 2, 2, H, W)
        self.dTime = data['dTime'].to(self.device)  # (b, n_seq, 9)

        self.ker_rot = self.compute_traj_from_rots(self.rot_mat, self.coordinate) # (b, n, 16, h, w)

        b, n_seq, c, h, w = self.ker_rot.size()

        forward_tran, backward_tran = self.tau[:, :, 0:1, :, :, :], self.tau[:, :, 1:,:,:,:] # (b, n_seq, 1, 2, H, W)
        center = self.dTime.shape[-1]//2
        forward_tran = forward_tran * torch.abs(self.dTime[:, :, :center]).reshape(b, n_seq, -1, 1, 1, 1)
        backward_tran = backward_tran * self.dTime[:, :, center+1:].reshape(b, n_seq, -1, 1, 1, 1)

        self.ker_tran = torch.cat([forward_tran, backward_tran], dim=2).reshape(b, n_seq, -1, h, w)


    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)

        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        # rotation matrixes computed from gyro
        self.rot_mat = data['rot_mat'].to(self.device) # (b, n, 8, 3, 3)
        self.coordinate = data['coordinate'].to(self.device) # (b, 1, 2, H, W)

        self.tau = data['tau'].to(self.device) # (b, n_seq, 2, 2, H, W)
        self.dTime = data['dTime'].to(self.device)  # (b, n_seq, 9)

        self.ker_rot = self.compute_traj_from_rots(self.rot_mat, self.coordinate) # (b, n, 16, h, w)

        b, n_seq, c, h, w = self.ker_rot.size()

        forward_tran, backward_tran = self.tau[:, :, 0:1, :, :, :], self.tau[:, :, 1:,:,:,:] # (b, n_seq, 1, 2, H, W)
        center = self.dTime.shape[-1]//2
        forward_tran = forward_tran * torch.abs(self.dTime[:, :, :center]).reshape(b, n_seq, -1, 1, 1, 1)
        backward_tran = backward_tran * self.dTime[:, :, center+1:].reshape(b, n_seq, -1, 1, 1, 1)

        self.ker_tran = torch.cat([forward_tran, backward_tran], dim=2).reshape(b, n_seq, -1, h, w)

        # RSBlur pipeline
        if 'img_sat_mask' in data:
            self.img_sat_mask = data['img_sat_mask'].to(self.device)

        # for augmentation
        self.hflip = data['hflip']
        self.vflip = data['vflip']
        self.rot90 = data['rot90']


        # Apply random hflip, vflip, and rot90 augmentations to both images and
        # the corresponding blur kernels
        b = self.hflip.shape[0]
        for i in range(b):
            self.lq[i, :, :, :, :] = self._augment(self.lq[i, :, :, :, :], self.hflip[i], self.vflip[i], self.rot90[i])
            self.img_sat_mask[i, :, :, :, :] = self._augment(self.img_sat_mask[i, :, :, :, :], self.hflip[i],
                                                             self.vflip[i], self.rot90[i])

            self.gt[i, :, :, :, :] = self._augment(self.gt[i, :, :, :, :], self.hflip[i], self.vflip[i], self.rot90[i])
            self.ker_rot[i, :, :, :, :] = self._augment_kers(self.ker_rot[i, :, :, :, :], self.hflip[i], self.vflip[i], self.rot90[i])
            self.ker_tran[i, :, :, :, :] = self._augment_kers(self.ker_tran[i, :, :, :, :], self.hflip[i], self.vflip[i], self.rot90[i])


        if 'alpha_saturation' in data:
            self.alpha_saturation = data['alpha_saturation'].to(self.device).float()

        if 'red_gain' in data:
            self.red_gain = data['red_gain'].to(self.device)

        if 'blue_gain' in data:
            self.blue_gain = data['blue_gain'].to(self.device)

        if 'beta1' in data:
            self.beta1 = data['beta1'].to(self.device)

        if 'beta2' in data:
            self.beta2 = data['beta2'].to(self.device)

        if 'ccm_RAW2RGB' in data:
            self.ccm_RAW2RGB = data['ccm_RAW2RGB'].to(self.device)

        if 'ccm_RGB2RAW' in data:
            self.ccm_RGB2RAW = data['ccm_RGB2RAW'].to(self.device)

        # Due to artifacts of demosaic on edges, we use bigger images and randomly crop.
        boundary_size = self.opt["datasets"]["train"]["boundary_size"]
        gt_size = self.opt["datasets"]["train"]["gt_size"]
        # randomly crop according to the boundary size
        start_h = random.randrange(0, boundary_size // 2)
        start_w = random.randrange(0, boundary_size // 2)

        # on-the-fly RSBlur pipeline on wide images
        if self.opt['datasets']['train']['RSBlur']:
            with torch.no_grad():
                bayer_pattern = random.choice(['RGGB', 'BGGR', 'GRBG', 'GBRG'])
                rsblur_lq = self.RSBlurPipeline(self.lq[0], self.img_sat_mask[0], self.red_gain, self.blue_gain, self.beta1, self.beta2,
                                                   self.alpha_saturation, bayer_pattern, self.ccm_RAW2RGB, self.ccm_RGB2RAW)
            lq = rsblur_lq[:,:,start_h:start_h+gt_size, start_w:start_w+gt_size]

        else:
            lq = self.lq[0,:,:,start_h:start_h+gt_size, start_w:start_w+gt_size]

        self.ori_lq = self.lq[:,:,:,start_h:start_h+gt_size, start_w:start_w+gt_size]
        self.lq = lq[None,:,:,:,:]
        self.gt = self.gt[:,:,:,start_h:start_h+gt_size, start_w:start_w+gt_size]
        self.ker_rot = self.ker_rot[:,:,:,start_h:start_h+gt_size, start_w:start_w+gt_size]
        self.ker_tran = self.ker_tran[:,:,:,start_h:start_h+gt_size, start_w:start_w+gt_size]


    def RSBlurPipeline(self, blurred_pt, sat_mask_pt, red_gain, blue_gain, beta1, beta2, alpha_saturation, bayer_pattern_W, ccm_RAW2RGB, ccm_RGB2RAW):

        blurred_pt = blurred_pt.permute(0, 2, 3, 1)
        sat_mask_pt = sat_mask_pt.permute(0, 2, 3, 1)

        batch_size, _, _, _ = blurred_pt.shape

        # inverse tone mapping
        blurred_L = rgb2lin_pt(blurred_pt)

        # saturation synthesis
        blurred_L = blurred_L + (alpha_saturation.view(1, 1, 1, 1) * sat_mask_pt)
        blurred_L = torch.clamp(blurred_L, 0, 1)

        blurred_sat = blurred_L.clone()

        # from linear RGB to XYZ
        img_Cam = apply_cmatrix(blurred_L, ccm_RGB2RAW)

        # Mosaic
        img_mosaic = mosaic_bayer(img_Cam, bayer_pattern_W)

        # inverse white balance
        img_mosaic = WB_img(img_mosaic, bayer_pattern_W, 1 / red_gain, 1 / blue_gain)

        # -------- ADDING POISSON-GAUSSIAN NOISE ON RAW -
        img_mosaic_noise = add_Poisson_noise_random(img_mosaic, beta1, beta2)

        # -------- ISP PROCESS --------------------------
        # White balance
        img_demosaic = WB_img(img_mosaic_noise, bayer_pattern_W, red_gain, blue_gain)

        # demosaic
        img_demosaic = torch.nn.functional.pixel_shuffle(img_demosaic.permute(0, 3, 1, 2), 2)
        if bayer_pattern_W == 'RGGB':
            img_demosaic = self.demosaic[0](img_demosaic).permute(0, 2, 3, 1)
        elif bayer_pattern_W == 'BGGR':
            img_demosaic = self.demosaic[1](img_demosaic).permute(0, 2, 3, 1)
        elif bayer_pattern_W == 'GRBG':
            img_demosaic = self.demosaic[2](img_demosaic).permute(0, 2, 3, 1)
        elif bayer_pattern_W == 'GBRG':
            img_demosaic = self.demosaic[3](img_demosaic).permute(0, 2, 3, 1)

        # from Cam to linear RGB
        img_IL = apply_cmatrix(img_demosaic, ccm_RAW2RGB)

        # tone mapping
        img_IL = torch.clamp(img_IL, 0, 1)
        img_Irgb = lin2rgb_pt(img_IL)

        blurred = img_Irgb

        # don't add noise on saturated region
        sat_region = torch.ge(blurred_sat, 1.0)
        non_sat_region = torch.logical_not(sat_region)
        blurred = (blurred_sat * sat_region) + (blurred * non_sat_region)

        blurred = blurred.permute(0, 3, 1, 2)

        return blurred


    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        self.lq = self.lq.half()
        self.ker_rot = self.ker_rot.half()
        self.ker_tran = self.ker_tran.half()
        with torch.cuda.amp.autocast():
            output  = self.net_g(self.lq, self.ker_rot, self.ker_tran)
            
            self.output = output
            loss_dict = OrderedDict()
            l_pix = self.cri_pix(output, self.gt[:,1:-1,:,:])
            loss_dict['l_pix'] = l_pix

            l_total = l_pix  + 0 * sum(p.sum() for p in self.net_g.parameters())

        # l_total.backward()
        self.scaler.scale(l_total).backward()
        self.scaler.unscale_(self.optimizer_g)
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.scaler.step(self.optimizer_g)
        self.scaler.update()
        
        for k,v in self.reduce_loss_dict(loss_dict).items():
            self.log_dict[k].update(v)
        
        # exit(0)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                self.output = self.net_g(self.lq, self.ker_rot, self.ker_tran)
        self.net_g.train()

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

            self.feed_data_test(val_data)
            self.test()

            visuals = self.get_current_visuals()

            result_list = torch.split(visuals['result'], 1, 0)
            gt_list = torch.split(visuals['gt'], 1, 0)

            result_list = [tensor.squeeze(0) for tensor in result_list]
            gt_list = [tensor.squeeze(0) for tensor in gt_list]

            sr_img_list = tensor2img(result_list, rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img_list = tensor2img(gt_list, rgb2bgr=rgb2bgr)


            torch.cuda.empty_cache()

            if self.opt['val']['save_img']:
                for i, (sr_img, gt_img) in enumerate(zip(sr_img_list, gt_img_list)):
                    lq_path = val_data['lq_path'][i + 1][0]
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

                    for i, (sr_img, gt_img) in enumerate(zip(sr_img_list, gt_img_list)):
                        if save_img:

                            lq_path = val_data['lq_path'][i+1]
                            video_name = lq_path.split('/')[-3]
                            img_name = os.path.basename(lq_path)
                            save_img_path = osp.join(
                                            self.opt['path']['visualization'], dataset_name, video_name,
                                            f'{img_name}.png')
                            print(save_img_path)
                            imwrite(sr_img, save_img_path)

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

            pbar.update(1)
            #pbar.set_description(f'Test {img_name}')
            cnt += len(sr_img_list)
            # if cnt == 300:
            #     break

            # if cnt >= 20:
            #     break

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

        out_dict['lq'] = self.lq.detach().cpu()[0,2:-2,:,:,:]
        out_dict['result'] = self.output.detach().cpu()[0,1:-1,:,:,:]
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()[0,2:-2,:,:,:]

        if hasattr(self, 'ori_lq'):
            out_dict['ori_lq'] = self.ori_lq.detach().cpu()[0,2:-2,:,:,:]

        if hasattr(self, 'ker_rot'):
            out_dict['ker_rot'] = self.ker_rot.detach().cpu()[0,2:-2,:,:,:]

        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

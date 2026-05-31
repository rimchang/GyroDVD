import os
import datetime
import logging
import math
import time
import torch
import shutil
import cv2
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options
from basicsr.models import build_model

import torch.multiprocessing as mp
mp.set_start_method('fork')

def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        new_name = new_name.replace('tb_logger', 'tb_logger_archived')
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        shutil.move(path, new_name)
    os.makedirs(path, exist_ok=True)

def viz_accum_grid(img, accum_grid):

    image = img.copy()
    B, C, H, W = accum_grid.size()

    xx = torch.arange(0, W, 60).view(1, -1)
    yy = torch.arange(0, H, 60).view(-1, 1)
    xx = xx.repeat(yy.shape[0], 1)
    yy = yy.repeat(1, xx.shape[1])

    for index in range(accum_grid.shape[0] - 1):
        index_grid = accum_grid[index, :, yy, xx]

        index_xx = index_grid[0, :, :].numpy().astype('int32') + xx.numpy()
        index_yy = index_grid[1, :, :].numpy().astype('int32') + yy.numpy()

        next_grid = accum_grid[index + 1, :, yy, xx]

        next_xx = next_grid[0, :, :].numpy().astype('int32') + xx.numpy()
        next_yy = next_grid[1, :, :].numpy().astype('int32') + yy.numpy()

        # import pdb; pdb.set_trace()
        for (x, y, x_next, y_next) in zip(index_xx.flatten(), index_yy.flatten(), next_xx.flatten(), next_yy.flatten()):
            cv2.line(image, (int(x), int(y)), (int(x_next), int(y_next)), (0, int(255*index/accum_grid.shape[0]), 0), thickness=2)

    return image


def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    wandb_logger = None
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        wandb_logger = init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join('logs','tb_logger', opt['name']))
    return tb_logger,wandb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info('Training statistics:'
                        f'\n\tNumber of train seq: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
    return resume_state

def train_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    
    # load resume states if necessary
    resume_state = load_resume_state(opt)

    print(opt['path'], "opt['path']")

    # mkdir for experiments and logger
    # print(opt["path"]["resume_state"])
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            os.makedirs(osp.join(opt['root_path'], 'tb_logger_archived'), exist_ok=True)
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    # initialize wandb and tb loggers
    tb_logger,wandb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    # create model
    model = build_model(opt)
    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']

    else:
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger,wandb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:   
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.' "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()
    new_iter = 0
    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()

            current_iter += 1
            new_iter += 1
            if current_iter > total_iters:
                break
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            # update learning rate
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            iter_timer.record()
            if current_iter == 1:
                # reset start time in msg_logger for more accurate eta_time
                # not work in resume mode
                msg_logger.reset_start_time()

            # log
            # if True:
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)
                """ out_dict = model.get_train_visuals()
                show_tensor = []
                n,t,c,h,w = out_dict['lq'].shape
                out_dict['lq'] = F.interpolate(out_dict['lq'].view(-1, 3, h, w), scale_factor=0.25, mode='bicubic')\
                .view(n, t, 3, h // 4, w // 4)
                out_dict['results'] = F.interpolate(out_dict['results'].view(-1, 3, h, w), scale_factor=0.25, mode='bicubic')\
                .view(n, t, 3, h // 4, w // 4)
                out_dict['masks'] = F.interpolate(out_dict['masks'].view(-1, 1, h//64, w//64), scale_factor=16, mode='nearest')\
                .view(n, 1, 1, h // 4, w // 4)
                out_dict['show_motion_st_win'] = F.interpolate(out_dict['show_motion_st_win'].view(-1, 1, h//64, w//64), scale_factor=16, mode='nearest')\
                .view(n, t, 1, h // 4, w // 4)
                out_dict['show_motion_st_win_hard_map'] = F.interpolate(out_dict['show_motion_st_win_hard_map'].view(-1, 1, h//64, w//64), scale_factor=16, mode='nearest')\
                .view(n, t, 1, h // 4, w // 4)
                out_dict['qmask'] = F.interpolate(out_dict['qmask'].view(-1, 1, h//64, w//64), scale_factor=16, mode='nearest')\
                .view(n, t, 1, h // 4, w // 4)
                out_dict['kvmask'] = F.interpolate(out_dict['kvmask'].view(-1, 1, h//64, w//64), scale_factor=16, mode='nearest')\
                .view(n, t, 1, h // 4, w // 4)
                
                
                show_tensor.append(out_dict['lq'])
                show_tensor.append(out_dict['show_motion_st'].repeat(1,1,3,1,1))
                show_tensor.append(out_dict['show_motion_st_win'].repeat(1,1,3,1,1))
                show_tensor.append(out_dict['masks'].repeat(1,t,3,1,1))
                
                show_tensor.append(out_dict['show_motion_st_win_hard_map'].repeat(1,1,3,1,1))
                show_tensor.append(out_dict['qmask'].repeat(1,1,3,1,1))
                show_tensor.append(out_dict['kvmask'].repeat(1,1,3,1,1))


                show_tensor = torch.cat(show_tensor,1)
                

                msg_logger.log_video_images({"video":show_tensor[0],"clip_name":"lq_s_st_out","iter":current_iter}) """
                # msg_logger.log_video_images({"video":out_dict['results'][0],"clip_name":"results","iter":current_iter})
                # msg_logger.log_video_images({"video":out_dict['masks'][0],"clip_name":"masks","iter":current_iter})
                # msg_logger.log_video_images({"video":out_dict['show_motion_s'][0],"clip_name":"show_motion_s","iter":current_iter})
                # msg_logger.log_video_images({"video":out_dict['show_motion_st'][0],"clip_name":"show_motion_st","iter":current_iter})
                

            
            """ if (current_iter -1) % 20 == 0:
                clip_name = train_data["folder"]
                clip_data = model.get_current_training_data()
                B,T,C,H,W = clip_data['lq'].shape
                for i in range(B):
                    clip_seq = torch.stack([clip_data['lq'][i],clip_data['output'][i]])
                    show_lq_gt = torchvision.utils.make_grid(clip_seq.reshape(2,T*C,H,W),nrow=2,padding=5)
                    # show_gt = torchvision.utils.make_grid(clip_data['output'][i].reshape(B,T*C,H,W),nrow=B,padding=5)
                    _,gH,gW = show_lq_gt.shape1000
                    show_lq_gt = show_lq_gt.reshape(1,T,C,gH,gW)
                    
                    
                    seqs_log = {'video':show_lq_gt,'clip_name':clip_name[0],'iter':current_iter} 
                    msg_logger.log_image_video( seqs_log) """

            # save images
            """ if current_iter % (opt['logger']['show_tf_imgs_freq']) == 0:
                visual_imgs = model.get_current_visuals()
                if tb_logger:
                    for k, v in visual_imgs.items(): 
                        tb_logger.add_images(f'ckpt_imgs/{k}', v.clamp(0, 1), current_iter) """

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)
            

            """ if current_iter % opt['logger']['save_latest_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, -1) """

            # validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0) or new_iter == 1000:
            # if True:
                if len(val_loaders) > 1:
                    logger.warning('Multiple validation datasets are *only* supported by SRModel.')
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger,wandb_logger, opt['val']['save_img'])

                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            #if current_iter % 10 == 0:
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq']//2 == 0) or new_iter == 10 or new_iter == 100 or current_iter % 1000 == 0:
                vis_results = model.get_current_visuals()


                input_s = vis_results['lq'][1, :, :, :].permute(1, 2, 0).cpu().detach().numpy()
                cv2.imwrite(os.path.join(opt['path']['visualization'], "%08d_0_input.png" % (current_iter)),
                            input_s[:, :, ::-1] * 255)

                input_s = vis_results['result'][1, :, :, :].permute(1, 2, 0).cpu().detach().numpy()
                cv2.imwrite(os.path.join(opt['path']['visualization'], "%08d_1_deblurred.png" % (current_iter)),
                            input_s[:, :, ::-1] * 255)

                if 'gt' in vis_results:
                    input_s = vis_results['gt'][1, :, :, :].permute(1, 2, 0).cpu().detach().numpy()
                    cv2.imwrite(os.path.join(opt['path']['visualization'], "%08d_2_gt.png" % (current_iter)),
                                input_s[:, :, ::-1] * 255)


                if 'ori_lq' in vis_results:
                    input_s = vis_results['ori_lq'][1,:,:,:].permute(1, 2, 0).cpu().detach().numpy()
                    cv2.imwrite(os.path.join(opt['path']['visualization'], "%08d_3_ori_input.png" % (current_iter)),
                                input_s[:, :, ::-1] * 255)


            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()
            """ if current_iter >= 10500:
                break """
        """ if current_iter >= 10500:
                break """

        # end of iter

    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()
    if wandb_logger:
        wandb_logger.finish()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)

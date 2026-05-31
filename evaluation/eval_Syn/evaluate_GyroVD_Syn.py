import os
import skimage
import numpy as np
from glob import glob
from natsort import natsorted
from skimage import io
import cv2
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from tqdm import tqdm
import concurrent.futures
import argparse

parser = argparse.ArgumentParser(description='eval arg')
parser.add_argument('--input_dir', type=str, default='')
parser.add_argument('--out_txt', type=str, default='')
parser.add_argument('--gt_root', type=str, default='../../dataset/GyroVD_Syn_test')
parser.add_argument('--core', type=int, default=8)
args = parser.parse_args()

def compute_psnr(image_true, image_test):
    return peak_signal_noise_ratio(image_true, image_test, data_range=1.0)


def compute_ssim(tar_img, prd_img):
    return structural_similarity(tar_img, prd_img, multichannel=True, data_range=1.0)

def proc(filename):
    tar, prd = filename
    tar_img = io.imread(tar)
    tar_img = cv2.rotate(tar_img, cv2.ROTATE_90_CLOCKWISE)
    prd_img = io.imread(prd)

    if prd_img.shape[2] == 4:
        prd_img = prd_img[:,:,:3]

    tar_img = tar_img.astype(np.float32) / 255.0
    prd_img = prd_img.astype(np.float32) / 255.0

    PSNR = compute_psnr(tar_img, prd_img)
    SSIM = compute_ssim(tar_img, prd_img)
    return (PSNR, SSIM)


if __name__ == '__main__':

    if skimage.__version__ != '0.17.2':
        print("please use skimage==0.17.2 and python3")
        exit()
        
    input_dir = args.input_dir
    if args.out_txt == '':
        out_txt = input_dir.split('/')[-3] + '.txt'
    else:
        out_txt = args.out_txt
    print(out_txt)

    # find mapping output path <=> gt path
    with open('datalist/GyroVD_Syn_test.txt', 'rt') as f:
        datalist = f.readlines()

    path_list = []
    gt_list = []
    for txt_line in datalist:
        txt_split = txt_line.strip().split(' ')
        img_list = glob(os.path.join(args.gt_root, txt_split[0], 'gt/*.png'))
        img_list = natsorted(img_list)[2:-2]

        for gt_path in img_list:
            img_name = os.path.basename(gt_path)
            video_name = gt_path.split('/')[-3]

            out_path = os.path.join(args.input_dir, video_name, img_name.replace('_gt.png', '_blur.png'))
            assert os.path.exists(out_path) and os.path.exists(gt_path)

            path_list.append(out_path)
            gt_list.append(gt_path)

    assert len(path_list) == (77*96), "Predicted files not found"
    assert len(gt_list) == (77*96), "Target files not found"


    psnr, ssim, files = [], [], []
    img_files = [(i, j) for i, j in zip(gt_list, path_list)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.core) as executor:
        for filename, PSNR_SSIM in zip(img_files, executor.map(proc, img_files)):
            psnr.append(PSNR_SSIM[0])
            ssim.append(PSNR_SSIM[1])
            files.append(filename[0])


    # evaluation according to the blur sizes
    with open('datalist/statistics/GyroVD_Syn_test_blur_size.txt', 'rt') as f:
        lines = [line.strip().split() for line in f if line.strip()]

    # 2. blur size
    blur_values = np.array([float(x[1]) for x in lines])

    # 3. 1/3, 2/3 분위 계산
    q1 = np.quantile(blur_values, 1 / 3)
    q2 = np.quantile(blur_values, 2 / 3)

    # 4. split list
    small_list = [ln[0].split('/')[1] for ln, b in zip(lines, blur_values) if b <= q1]
    medium_list = [ln[0].split('/')[1] for ln, b in zip(lines, blur_values) if q1 < b <= q2]
    large_list = [ln[0].split('/')[1] for ln, b in zip(lines, blur_values) if b > q2]

    small_psnrs = []
    small_ssims = []
    medium_psnrs = []
    medium_ssims = []
    large_psnrs = []
    large_ssims = []

    txt_list = []
    for i, values in enumerate(files):
        tar_path = values
        tar_path = '/'.join(tar_path.split('/')[-4:])
        img_name = os.path.basename(tar_path)

        if img_name in small_list:
            small_psnrs.append(psnr[i])
            small_ssims.append(ssim[i])
        elif img_name in medium_list:
            medium_psnrs.append(psnr[i])
            medium_ssims.append(ssim[i])
        elif img_name in large_list:
            large_psnrs.append(psnr[i])
            large_ssims.append(ssim[i])
        else:
            import pdb; pdb.set_trace()
            raise Exception('Invalid target image')

        txt = '{:s} {:f} {:f}\n'.format(tar_path, psnr[i], ssim[i])
        txt_list.append(txt)

    # save results on txt file

    avg_psnr = sum(small_psnrs) / len(small_psnrs)
    avg_ssim = sum(small_ssims) / len(small_ssims)

    txt = 'For {:s} dataset PSNR on small set: {:f} SSIM: {:f}\n'.format(input_dir, avg_psnr, avg_ssim)
    print(txt)
    txt_list.append(txt)

    avg_psnr = sum(medium_psnrs) / len(medium_psnrs)
    avg_ssim = sum(medium_ssims) / len(medium_ssims)

    txt = 'For {:s} dataset PSNR on medium set: {:f} SSIM: {:f}\n'.format(input_dir, avg_psnr, avg_ssim)
    print(txt)
    txt_list.append(txt)

    avg_psnr = sum(large_psnrs) / len(large_psnrs)
    avg_ssim = sum(large_ssims) / len(large_ssims)

    txt = 'For {:s} dataset PSNR on large set: {:f} SSIM: {:f}\n'.format(input_dir, avg_psnr, avg_ssim)
    print(txt)
    txt_list.append(txt)

    avg_psnr = sum(psnr) / len(psnr)
    avg_ssim = sum(ssim) / len(ssim)

    txt = 'For {:s} dataset PSNR: {:f} SSIM: {:f}\n'.format(input_dir, avg_psnr, avg_ssim)
    print(txt)
    txt_list.append(txt)

    with open(out_txt, 'wt') as f:
        f.writelines(txt_list)
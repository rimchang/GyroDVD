import argparse
import glob
import os
from pyiqa import create_metric
from tqdm import tqdm
import csv
from time import time
from PIL import Image
import torchvision.transforms.functional as TF
import torch
import pyiqa

def main():
    """Inference demo for pyiqa."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--target', type=str, default=None, help='input image/folder path.'
    )
    parser.add_argument(
        '-r',
        '--ref',
        type=str,
        default=None,
        help='reference image/folder path if needed.',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='reference image/folder path if needed.',
    )
    parser.add_argument(
        '--metric_mode',
        type=str,
        default='NR',
        help='metric mode Full Reference or No Reference. options: FR|NR.',
    )
    parser.add_argument(
        '-m',
        '--metric_name',
        type=str,
        default='PSNR',
        help='IQA metric name, case sensitive.',
    )
    parser.add_argument(
        '--save_file', type=str, default=None, help='path to save results.'
    )

    # Add a --verbose flag
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',  # This makes it a flag (True when used, False otherwise)
        help='Enable verbose output',
    )

    args = parser.parse_args()

    metric_name = args.metric_name

    # set up IQA model
    iqa_model = create_metric(
        metric_name, metric_mode=args.metric_mode, device=args.device
    )
    metric_mode = iqa_model.metric_mode

    with open('datalist/GyroVD_Real.txt', 'rt') as f:
        video_list = f.readlines()
    video_list = [video.strip() for video in video_list]

    input_paths = []
    vid2day = {}
    for video_path in video_list:

        day_name = video_path.split('/')[0]
        video_name = video_path.split('/')[1]

        input_path = sorted(glob.glob(os.path.join(args.target, day_name, video_name, '*.png')))
        if len(input_path) == 0:
            input_path = sorted(glob.glob(os.path.join(args.target, video_name, '*.png')))
        input_paths += input_path
        vid2day[video_name] = day_name

    print(len(input_paths))
    assert len(input_paths) == 96*100, print(len(input_paths))

    if args.save_file:
        os.makedirs(os.path.dirname(args.save_file), exist_ok = True)
        sf = open(args.save_file, 'w')
        sfwriter = csv.writer(sf)

    new_width = 1080//2
    new_height = 1920//2

    avg_score = 0
    test_img_num = len(input_paths)
    if not 'fid' in metric_name:
        pbar = tqdm(total=test_img_num, unit='image')
        for idx, img_path in enumerate(input_paths):
            img_name = os.path.basename(img_path)
            video_name = img_path.split('/')[-2]
            if metric_mode == 'FR':
                ref_img_path = ref_paths[idx]
            else:
                ref_img_path = None

            start_time = time()

            img = Image.open(img_path)
            img = img.resize((new_width, new_height), resample=Image.BICUBIC)
            img_tensor = TF.to_tensor(img).unsqueeze(0)

            score = iqa_model(img_tensor, ref_img_path).cpu().item()
            # score = iqa_model(img_path, ref_img_path).cpu().item()
            # score = max(score, 0) # for negative values of brisque
            end_time = time()
            avg_score += score

            pbar.update(1)
            # pbar.set_description(f'{metric_name} of {img_name}: {score}')
            # pbar.write(
            #     f'{metric_name} of {img_name}: {score}\tTime: {end_time - start_time:.2f}s'
            # )

            if args.save_file:
                sfwriter.writerow([video_name + '/' + img_name, score])

        pbar.close()
        avg_score /= test_img_num
    else:
        assert os.path.isdir(args.target) and os.path.isdir(args.ref), (
            'input path must be a folder for FID.'
        )
        avg_score = iqa_model(args.target, args.ref)

    if args.verbose and torch.cuda.is_available():
        print(torch.cuda.memory_summary())

    msg = f'Average {metric_name} score of {args.target} with {test_img_num} images is: {avg_score}'
    print(msg)

    if args.save_file:
        sfwriter.writerow(['AVG', avg_score])

    if args.save_file:
        sf.close()

    if args.save_file:
        print(f'Done! Results are in {args.save_file}.')
    else:
        print(f'Done!')


if __name__ == '__main__':
    if pyiqa.__version__ != '0.1.14.1':
        print("please use pyiqa==0.1.14.1")
        exit()
    main()

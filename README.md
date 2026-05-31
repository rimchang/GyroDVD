## Gyro-based Deep Video Deblurring
##### [Project](http://cg.postech.ac.kr/research/GyroDVD/) | [Paper](https://cg.postech.ac.kr/researches/GyroDVD/assets/pdf/GyroDVD.pdf) | [Supple](https://cg.postech.ac.kr/researches/GyroDVD/assets/pdf/GyroDVD_supplementary_materials.zip)

#### Official Implementation of CVPR 2026 Paper 

> Gyro-based Deep Video Deblurring<br>
> Jaesung Rim<sup>1</sup>, Woohyeok Kim<sup>1</sup>, Haeyun Lee<sup>2</sup>, Heemin Yang<sup>1</sup>, Ke Wang<sup>3</sup>, Sunghyun Cho<sup>1</sup><br>
> <sup>1</sup>POSTECH, <sup>2</sup>KOREATECH, <sup>3</sup>Pika Labs<br>
> *IEEE Conference on Computer Vision and Pattern Recognition (**CVPR**) 2026*<br>


## Install

```
conda create -n GyroDVD python=3.8
conda activate GyroDVD
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
pip install git+https://github.com/cheind/pytorch-debayer
python setup.py develop
```

## Download

### Dataset [[Hugging Face]](https://huggingface.co/datasets/rimchang/GyroDVD/tree/main) 

<details>
<summary><strong>Descriptions</strong> (click) </summary>

For the training set, noise and saturation pixels are synthesized on-the-fly in `GyroDVD_model.py`. Please refer to [RSBlurPipeline](https://github.com/rimchang/GyroDVD/blob/0c4be0d005d15d195cb67b30c9f74781d1b20cd6/basicsr/models/GyroDVD_model.py#L310).

For the validation and test sets, noise and saturation pixels have already been applied and are included in the released dataset.

- GyroVD_Syn_train.tar.gz : 50,500 frames for training.
  - GyroVD_Syn_train_tau.tar.gz : pre-computed tau. (only required for training.) 
- GyroVD_Syn_val.tar.gz : 5,000 frames for validation. 
  - GyroVD_Syn_val_tau.tar.gz : pre-computed tau for validation set. (only required for training.) 
- GyroVD_Syn_test.tar.gz : 7,700 frames for evaluation. 
  - GyroVD_Syn_test_tau.tar.gz : pre-computed tau for test set. (only required for training.) 
- GyroVD_Real.tar.gz: 10,000 real-world frames for evaluation.
#### To facilitate future research, we also provide accelerometer and magnetometer data.


### The GyroVD-Syn dataset

```bash
# GyroVD_Syn_train.tar.gz
GyroVD_Syn_train
├── 0413/VID_20250412_101526
│   ├── blur # blurred frames
│   │   ├── 000001_160466752524897_160466785733026_160466786578871_avg09_blur.png
│   │   ├── 000001_(exp_start_1th)_(exp_start_9th)_(exp_end)_avg09_blur.png
│   │   ...
│   ├── gt # gt frames
│   │   ├── 000001_160466752524897_160466785733026_160466786578871_avg09_gt.png
│   │   ...
│   ├── meta_info
│   │   ├── gyro.csv # gyroscope data
│   │   ├── accel.csv # accelerometer data
│   │   ├── magnetic.csv # magnetometer data
│   │   ├── cam_metainfo.csv # camera meta data
│   ├── rot # pre-computed rotation matrices
│   ├── sat_mask # saturation mask for RSBlur pipeline
├── 0413/VID_20250412_101616
│   ...
...
```

### The GyroVD-Real dataset
```bash
# GyroVD_Real.tar.gz
GyroVD-Real
├── 1029_GP/VID_20251029_013931
│   ├── blur # blurred frames
│   │   ├── 000001_193795012257259_193795012257259_193795111113483_blur.jpg
│   │   ...
│   ├── meta_info
│   │   ├── gyro.csv # gyroscope data
│   │   ├── accel.csv # accelerometer data
│   │   ├── magnetic.csv # magnetometer data
│   │   ├── cam_metainfo.csv # camera meta data
│   ...
...
```
</details>

### Deblurred Results [[Hugging Face]](https://huggingface.co/datasets/rimchang/GyroDVD_results/tree/main) 


### Pre-trained models [[link]](./model_zoos/)
<details>
<summary><strong>Descriptions</strong> (click) </summary>

- GyroDVD_48.pth: Weight of GyroDVD-48.
- GyroDVD_64.pth: Weight of GyroDVD-64.
- GyroDVD_96.pth: Weight of GyroDVD-96.
- GyroDVD_128.pth: Weight of GyroDVD-128.
- raft-small.pth: Weight of RAFT_small for optical flow estimation.
</details>

## Demo
```bash
# demo of samples from GyroVD-Real and GyroVD-Syn
python inference_GyroVDReal.py --dataset_root=demo/GyroVD_Real  --out_path=results/GyroDVD_128_Real_demo --model_size=128
python inference_GyroVDSyn.py --dataset_root=demo/GyroVD_Syn  --out_path=results/GyroDVD_128_Syn_demo --model_size=128
```

## Testing

```bash
# datasets should be located in dataset
# pre-trained weights should be located in model_zoos

## test on GyroVD-Real
python inference_GyroVDReal.py --dataset_root=dataset/GyroVD_Real  --out_path=results/GyroDVD_128_Real --model_size=128

## test on GyroVD-Syn
python inference_GyroVDSyn.py --dataset_root=dataset/GyroVD_Syn_test  --out_path=results/GyroDVD_128_Syn --model_size=128
```

## Evaluation

```bash
# ./GyroDVD

# compute PSNR and SSIM on GyroVD-Syn
python evaluation/eval_Syn/evaluate_GyroVD_Syn.py --input_dir=results/GyroDVD_128_Syn --gt_root=dataset/GyroVD_Syn_test --out_txt=results/GyroDVD_128_Syn.txt

# Compute TOP-IQA on GyroVD-Real. Requires IQA-PyTorch.
python evaluation/eval_Real/inference_iqa_resize.py -t=results/GyroDVD_128_Real -m=topiq_nr --save_file=results/GyroDVD_128_Real_topiq.txt;

# Compute BRISQUE and NIQE on GyroVD-Real. Tested on MATLAB R2023
# We found that BRISQUE and NIQE in IQA-PyTorch are sometimes not robust, so we use the MATLAB implementations for evaluation.
# ./evaluation/eval_Real
addpath('compute_iqa_matlab'); 
compute_iqa_matlab('../../results/GyroDVD_128_Real', 'brisque', '../../results/GyroDVD_128_Real_brisque.txt');
compute_iqa_matlab('../../results/GyroDVD_128_Real', 'niqe', '../../results/GyroDVD_128_Real_niqe.txt');

```

## Training

```bash
# Download the GyroVD_Syn_train_tau dataset before training.

# GyroDVD is trained with batch size of 4
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4016 basicsr/train.py -opt options/train/GyroDVD_128_train.yml --auto_resume --launcher pytorch

```

## License

The GyroVD dataset is released under CC BY 4.0 license.

## Acknowledgements

The code is based on [BasicSR](https://github.com/XPixelGroup/BasicSR), [RAFT](https://github.com/princeton-vl/RAFT), [ShiftNet](https://github.com/dasongli1/shift-net) and [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch).

## Citation

If you use our dataset and code for your research, please cite our paper.

```bibtex
@inproceedings{GyroDVD_rim,
 title={Gyro-based Deep Video Deblurring},
 author={Jaesung Rim and Woohyeok Kim and Haeyun Lee and Heemin Yang and Ke Wang and Sunghyun Cho},
 booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
 year={2026}
}
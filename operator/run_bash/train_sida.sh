#!/bin/bash

source activate ambient

sigma=0.2
corruption_probability=1.0
dp=1
consistency_coeff=0.0

CUDA_VISIBLE_DEVICES=1,2,3,4,6,7 python -m torch.distributed.run --standalone --nproc_per_node=6 sida_train.py \
  --alpha 1 \
  --tmax 800 \
  --init_sigma 2.5 \
  --outdir 'sida_result/cifar10' \
  --resume 'sida_result/cifar10' \
  --nosubdir 0 \
  --data '/home/tychen/Dataset/cifar10-32x32.zip' \
  --arch ddpmpp \
  --edm_model 'cifar10_pretrain/00001-cifar10-32x32-uncond-ddpmpp-edm-gpus5-batch1600-fp32-hO3N5/network-snapshot-200000.pkl' \
  --metrics fid50k_full,is50k \
  --tick 10 \
  --snap 50 \
  --dump 200 \
  --lr 1e-5 \
  --glr 1e-5 \
  --fp16 0 \
  --ls 1 \
  --lsg 100 \
  --lsd 1 \
  --lsg_gan 0.01 \
  --duration 300 \
  --data_stat '/home/tychen/Dataset/fid-refs/cifar10-32x32.npz' \
  --detector_url 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt' \
  --use_gan 1 \
  --save_best_and_last 1 \
  --sigma=$sigma \
  --corruption_probability=$corruption_probability \
  --dataset_keep_percentage=$dp \
  --consistency_coeff=$consistency_coeff \
  --expr_id="cifar_cp${corruption_probability}_sigma${sigma}_dp${dp}_cc${consistency_coeff}" \
  --batch=576




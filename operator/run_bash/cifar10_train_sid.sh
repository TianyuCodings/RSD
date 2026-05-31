#!/bin/bash

 
 
sigma=0.0
corruption_probability=0.0
dp=0.1
consistency_coeff=0.0
edm_model=$1

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --standalone --nproc_per_node=4 sid_train.py \
  --alpha 1.2 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 256 \
    --batch-gpu 64 \
    --outdir 'cifar10/SiD' \
    --data '/home/ubuntu/ext-mamba-illinois/yasi/vision_data/cifar10-32x32.zip' \
    --arch ddpmpp \
    --edm_model $1 \
    --metrics fid50k_full \
    --tick 10 \
    --snap 50 \
    --dump 500 \
    --lr 1e-5 \
    --glr 1e-5 \
    --fp16 0 \
    --ls 1 \
    --lsg 100 \
    --duration 100 \
    --data_stat 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz' \
    --sigma=$sigma \
    --corruption_probability=$corruption_probability \
    --dataset_keep_percentage=$dp \
    --consistency_coeff=$consistency_coeff \
    --expr_id="cifar10_cp${corruption_probability}_sigma${sigma}_dp${dp}_cc${consistency_coeff}" \
    
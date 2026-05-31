#!/bin/bash

 
edm_model=$1
sigma=0.2
corruption_probability=1.0
dp=1
consistency_coeff=0.0

 
 
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc_per_node=4 sid_train.py \
    --alpha 1.0 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 512 \
    --batch-gpu 128 \
    --outdir 'afhq/SiD/' \
    --data '/home/ubuntu/ext-mamba-illinois/yasi/afhqv2-64x64.zip' \
    --arch ddpmpp \
    --edm_model $edm_model \
    --metrics fid50k_full \
    --tick 10 \
    --snap 50 \
    --dump 500 \
    --lr 5e-6 \
    --glr 5e-6 \
    --fp16 1 \
    --ls 1 \
    --lsg 100 \
    --dropout 0.05 \
    --augment 0.15 \
    --cres 1,2,2,2 \
    --duration 100 \
    --data_stat '/home/ubuntu/ext-mamba-illinois/yasi/afhqv2-64x64.npz' \
    --sigma=$sigma \
    --corruption_probability=$corruption_probability \
    --dataset_keep_percentage=$dp \
    --consistency_coeff=$consistency_coeff \
    --expr_id="afhq_cp${corruption_probability}_sigma${sigma}_dp${dp}_cc${consistency_coeff}" \
    
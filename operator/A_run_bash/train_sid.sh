#!/bin/bash

 

sigma=0.00
corruption_probability=1.0
dp=1
consistency_coeff=0.0
edm_model=ffhq_pretrain/edm-ffhq-64x64-uncond-vp.pkl

CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.run --standalone --nproc_per_node=2 sid_train.py \
  --alpha 1.2 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 2 \
    --batch-gpu 4 \
    --outdir 'A/ffhq/SiD/' \
    --data '/home/tychen/Dataset/ffhq-64x64.zip' \
    --arch ddpmpp \
    --edm_model $edm_model \
    --metrics fid01k_full \
    --tick 10 \
    --snap 50 \
    --dump 500 \
    --lr 1e-5 \
    --glr 1e-5 \
    --fp16 1 \
    --ls 1 \
    --lsg 100 \
    --dropout 0.05 \
    --augment 0.15 \
    --cres 1,2,2,2 \
    --g_beta1 0.9 \
    --duration 100 \
    --data_stat '/home/tychen/Dataset/fid-refs/ffhq-64x64.npz' \
    --sigma=$sigma \
    --corruption_probability=$corruption_probability \
    --dataset_keep_percentage=$dp \
    --consistency_coeff=$consistency_coeff \
    --expr_id="test" \
    --operator='gaussian' \
    
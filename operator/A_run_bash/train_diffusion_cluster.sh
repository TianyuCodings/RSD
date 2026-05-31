#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=8 train.py \
    --outdir=/blob/v-tianyuchen/Projects/Ambient_A/ffhq_pretrain \
    --data=/blob/v-tianyuchen/datasets/ffhq-64x64.zip \
    --sigma=0.0 \
    --corruption_probability=1.0 \
    --dataset_keep_percentage=1.0 \
    --consistency_coeff=0.0 \
    --expr_id="test" \
    --cond=0 \
    --arch=ddpmpp  \
    --cres=1,2,2,2 \
    --dropout=0.05 \
    --augment=0.15 \
    --batch=256 \
    --lr=2e-4 \
    --operator='gaussian' #[gaussian, identity]

#!/bin/bash
 
wandb login $WANDB_API_KEY

export CUDA_VISIBLE_DEVICES=3,4
export WANDB_ENTITY="ambient_ty"  # Replace with your W&B entity

  torchrun --standalone --nproc_per_node=2 sid_train.py \
  --alpha 1.2 \
  --tmax 800 \
  --init_sigma 2.5 \
  --batch 512 \
  --batch-gpu 32 \
  --outdir "./test" \
  --data '/home/tychen/Dataset/celeba_hq-64x64.zip' \
  --arch ddpmpp \
  --edm_model '/home/tychen/Dataset/Ambient_diffusion_pretrain_ckpt/network-snapshot-195121.pkl' \
  --metrics fid500_full \
  --tick 10 \
  --snap 50 \
  --dump 500 \
  --lr 1e-5 \
  --glr 1e-5 \
  --fp16 1 \
  --ls 1 \
  --lsg 100 \
  --dropout 0.1 \
  --augment 0.15 \
  --cres 1,2,2,2 \
  --g_beta1 0.9 \
  --duration 100 \
  --data_stat '/home/tychen/Dataset/fid-refs/celeba_hq-64x64.npz' \
  --corruption_probability=0.8 --delta_probability=0.1 \
  --experiment_name=test \
  --max_grad_norm=1.0 \
  --cond=0
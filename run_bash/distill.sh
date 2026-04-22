#!/bin/bash
# Stage 2: Distill a pretrained teacher into a one-step generator via RSD.
#
# Usage: bash run_bash/distill.sh <dataset> <teacher_ckpt> [num_gpus]
#   dataset:      ffhq | celeba | afhq | cifar10
#   teacher_ckpt: path to pretrained teacher .pkl file
#   num_gpus:     number of GPUs (default: 4)
#
# Example: bash run_bash/distill.sh ffhq ffhq/ffhq_pretrain/.../network-snapshot-XXXXXX.pkl 8
#
# ============================================================
# Set dataset paths before running
FFHQ_DATA="/path/to/ffhq-64x64.zip"
FFHQ_STAT="/path/to/ffhq-64x64.npz"
CELEBA_DATA="/path/to/celeba_hq-64x64.zip"
CELEBA_STAT="/path/to/celeba_hq-64x64.npz"
AFHQ_DATA="/path/to/afhqv2-64x64.zip"
AFHQ_STAT="/path/to/afhqv2-64x64.npz"
CIFAR10_DATA="/path/to/cifar10-32x32.zip"
CIFAR10_STAT="https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz"
# ============================================================

DATASET=${1:-ffhq}
EDM_MODEL=$2
NUM_GPUS=${3:-4}

if [ -z "$EDM_MODEL" ]; then
  echo "Usage: bash run_bash/distill.sh <dataset> <teacher_ckpt> [num_gpus]"
  exit 1
fi

sigma=0.2
corruption_probability=1.0
dp=1
consistency_coeff=0.0

case $DATASET in
  ffhq)
    outdir="ffhq/RSD"
    dataset_path=$FFHQ_DATA; data_stat=$FFHQ_STAT
    alpha=1.2; lr=1e-5; glr=1e-5; g_beta1=0.9; fp16=1
    batch=512; batch_gpu=128; dropout=0.05; augment=0.15; cres=1,2,2,2
    ;;
  celeba)
    outdir="celebahq/RSD"
    dataset_path=$CELEBA_DATA; data_stat=$CELEBA_STAT
    alpha=1.2; lr=1e-5; glr=1e-5; g_beta1=0.9; fp16=1
    batch=512; batch_gpu=128; dropout=0.05; augment=0.15; cres=1,2,2,2
    ;;
  afhq)
    outdir="afhq/RSD"
    dataset_path=$AFHQ_DATA; data_stat=$AFHQ_STAT
    alpha=1.0; lr=5e-6; glr=5e-6; g_beta1=0; fp16=1
    batch=512; batch_gpu=128; dropout=0.05; augment=0.15; cres=1,2,2,2
    ;;
  cifar10)
    outdir="cifar10/RSD"
    dataset_path=$CIFAR10_DATA; data_stat=$CIFAR10_STAT
    sigma=0.1
    alpha=1.2; lr=1e-5; glr=1e-5; g_beta1=0; fp16=0
    batch=256; batch_gpu=64; dropout=0.13; augment=0.12; cres=""
    ;;
  *)
    echo "Unknown dataset: $DATASET. Choose from: ffhq | celeba | afhq | cifar10"
    exit 1
    ;;
esac

EXTRA_ARGS="--dropout=$dropout --augment=$augment --batch=$batch --batch-gpu=$batch_gpu"
if [ -n "$cres" ]; then EXTRA_ARGS="$EXTRA_ARGS --cres=$cres"; fi

python -m torch.distributed.run --standalone --nproc_per_node=$NUM_GPUS rsd_train.py \
    --alpha $alpha \
    --tmax 800 \
    --init_sigma 2.5 \
    --outdir $outdir \
    --data $dataset_path \
    --arch ddpmpp \
    --edm_model $EDM_MODEL \
    --metrics fid50k_full \
    --tick 10 \
    --snap 50 \
    --dump 500 \
    --lr $lr \
    --glr $glr \
    --g_beta1 $g_beta1 \
    --fp16 $fp16 \
    --ls 1 \
    --lsg 100 \
    --duration 100 \
    --data_stat $data_stat \
    --sigma=$sigma \
    --corruption_probability=$corruption_probability \
    --dataset_keep_percentage=$dp \
    --consistency_coeff=$consistency_coeff \
    --expr_id="${DATASET}_cp${corruption_probability}_sigma${sigma}_dp${dp}_cc${consistency_coeff}" \
    $EXTRA_ARGS

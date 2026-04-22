#!/bin/bash
# Stage 1: Pretrain a corruption-aware diffusion teacher on noisy data.
#
# Usage: bash run_bash/pretrain.sh <dataset> [num_gpus]
#   dataset:  ffhq | celeba | afhq | cifar10
#   num_gpus: number of GPUs (default: 4)
#
# Example: bash run_bash/pretrain.sh ffhq 8
#
# ============================================================
# Set dataset paths before running
FFHQ_DATA="/path/to/ffhq-64x64.zip"
CELEBA_DATA="/path/to/celeba_hq-64x64.zip"
AFHQ_DATA="/path/to/afhqv2-64x64.zip"
CIFAR10_DATA="/path/to/cifar10-32x32.zip"
# ============================================================

DATASET=${1:-ffhq}
NUM_GPUS=${2:-4}

sigma=0.2
corruption_probability=1
dp=1
consistency_coeff=0.0

case $DATASET in
  ffhq)
    outdir="ffhq/ffhq_pretrain"
    dataset_path=$FFHQ_DATA
    arch=ddpmpp; cres=1,2,2,2; dropout=0.05; augment=0.15; batch=256; lr=2e-4
    ;;
  celeba)
    outdir="celebahq/celebahq_pretrain"
    dataset_path=$CELEBA_DATA
    arch=ddpmpp; cres=1,2,2,2; dropout=0.15; augment=0.15; batch=256; lr=2e-4
    ;;
  afhq)
    outdir="afhq/afhq_pretrain"
    dataset_path=$AFHQ_DATA
    arch=ddpmpp; cres=1,2,2,2; dropout=0.25; augment=0.15; batch=256; lr=2e-4
    ;;
  cifar10)
    outdir="cifar10/cifar_pretrain"
    dataset_path=$CIFAR10_DATA
    sigma=0.1
    arch=ddpmpp; cres=""; dropout=0.13; augment=0.12; batch=512; lr=1e-4
    ;;
  *)
    echo "Unknown dataset: $DATASET. Choose from: ffhq | celeba | afhq | cifar10"
    exit 1
    ;;
esac

EXTRA_ARGS="--arch=$arch --dropout=$dropout --augment=$augment --batch=$batch --lr=$lr"
if [ -n "$cres" ]; then EXTRA_ARGS="$EXTRA_ARGS --cres=$cres"; fi

python -m torch.distributed.run --nproc_per_node=$NUM_GPUS train.py \
    --outdir=$outdir \
    --data=$dataset_path \
    --sigma=$sigma \
    --corruption_probability=$corruption_probability \
    --dataset_keep_percentage=$dp \
    --consistency_coeff=$consistency_coeff \
    --expr_id="${DATASET}_cp${corruption_probability}_sigma${sigma}_dp${dp}_cc${consistency_coeff}" \
    --cond=0 \
    $EXTRA_ARGS

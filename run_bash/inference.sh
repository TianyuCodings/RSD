#!/bin/bash
# Generate images from a pretrained teacher using truncated EDM sampling.
#
# Usage: bash run_bash/inference.sh <network_pkl> <outdir> [num_gpus]
#
# Example: bash run_bash/inference.sh ffhq/ffhq_pretrain/.../network-snapshot-XXXXXX.pkl outputs/generated 8

NETWORK=$1
OUTDIR=${2:-"outputs/generated"}
NUM_GPUS=${3:-8}

if [ -z "$NETWORK" ]; then
  echo "Usage: bash run_bash/inference.sh <network_pkl> <outdir> [num_gpus]"
  exit 1
fi

python -m torch.distributed.run --nproc_per_node=$NUM_GPUS --standalone scripts/generate.py \
    --seeds=0-49999 \
    --network=$NETWORK \
    --outdir=$OUTDIR \
    --batch=64 \
    --stop_variance=0.2

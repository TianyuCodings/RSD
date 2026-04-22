#!/bin/bash
# Compute FID of generated images against reference statistics.
#
# Usage: bash run_bash/evaluate.sh <gen_path> <ref_stats.npz> [out.json] [num_gpus]
#
# Example: bash run_bash/evaluate.sh outputs/generated /path/to/ffhq-64x64.npz fid_result.json 8

GEN_PATH=$1
REF_STATS=$2
OUT_PATH=${3:-"fid_result.json"}
NUM_GPUS=${4:-8}

if [ -z "$GEN_PATH" ] || [ -z "$REF_STATS" ]; then
  echo "Usage: bash run_bash/evaluate.sh <gen_path> <ref_stats.npz> [out.json] [num_gpus]"
  exit 1
fi

python -m torch.distributed.run --standalone scripts/eval_fid.py \
    --gen_path=$GEN_PATH \
    --ref_stats=$REF_STATS \
    --out_path=$OUT_PATH

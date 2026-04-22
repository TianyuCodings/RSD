#!/bin/bash
# One-step generation from a distilled RSD generator.
#
# Usage: bash run_bash/generate_rsd.sh <network_pkl> [outdir]
#
# Example: bash run_bash/generate_rsd.sh ffhq/RSD/.../network-snapshot-XXXXXX.pkl outputs/rsd_generated

NETWORK=$1
OUTDIR=${2:-"outputs/rsd_generated"}

if [ -z "$NETWORK" ]; then
  echo "Usage: bash run_bash/generate_rsd.sh <network_pkl> [outdir]"
  exit 1
fi

python scripts/rsd_generate_onestep.py \
    --outdir=$OUTDIR \
    --seeds=0-63 \
    --batch=64 \
    --network=$NETWORK

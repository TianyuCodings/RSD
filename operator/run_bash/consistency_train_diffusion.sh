

# conda activate ambient
outdir="ffhq/ffhq_consistency"
dataset_path="/home/ubuntu/ext-mamba-illinois/yasi/vision_data/ffhq-64x64.zip"
sigma=0.2
corruption_probability=1
dp=1
consistency_coeff=1.0

if [ -z "$1" ]
then
    cuda_group=0
else
    cuda_group=$1
fi
cuda_devices=$(($cuda_group*4+0)),$(($cuda_group*4+1)),$(($cuda_group*4+2)),$(($cuda_group*4+3))

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=8 train.py \
    --outdir=$outdir \
    --data=$dataset_path \
    --sigma=$sigma \
    --corruption_probability=$corruption_probability \
    --dataset_keep_percentage=$dp \
    --consistency_coeff=$consistency_coeff \
    --expr_id="ffhq_cp${corruption_probability}_sigma${sigma}_dp${dp}_cc${consistency_coeff}" \
    --cond=0 \
    --arch=ddpmpp  \
    --cres=1,2,2,2 \
    --dropout=0.05 \
    --augment=0.15 \
    --batch=256 \
    --lr=2e-4 \
    --consistency_batch_size=16 \
    --transfer="/home/ubuntu/ext-mamba-illinois/yasi/Ambient/ffhq/ffhq_pretrain/00000-ffhq-64x64-uncond-ddpmpp-edm-gpus8-batch256-fp32-jhQbf/network-snapshot-095334.pkl"

    # --resume="/home/ubuntu/ext-mamba-illinois/yasi/Ambient/ffhq/ffhq_pretrain/00000-ffhq-64x64-uncond-ddpmpp-edm-gpus8-batch256-fp32-jhQbf/training-state-075264.pt"


    # --batch=256 
    # --lr=2e-4

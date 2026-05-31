
# conda activate ambient
outdir="afhq/afhq_pretrain"
dataset_path="/home/ubuntu/ext-mamba-illinois/yasi/afhqv2-64x64.zip"
sigma=0.2
corruption_probability=1
dp=1
consistency_coeff=0.0


CUDA_VISIBLE_DEVICES=0,1,2,3  python -m torch.distributed.run --nproc_per_node=4 train.py \
    --outdir=$outdir \
    --data=$dataset_path \
    --sigma=$sigma \
    --corruption_probability=$corruption_probability \
    --dataset_keep_percentage=$dp \
    --consistency_coeff=$consistency_coeff \
    --expr_id="afhq_cp${corruption_probability}_sigma${sigma}_dp${dp}_cc${consistency_coeff}" \
    --cond=0 \
    --arch=ddpmpp  \
    --cres=1,2,2,2 \
    --dropout=0.25 \
    --augment=0.15 \
    --batch=256 \
    --lr=2e-4 
    # --resume="/home/ubuntu/ext-mamba-illinois/yasi/Ambient/ffhq_pretrain/00006-ffhq-64x64-uncond-ddpmpp-edm-gpus8-batch256-fp32-whCCA/training-state-176385.pt"
    


    # --batch=256 
    # --lr=2e-4

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 python -m torch.distributed.run --nproc_per_node=7 --standalone generate.py \
#    --seeds=0-10000 \
#    --network=giannisdaras/ambient_laws_cifar_sigma_0.05_corruption_0.9_keep_1.0 \
#    --outdir="outputs" \
#    --batch=4


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=8 --standalone generate.py \
    --seeds=0-49999 \
    --network="/home/ubuntu/ext-mamba-illinois/yasi/Ambient/ffhq_pretrain/00006-ffhq-64x64-uncond-ddpmpp-edm-gpus8-batch256-fp32-whCCA/network-snapshot-176385.pkl" \
    --outdir="/data/xuchenheng/ffhq_pretrain/generate3" \
    --batch=64 \
    --stop_variance=0.2


# 032948.pkl, FID 20
# 081102  19.054534937466045

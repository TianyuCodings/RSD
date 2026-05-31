 
wandb login $WANDB_API_KEY

export CUDA_VISIBLE_DEVICES=3,4

#data=/blob/v-tianyuchen/datasets/celeba_hq-64x64.zip
#output=/blob/v-tianyuchen/Projects/Ambient_A/test
data=/home/tychen/Dataset/celeba_hq-64x64.zip
output=./test

# Set up wandb environment variables
export WANDB_ENTITY="ambient_ty"  # Replace with your W&B entity

torchrun --standalone --nproc_per_node=2 train.py \
--outdir=$output \
--experiment_name=test --dump=200  \
--cond=0 --arch=ddpmpp --precond=ambient --cres=1,2,2,2 --lr=2e-4 --dropout=0.1 --augment=0.15 \
--data=$data \
--norm=2 --max_grad_norm=1.0 --mask_full_rgb=True --corruption_probability=0.8 --delta_probability=0.1 --batch=32 --max_size=30000 \
--experiment_name=test \
--transfer='/home/tychen/Dataset/Ambient_diffusion_pretrain_ckpt/network-snapshot-195121.pkl'
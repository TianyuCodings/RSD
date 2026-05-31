 
wandb login $WANDB_API_KEY

export WANDB_ENTITY="ambient_ty"  # Replace with your W&B entity
sigma=0.0

############################################ celeba ############################################

# corruption_probability=0.8
# torchrun --standalone --nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) sid_train.py \
#     --alpha 1.2 \
#     --tmax 800 \
#     --init_sigma 2.5 \
#     --batch 512 \
#     --batch-gpu 16 \
#     --outdir "/blob/v-tianyuchen/Projects/Ambient_A/celeba/SiD_ambient_diffusion_random_inpainting_corruption_probability_${corruption_probability}_sigma_0.0/" \
#     --data '/blob/v-tianyuchen/datasets/celeba_hq-64x64.zip' \
#     --arch ddpmpp \
#     --edm_model /blob/v-tianyuchen/Projects/Ambient_A/ambient_diffusion_ckpt/checkpoints/celeba08/network-snapshot-195121.pkl \
#     --metrics fid50k_full \
#     --tick 10 \
#     --snap 50 \
#     --dump 500 \
#     --lr 1e-5 \
#     --glr 1e-5 \
#     --fp16 1 \
#     --ls 1 \
#     --lsg 100 \
#     --dropout 0.05 \
#     --augment 0.15 \
#     --cres 1,2,2,2 \
#     --g_beta1 0.9 \
#     --duration 100 \
#     --data_stat '/blob/v-tianyuchen/datasets/celeba_hq-64x64.npz' \
#     --corruption_probability=${corruption_probability} \
#     --delta_probability=0.1 \
#     --max_grad_norm=1.0 \
#     --experiment_name=fix_distill_celeba_random_${corruption_probability}_sigma_${sigma}


# corruption_probability=0.9
# torchrun --standalone --nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) sid_train.py \
#     --alpha 1.2 \
#     --tmax 800 \
#     --init_sigma 2.5 \
#     --batch 512 \
#     --batch-gpu 16 \
#     --outdir "/blob/v-tianyuchen/Projects/Ambient_A/celeba/SiD_ambient_diffusion_random_inpainting_corruption_probability_${corruption_probability}_sigma_0.0/" \
#     --data '/blob/v-tianyuchen/datasets/celeba_hq-64x64.zip' \
#     --arch ddpmpp \
#     --edm_model /blob/v-tianyuchen/Projects/Ambient_A/ambient_diffusion_ckpt/checkpoints/celeba09/network-snapshot-130181.pkl \
#     --metrics fid50k_full \
#     --tick 10 \
#     --snap 50 \
#     --dump 500 \
#     --lr 1e-5 \
#     --glr 1e-5 \
#     --fp16 1 \
#     --ls 1 \
#     --lsg 100 \
#     --dropout 0.05 \
#     --augment 0.15 \
#     --cres 1,2,2,2 \
#     --g_beta1 0.9 \
#     --duration 100 \
#     --data_stat '/blob/v-tianyuchen/datasets/celeba_hq-64x64.npz' \
#     --corruption_probability=${corruption_probability} \
#     --delta_probability=0.1 \
#     --max_grad_norm=1.0 \
#     --experiment_name=fix_distill_celeba_random_${corruption_probability}_sigma_${sigma}


# corruption_probability=0.6
# torchrun --standalone --nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) sid_train.py \
#     --alpha 1.2 \
#     --tmax 800 \
#     --init_sigma 2.5 \
#     --batch 512 \
#     --batch-gpu 16 \
#     --outdir "/blob/v-tianyuchen/Projects/Ambient_A/celeba/SiD_ambient_diffusion_random_inpainting_corruption_probability_${corruption_probability}_sigma_0.0/" \
#     --data '/blob/v-tianyuchen/datasets/celeba_hq-64x64.zip' \
#     --arch ddpmpp \
#     --edm_model /blob/v-tianyuchen/Projects/Ambient_A/ambient_diffusion_ckpt/checkpoints/celeba06/network-snapshot-185083.pkl \
#     --metrics fid50k_full \
#     --tick 10 \
#     --snap 50 \
#     --dump 500 \
#     --lr 1e-5 \
#     --glr 1e-5 \
#     --fp16 1 \
#     --ls 1 \
#     --lsg 100 \
#     --dropout 0.05 \
#     --augment 0.15 \
#     --cres 1,2,2,2 \
#     --g_beta1 0.9 \
#     --duration 100 \
#     --data_stat '/blob/v-tianyuchen/datasets/celeba_hq-64x64.npz' \
#     --corruption_probability=${corruption_probability} \
#     --delta_probability=0.1 \
#     --max_grad_norm=1.0 \
#     --experiment_name=fix_distill_celeba_random_${corruption_probability}_sigma_${sigma}



######################################### cifar 10 #########################################

# corruption_probability=0.8
# torchrun --standalone --nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) sid_train.py \
#     --alpha 1.2 \
#     --tmax 800 \
#     --init_sigma 2.5 \
#     --batch 256 \
#     --batch-gpu 16 \
#     --outdir "/blob/v-tianyuchen/Projects/Ambient_A/cifar10/SiD_ambient_diffusion_random_inpainting_corruption_probability_${corruption_probability}_sigma_0.0/" \
#     --data '/blob/v-tianyuchen/datasets/cifar10-32x32.zip' \
#     --arch ddpmpp \
#     --edm_model /blob/v-tianyuchen/Projects/Ambient_A/ambient_diffusion_ckpt/checkpoints/cifar08/network-snapshot-152634.pkl \
#     --metrics fid50k_full \
#     --tick 10 \
#     --snap 50 \
#     --dump 500 \
#     --lr 1e-5 \
#     --glr 1e-5 \
#     --fp16 0 \
#     --ls 1 \
#     --lsg 100 \
#     --duration 100 \
#     --data_stat '/blob/v-tianyuchen/datasets/cifar10-32x32.npz' \
#     --corruption_probability=${corruption_probability} \
#     --delta_probability=0.1 \
#     --max_grad_norm=1.0 \
#     --mask_full_rgb=True \
#     --experiment_name=distill_cifar10_random_${corruption_probability}_sigma_${sigma}



corruption_probability=0.6
torchrun --standalone --nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) sid_train.py \
    --alpha 1.2 \
    --tmax 800 \
    --init_sigma 2.5 \
    --batch 256 \
    --batch-gpu 16 \
    --outdir "/blob/v-tianyuchen/Projects/Ambient_A/cifar10/SiD_ambient_diffusion_random_inpainting_corruption_probability_${corruption_probability}_sigma_0.0/" \
    --data '/blob/v-tianyuchen/datasets/cifar10-32x32.zip' \
    --arch ddpmpp \
    --edm_model /blob/v-tianyuchen/Projects/Ambient_A/ambient_diffusion_ckpt/checkpoints/cifar06/network-snapshot-130458.pkl \
    --metrics fid50k_full \
    --tick 10 \
    --snap 50 \
    --dump 500 \
    --lr 1e-5 \
    --glr 1e-5 \
    --fp16 0 \
    --ls 1 \
    --lsg 100 \
    --duration 100 \
    --data_stat '/blob/v-tianyuchen/datasets/cifar10-32x32.npz' \
    --corruption_probability=${corruption_probability} \
    --delta_probability=0.1 \
    --max_grad_norm=1.0 \
    --mask_full_rgb=True \
    --experiment_name=distill_cifar10_random_${corruption_probability}_sigma_${sigma}

R=2
EXPERIMENT_NAME=brainMRI_prior_R=$R
GPUS_PER_NODE=1
GPU=0
MODEL_PATH="/home/yasmin/projects/tianyu/ambient-diffusion-mri/ambient_models/ambient/R=${R}"
MAPS_PATH=/mnt/shared/yasmin/DATASETS/brain_multicoil_train_batch_0/processed_32dB/mri_numpy_4.zip
SEEDS=16
BATCH=2

CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node=$GPUS_PER_NODE \
    prior.py --gpu=$GPU --network=$MODEL_PATH/  --maps_path=$MAPS_PATH\
    --outdir=results/$EXPERIMENT_NAME \
    --experiment_name=$EXPERIMENT_NAME \
    --ref=$MODEL_PATH/stats.jsonl \
    --seeds=$SEEDS --batch=$BATCH \
    --mask_full_rgb=True --num_masks=1 --guidance_scale=0.0 \
    --training_options_loc=$MODEL_PATH/training_options.json \
    --num=$SEEDS --img_channels=2 --with_wandb=True
# R=2
# EXPERIMENT_NAME=sid_brainMRI_R=$R
# GPUS_PER_NODE=8
# MODEL_PATH="/blob/v-tianyuchen/Projects/Ambient_A/mri/R=2/network-snapshot.pkl"
# DATA_PATH=/blob/v-tianyuchen/Projects/Ambient_A/mri/mri_data_numpy.zip
# OUTDIR=/blob/v-tianyuchen/Projects/Ambient_A/mri/$EXPERIMENT_NAME
# CORR=$R
# DELTA=$((R+1)) # delta = R + 1
# BATCH=24 # batch size
# METHOD=ambient
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# # 5e-6 is 19
# # lr=5e-6, glr=1e-6 is 18 at the second tick. use this one.
# python -m torch.distributed.run --standalone   --nproc_per_node=$GPUS_PER_NODE \
# sid_train.py   --outdir=$OUTDIR --experiment_name=$EXPERIMENT_NAME \
# --dump=50 --cond=0 --arch=ddpmpp \
# --precond=$METHOD --cres=1,1,1,1 --lr=5e-6 --dropout=0.1 --augment=0 \
# --data=$DATA_PATH --norm=2 --max_grad_norm=1.0 --mask_full_rgb=1 \
# --corruption_probability=$CORR --delta_probability=$DELTA --batch=$BATCH \
# --normalize=0 --fp16=1  \
# --edm_model=$MODEL_PATH \
# --batch-gpu=3 \
# --glr=1e-6 \
# --g_beta1=0.0 \
# --alpha=1.2 \
# --tmax 800 \
# --init_sigma 2.5 \
# --tick 10 \
# --snap 50 \
# --lsg 100 \
# --seed 42 \



# 17.3
# R=4
# EXPERIMENT_NAME=sid_brainMRI_R=$R
# GPUS_PER_NODE=8
# MODEL_PATH="/blob/v-tianyuchen/Projects/Ambient_A/mri/R=4/network-snapshot.pkl"
# DATA_PATH=/blob/v-tianyuchen/Projects/Ambient_A/mri/mri_data_numpy.zip
# OUTDIR=/blob/v-tianyuchen/Projects/Ambient_A/mri/$EXPERIMENT_NAME
# CORR=$R
# DELTA=$((R+1)) # delta = R + 1
# BATCH=24 # batch size
# METHOD=ambient
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"


# python -m torch.distributed.run --standalone   --nproc_per_node=$GPUS_PER_NODE \
# sid_train.py   --outdir=$OUTDIR --experiment_name=$EXPERIMENT_NAME \
# --dump=50 --cond=0 --arch=ddpmpp \
# --precond=$METHOD --cres=1,1,1,1 --lr=5e-6 --dropout=0.1 --augment=0 \
# --data=$DATA_PATH --norm=2 --max_grad_norm=1.0 --mask_full_rgb=1 \
# --corruption_probability=$CORR --delta_probability=$DELTA --batch=$BATCH \
# --normalize=0 --fp16=1  \
# --edm_model=$MODEL_PATH \
# --batch-gpu=3 \
# --glr=1e-6 \
# --g_beta1=0.0 \
# --alpha=1.2 \
# --tmax 800 \
# --init_sigma 2.5 \
# --tick 10 \
# --snap 50 \
# --lsg 100 \
# --seed 42 \


# 21.78
# R=6
# EXPERIMENT_NAME=sid_brainMRI_R=$R
# GPUS_PER_NODE=8
# MODEL_PATH="/blob/v-tianyuchen/Projects/Ambient_A/mri/R=6/network-snapshot.pkl"
# DATA_PATH=/blob/v-tianyuchen/Projects/Ambient_A/mri/mri_data_numpy.zip
# OUTDIR=/blob/v-tianyuchen/Projects/Ambient_A/mri/$EXPERIMENT_NAME
# CORR=$R
# DELTA=$((R+1)) # delta = R + 1
# BATCH=24 # batch size
# METHOD=ambient
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"


# python -m torch.distributed.run --standalone   --nproc_per_node=$GPUS_PER_NODE \
# sid_train.py   --outdir=$OUTDIR --experiment_name=$EXPERIMENT_NAME \
# --dump=50 --cond=0 --arch=ddpmpp \
# --precond=$METHOD --cres=1,1,1,1 --lr=5e-6 --dropout=0.1 --augment=0 \
# --data=$DATA_PATH --norm=2 --max_grad_norm=1.0 --mask_full_rgb=1 \
# --corruption_probability=$CORR --delta_probability=$DELTA --batch=$BATCH \
# --normalize=0 --fp16=1  \
# --edm_model=$MODEL_PATH \
# --batch-gpu=3 \
# --glr=1e-6 \
# --g_beta1=0.0 \
# --alpha=1.2 \
# --tmax 800 \
# --init_sigma 2.5 \
# --tick 10 \
# --snap 50 \
# --lsg 100 \
# --seed 42 \


# fid: 22.52
R=8
EXPERIMENT_NAME=sid_brainMRI_R=$R
GPUS_PER_NODE=8
MODEL_PATH="/blob/v-tianyuchen/Projects/Ambient_A/mri/R=8/network-snapshot.pkl"
DATA_PATH=/blob/v-tianyuchen/Projects/Ambient_A/mri/mri_data_numpy.zip
OUTDIR=/blob/v-tianyuchen/Projects/Ambient_A/mri/$EXPERIMENT_NAME
CORR=$R
DELTA=$((R+1)) # delta = R + 1
BATCH=24 # batch size
METHOD=ambient
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"


python -m torch.distributed.run --standalone   --nproc_per_node=$GPUS_PER_NODE \
sid_train.py   --outdir=$OUTDIR --experiment_name=$EXPERIMENT_NAME \
--dump=50 --cond=0 --arch=ddpmpp \
--precond=$METHOD --cres=1,1,1,1 --lr=5e-6 --dropout=0.1 --augment=0 \
--data=$DATA_PATH --norm=2 --max_grad_norm=1.0 --mask_full_rgb=1 \
--corruption_probability=$CORR --delta_probability=$DELTA --batch=$BATCH \
--normalize=0 --fp16=1  \
--edm_model=$MODEL_PATH \
--batch-gpu=3 \
--glr=1e-6 \
--g_beta1=0.0 \
--alpha=1.2 \
--tmax 800 \
--init_sigma 2.5 \
--tick 10 \
--snap 50 \
--lsg 100 \
--seed 42 \
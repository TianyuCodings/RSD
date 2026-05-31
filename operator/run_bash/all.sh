

network="$1"
dataset="$2"
name=$(basename "$network" | cut -d. -f1)
stop_variance=0.04

# if we have $3 in put then we use it as cuda group, group 0 is 0,1,2,3, group 1 is 4,5,6,7
if [ -z "$3" ]
then
    cuda_group=0
else
    cuda_group=$3
fi
# cuda_devices=$(($cuda_group*4+0)),$(($cuda_group*4+1)),$(($cuda_group*4+2)),$(($cuda_group*4+3))

# let cuda_devices=0,1,2,3,4,5,6,7
cuda_devices=0,1,2,3,4,5,6,7

# CUDA_VISIBLE_DEVICES=$cuda_devices   python -m torch.distributed.run --nproc_per_node=4 --standalone generate.py \
#     --seeds=0-49999 \
#     --network="${network}" \
#     --outdir="/data/xuchenheng/${dataset}/${name}/" \
#     --batch=512 \
#     --stop_variance=${stop_variance}

# CUDA_VISIBLE_DEVICES=$cuda_devices   python -m torch.distributed.run --standalone eval_fid.py \
#     --gen_path="/data/xuchenheng/${dataset}/${name}" \
#     --ref_stats=/home/ubuntu/ext-mamba-illinois/yasi/vision_data/${dataset}-64x64.npz \
#     --out_path="/data/xuchenheng/${dataset}/${name}.json"

echo 'truncated sampling done'

name="${name}_full"
 

CUDA_VISIBLE_DEVICES=$cuda_devices   python -m torch.distributed.run --nproc_per_node=8 --standalone generate.py \
    --seeds=0-49999 \
    --network="${network}" \
    --outdir="/data/xuchenheng/${dataset}1/${name}/" \
    --batch=1024 \
    --stop_variance=0.0001

CUDA_VISIBLE_DEVICES=$cuda_devices   python -m torch.distributed.run --standalone eval_fid.py \
    --gen_path="/data/xuchenheng/${dataset}1/${name}" \
    --ref_stats=/home/ubuntu/ext-mamba-illinois/yasi/vision_data/${dataset}-64x64.npz \
    --out_path="/data/xuchenheng/${dataset}1/${name}.json"

echo 'full sampling done'


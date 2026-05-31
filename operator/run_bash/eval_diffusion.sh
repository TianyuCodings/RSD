
# name=generate4

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --standalone eval_fid.py \
    --gen_path="/data/xuchenheng/ffhq_pretrain/${name}/" \
    --ref_stats=/home/ubuntu/ext-mamba-illinois/yasi/ffhq-64x64.npz \
    --out_path="/data/xuchenheng/ffhq_pretrain/${name}.json"


for name in ffhq
do 
    python /home/ubuntu/ext-mamba-illinois/yasi/vision_data/add_noise.py "/home/ubuntu/ext-mamba-illinois/yasi/vision_data/${name}" "/home/ubuntu/ext-mamba-illinois/yasi/vision_data/${name}_noise"  
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --standalone eval_fid.py \
        --gen_path="/home/ubuntu/ext-mamba-illinois/yasi/vision_data/${name}_noise" \
        --ref_stats=/home/ubuntu/ext-mamba-illinois/yasi/vision_data/${name}-64x64.npz \
        --out_path="/data/xuchenheng/observation/${name}_noise.json"
         
done
# name=afhqv2_noise1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --standalone eval_fid.py \
#     --gen_path="/home/ubuntu/ext-mamba-illinois/yasi/vision_data/afhqv2_noise" \
#     --ref_stats=/home/ubuntu/ext-mamba-illinois/yasi/vision_data/afhqv2-64x64.npz \
#     --out_path="/data/xuchenheng/observation/${name}.json"
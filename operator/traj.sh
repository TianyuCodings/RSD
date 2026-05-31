CUDA_VISIBLE_DEVICES=0,1 python fid_traj.py \
    --folder_path='/home/ubuntu/ext-mamba-illinois/yasi/Ambient/celebahq/SiD/00000-celeba_hq-64x64-uncond-ddpmpp-edmglr1e-05-lr1e-05-initsigma2.5-gpus4-alpha1.2-batch512-tmax800-fp16' \
    --image_path="/data/xuchenheng/sid_images/celebahq" \
    --ref_path="/home/ubuntu/ext-mamba-illinois/yasi/vision_data/celeba_hq-64x64.npz" \
    --noisy_ref_path="/home/ubuntu/ext-mamba-illinois/yasi/vision_data/celeba_hq_noisy-64x64.npz"

MASTER_PORT=29501 CUDA_VISIBLE_DEVICES=2,3 python fid_traj.py \
    --folder_path='/home/ubuntu/ext-mamba-illinois/yasi/Ambient/ffhq/SiD/00004-ffhq-64x64-uncond-ddpmpp-edmglr1e-05-lr1e-05-initsigma2.5-gpus4-alpha1.2-batch512-tmax800-fp16' \
    --image_path="/data/xuchenheng/sid_images/ffhq" \
    --ref_path="/home/ubuntu/ext-mamba-illinois/yasi/vision_data/ffhq-64x64.npz" \
    --noisy_ref_path="/home/ubuntu/ext-mamba-illinois/yasi/vision_data/ffhq_noisy-64x64.npz"

MASTER_PORT=29502 CUDA_VISIBLE_DEVICES=4,5 python fid_traj.py \
    --folder_path='/home/ubuntu/ext-mamba-illinois/yasi/Ambient/afhq/SiD/00000-afhqv2-64x64-uncond-ddpmpp-edmglr5e-06-lr5e-06-initsigma2.5-gpus4-alpha1.0-batch512-tmax800-fp16' \
    --image_path="/data/xuchenheng/sid_images/afhqv2" \
    --ref_path="/home/ubuntu/ext-mamba-illinois/yasi/vision_data/afhqv2-64x64.npz" \
    --noisy_ref_path="/home/ubuntu/ext-mamba-illinois/yasi/vision_data/afhqv2_noisy-64x64.npz"
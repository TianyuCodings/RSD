CUDA_VISIBLE_DEVICES=0,1 python fid_new.py ref --data=/home/ubuntu/ext-mamba-illinois/yasi/vision_data/celeba_hq-64x64.zip --dest=/home/ubuntu/ext-mamba-illinois/yasi/vision_data/celeba_hq_noisy-64x64.zip --sigma_level=0.2 
CUDA_VISIBLE_DEVICES=2,3 python fid_new.py ref --data=/home/ubuntu/ext-mamba-illinois/yasi/vision_data/cifar10-32x32.zip --dest=/home/ubuntu/ext-mamba-illinois/yasi/vision_data/cifar10_noisy-32x32.zip --sigma_level=0.2 
CUDA_VISIBLE_DEVICES=4,5 python fid_new.py ref --data=/home/ubuntu/ext-mamba-illinois/yasi/vision_data/afhqv2-64x64.zip --dest=/home/ubuntu/ext-mamba-illinois/yasi/vision_data/afhqv2_noisy-64x64.zip --sigma_level=0.2 
# CUDA_VISIBLE_DEVICES=6,7 python fid_new.py ref --data=/home/ubuntu/ext-mamba-illinois/yasi/vision_data/ffhq-64x64.zip --dest=/home/ubuntu/ext-mamba-illinois/yasi/vision_data/ffhq_noisy-64x64.zip --sigma_level=0.2 &

wait


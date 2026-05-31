
gen_path=$1
out_path=$2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --standalone eval_fid.py --gen_path=${gen_path} --ref_stats=/home/ubuntu/ext-mamba-illinois/yasi/vision_data/cifar10-32x32.npz --out_path=/data/xuchenheng/EMDiffusion/${out_path}.json
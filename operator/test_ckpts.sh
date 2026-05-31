
network=giannisdaras/ambient_laws_celeba_sigma_0.05_corruption_0.1_keep_1.0 if 
outdir=/data/xuchenheng/celebahq/$network
dataset_stats=/home/ubuntu/ext-mamba-illinois/yasi/celeba_hq-64x64.npz

python -m torch.distributed.run --standalone generate.py \
    --seeds=0-49999 \
    --network=$network \
    --outdir=$outdir \
    --batch=64
    

python -m torch.distributed.run --standalone eval_fid.py \
    --gen_path=${outdir} \
    --ref_stats=${dataset_stats}
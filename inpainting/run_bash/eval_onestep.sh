CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --standalone --nproc_per_node=4  \
generate_onestep.py --outdir='/blob/v-tianyuchen/Projects/Ambient_A/celeba/SiD_ambient_diffusion_random_inpainting_corruption_probability_0.8_sigma_0.0/generate_images/' --seeds=0-49999 \
--batch=512  --network="/blob/v-tianyuchen/Projects/Ambient_A/celeba/SiD_ambient_diffusion_random_inpainting_corruption_probability_0.8_sigma_0.0/00015-celeba_hq-64x64-uncond-ddpmpp-edmglr1e-05-lr1e-05-initsigma2.5-gpus8-alpha1.2-batch512-tmax800-fp16/network-snapshot-1.200000-005120.pkl" \


# torchrun --standalone --nproc_per_node=8 fid.py calc --images=/blob/v-tianyuchen/Projects/Ambient_A/celeba/SiD_ambient_diffusion_random_inpainting_corruption_probability_0.8_sigma_0.0/generate_images/ \
#     --ref=/blob/v-tianyuchen/datasets/fid-refs/celeba_hq-64x64.npz
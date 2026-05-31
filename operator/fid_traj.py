# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Script for calculating Frechet Inception Distance (FID)."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import scipy.linalg
import torch
import dnnlib
from torch_utils import distributed as dist
from torch.distributed import barrier, broadcast
from training import dataset
import PIL
import pandas as pd
import argparse

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------

def calculate_inception_stats(
        image_path, num_expected=None, seed=0, max_batch_size=4,
        num_workers=3, prefetch_factor=2, device=torch.device('cuda'),
        sigma_level=0.0
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load Inception-v3 model.
    # This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    dist.print0('Loading Inception-v3 model...')
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048
    with dnnlib.util.open_url(detector_url, verbose=(dist.get_rank() == 0)) as f:
        detector_net = pickle.load(f).to(device)

    # List images.
    dist.print0(f'Loading images from "{image_path}"...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)
    if num_expected is not None and len(dataset_obj) < num_expected:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_expected}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = ((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches, num_workers=num_workers, prefetch_factor=prefetch_factor)

    # Accumulate statistics.
    dist.print0(f'Calculating statistics for {len(dataset_obj)} images...')
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    noisy_mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    noisy_sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    for images, _labels in tqdm.tqdm(data_loader, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        if images.shape[0] == 0:
            continue
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        if sigma_level > 0.0:
            images_dtype = images.dtype
            noisy_images = images / 127.5 -1
            noisy_images += torch.randn_like(noisy_images) * sigma_level
            noisy_images = (noisy_images * 127.5 + 128).clamp(0, 255).to(images_dtype)

        features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
        noisy_features = detector_net(noisy_images.to(device), **detector_kwargs).to(torch.float64)

        mu += features.sum(0)
        sigma += features.T @ features
        noisy_mu += noisy_features.sum(0)
        noisy_sigma += noisy_features.T @ noisy_features


    # Calculate grand totals.
    torch.distributed.all_reduce(mu)
    torch.distributed.all_reduce(sigma)
    mu /= len(dataset_obj)
    sigma -= mu.ger(mu) * len(dataset_obj)
    sigma /= len(dataset_obj) - 1

    torch.distributed.all_reduce(noisy_mu)
    torch.distributed.all_reduce(noisy_sigma)
    noisy_mu /= len(dataset_obj)
    noisy_sigma -= noisy_mu.ger(noisy_mu) * len(dataset_obj)
    noisy_sigma /= len(dataset_obj) - 1

    return mu.cpu().numpy(), sigma.cpu().numpy(), noisy_mu.cpu().numpy(), noisy_sigma.cpu().numpy()

#----------------------------------------------------------------------------

def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))

#
def calc(image_path, ref_path, sigma_level, num_expected, seed, batch):
    """Calculate FID for a given set of images."""
    dist.print0(f'Loading dataset reference statistics from "{ref_path}"...')
    ref = None
    if dist.get_rank() == 0:
        with dnnlib.util.open_url(ref_path) as f:
            ref = dict(np.load(f))

    mu, sigma, noisy_mu, noisy_sigma = calculate_inception_stats(image_path=image_path, num_expected=num_expected, seed=seed, max_batch_size=batch, sigma_level=sigma_level)
    return mu, sigma, noisy_mu, noisy_sigma


#----------------------------------------------------------------------------

def get_pkl_path_from_folder(folder_path):
    absolute_paths = []
    indexes = []

    # Loop through the folder to find all .pkl files
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pkl") and "network-snapshot" in file_name:
            # Get the absolute path
            absolute_path = os.path.join(folder_path, file_name)
            absolute_paths.append(absolute_path)

            # Extract the index from the file name using regex
            match = re.search(r"-(\d{6})\.pkl", file_name)
            if match:
                indexes.append(match.group(1))
    return absolute_paths, indexes


#----------------------------------------------------------------------------
def generator(network_pkl, outdir, subdirs, seeds, max_batch_size=1024, sigma_G=2.5, class_idx=None, device=torch.device('cuda')):
    """Generate random images using SiD".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png and out.npz
    python generate_onestep.py --outdir=out --seeds=0-63 --batch=64 \
        --network=<network_path>

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate_onestep.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=<network_path>
    """
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        sigma = sigma_G*torch.ones([batch_size, 1, 1, 1], device=device)
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        images = net(sigma_G*latents.to(torch.float64), sigma, class_labels).to(torch.float64)

        # Save images.
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)

    # Done.
    torch.distributed.barrier()
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------
def calc_fid_for_all_pkl(folder_path, image_path, ref_path, noisy_ref_path, sigma_level, seeds, batch, num_expected):
    # Get absolute paths and indexes from the folder
    absolute_paths, indexes = get_pkl_path_from_folder(folder_path)
    print(f"There are {len(indexes)} pkl to be evaluated.")
    print(f'Loading dataset reference statistics from "{ref_path}"...')
    with dnnlib.util.open_url(ref_path) as f:
        ref = dict(np.load(f))
    with dnnlib.util.open_url(noisy_ref_path) as f:
        noisy_ref = dict(np.load(f))

    torch.multiprocessing.set_start_method('spawn', force=True)
    dist.init()

    # seed_offsets = [0, 50000, 100000]
    seed_offsets = [0]

    for idx, pkl_path in enumerate(absolute_paths):
        fids, partial_noisy_fids, fully_noisy_fids = [], [], []

        for offset in seed_offsets:
            current_seeds = np.array(seeds) + offset
            # Generate images using the generator
            generator(pkl_path, image_path, 0, current_seeds)
            mu, sigma, noisy_mu, noisy_sigma = calculate_inception_stats(image_path=image_path,
                                                                         num_expected=num_expected,
                                                                         seed=0, max_batch_size=64,
                                                                         sigma_level=sigma_level)

            dist.print0('Calculating FID...')
            if dist.get_rank() == 0:
                fid = calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])
                partial_noisy_fid = calculate_fid_from_inception_stats(mu, sigma, noisy_ref['mu'], noisy_ref['sigma'])
                fully_noisy_fid = calculate_fid_from_inception_stats(noisy_mu, noisy_sigma, noisy_ref['mu'], noisy_ref['sigma'])
                fids.append(fid)
                partial_noisy_fids.append(partial_noisy_fid)
                fully_noisy_fids.append(fully_noisy_fid)

            torch.distributed.barrier()

        if dist.get_rank() == 0:
            # Compute mean and std across different seeds for each pkl
            mean_fid = np.mean(fids)
            std_fid = np.std(fids)
            mean_partial_noisy_fid = np.mean(partial_noisy_fids)
            std_partial_noisy_fid = np.std(partial_noisy_fids)
            mean_fully_noisy_fid = np.mean(fully_noisy_fids)
            std_fully_noisy_fid = np.std(fully_noisy_fids)

            df = pd.DataFrame({
                'index': [indexes[idx]],
                'path': [pkl_path],
                'mean_fid': [mean_fid],
                'std_fid': [std_fid],
                'mean_partial_noisy_fids': [mean_partial_noisy_fid],
                'std_partial_noisy_fids': [std_partial_noisy_fid],
                'mean_fully_noisy_fids': [mean_fully_noisy_fid],
                'std_fully_noisy_fids': [std_fully_noisy_fid]
            })

            # Save individual CSV file for each pkl
            output_path = os.path.join(folder_path, f"fid_traj.csv")
            if not os.path.exists(output_path):
                df.to_csv(output_path, index=False)
            else:
                df.to_csv(output_path, mode='a', header=False, index=False)




#----------------------------------------------------------------------------
def main():


    # folder_path = '/home/ubuntu/ext-mamba-illinois/yasi/Ambient/celebahq/celebahq_pretrain/00002-celeba_hq-64x64-uncond-ddpmpp-edm-gpus8-batch256-fp32-RQBDU'
    # image_path = "/data/xuchenheng/sid_images/celebahq"
    # ref_path = "/home/ubuntu/ext-mamba-illinois/yasi/vision_data/celeba_hq-64x64.npz"
    # noisy_ref_path = "/home/ubuntu/ext-mamba-illinois/yasi/vision_data/celeba_hq_noisy-64x64.npz"
    
    # input a new argument parser
    parser = argparse.ArgumentParser()
    # add a new argument to the parser
    parser.add_argument("--folder_path", type=str, help="Path to the folder containing the pkl files")
    parser.add_argument("--image_path", type=str, help="Path to the folder containing the generated images")
    parser.add_argument("--ref_path", type=str, help="Path to the reference statistics")
    parser.add_argument("--noisy_ref_path", type=str, help="Path to the noisy reference statistics")
    args = parser.parse_args()

    folder_path = args.folder_path
    image_path = args.image_path
    ref_path = args.ref_path
    noisy_ref_path = args.noisy_ref_path

    


    sigma_level = 0.2
    num_expected = 50000
    seeds = list(range(num_expected))
    batch = 10000
    calc_fid_for_all_pkl(folder_path, image_path, ref_path, noisy_ref_path, sigma_level, seeds, batch, num_expected)
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --standalone --nproc_per_node=4 fid_traj.py
#----------------------------------------------------------------------------


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
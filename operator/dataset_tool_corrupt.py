import torch
import torch.utils.data
import os
import io
import zipfile
import pickle
import tqdm
import numpy as np
from torchvision import transforms
from PIL import Image
from training import dataset
from torch_utils import distributed as dist
from forward_operator import get_operator

def process_and_save_images(
        image_path, output_zip, operator, num_expected=None, seed=0, max_batch_size=64,
        num_workers=3, prefetch_factor=2, device=torch.device('cuda')
):
    """
    Processes an image dataset using DDP, applies an operator to each image, and saves the processed images into a zip file.

    Args:
        image_path (str): Path to the input image dataset.
        output_zip (str): Path to the output zip file.
        operator (callable): A function that takes a torch tensor image and applies transformations.
        num_expected (int, optional): Expected number of images.
        seed (int): Random seed for reproducibility.
        max_batch_size (int): Maximum batch size for DataLoader.
        num_workers (int): Number of DataLoader workers.
        prefetch_factor (int): Prefetch factor for DataLoader.
        device (torch.device): Device to use for processing (default: CUDA).
    """

    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # List images.
    dist.print0(f'Loading images from "{image_path}"...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)

    if num_expected is not None and len(dataset_obj) < num_expected:
        raise RuntimeError(f'Found {len(dataset_obj)} images, but expected at least {num_expected}')
    if len(dataset_obj) < 2:
        raise RuntimeError(f'Found {len(dataset_obj)} images, but need at least 2')

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = ((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(
        dataset_obj, batch_sampler=rank_batches, num_workers=num_workers, prefetch_factor=prefetch_factor
    )

    # Initialize zip buffer (only rank 0 will write the zip)
    image_buffers = {}

    dist.print0(f'Processing {len(dataset_obj)} images...')

    for img_idx, (images, image_names) in enumerate(tqdm.tqdm(data_loader, unit='batch', disable=(dist.get_rank() != 0))):
        torch.distributed.barrier()
        if images.shape[0] == 0:
            continue

        # Ensure images are in correct dtype
        assert images.dtype == torch.uint8

        # Normalize images (if required)
        images = images.to(torch.float32) / 127.5 - 1

        # Move to GPU and apply operator
        images = operator(images.to(device, non_blocking=True))

        # Convert back to uint8 and move to CPU
        images = (images * 255).clamp(0, 255).to(torch.uint8).cpu()

        # Save processed images into buffer
        for i in range(images.shape[0]):
            img_pil = transforms.ToPILImage()(images[i])
            img_name = image_names[i]

            img_byte_arr = io.BytesIO()
            img_pil.save(img_byte_arr, format='PNG')
            image_buffers[img_name] = img_byte_arr.getvalue()

    # Gather processed images across all ranks
    gathered_images = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_images, image_buffers)

    # Rank 0 writes to zip file
    if dist.get_rank() == 0:
        with zipfile.ZipFile(output_zip, 'w') as zipf:
            for img_dict in gathered_images:
                if img_dict:
                    for img_name, img_data in img_dict.items():
                        zipf.writestr(img_name, img_data)

        dist.print0(f'Saved processed images to "{output_zip}".')

    return output_zip


if __name__ == "__main__":
    config = {'name': 'gaussian_blur', 'kernel_size': 9, 'intensity': 2.0, 'sigma': 0.0, 'device':"cuda:0"}
    operator = get_operator(**config)
    image_path = "/home/tychen/Dataset/ffhq-64x64.zip"
    output_zip = "/home/tychen/Dataset/ffhq-64x64_gaussian_blur.zip"
    process_and_save_images


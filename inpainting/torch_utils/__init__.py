# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

# empty

import numpy as np
import torch
import wandb
import PIL
import math

def save_images(images, image_path, num_rows=None, num_cols=None, save_wandb=False, down_factor=None, wandb_down_factor=None, 
                captions=None, font_size=40, text_color=(255, 255, 255), draw_horizontal_arrow=False, draw_vertical_arrow=False):
    if num_rows is None and num_cols is None:
        num_rows = int(np.sqrt(images.shape[0]))    
        num_cols = int(np.ceil(images.shape[0] / num_rows))
    elif num_rows is None and num_cols is not None:
        num_rows = int(np.ceil(images.shape[0] / num_cols))
    elif num_rows is not None and num_cols is None:
        num_cols = int(np.ceil(images.shape[0] / num_rows))
    
    if num_rows * num_cols != images.shape[0]:
        num_rows, num_cols = find_closest_factors(images.shape[0])
    
    image_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    image_size = images.shape[-2]
    grid_image = PIL.Image.new('RGB', (num_cols * image_size, num_rows * image_size))
    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            img = PIL.Image.fromarray(image_np[index])
            grid_image.paste(img, (j * image_size, i * image_size))
            if captions is not None:
                draw = PIL.ImageDraw.Draw(grid_image)
                # use LaTeX bold font
                font = PIL.ImageFont.truetype("cmr10.ttf", font_size)
                draw.text((j * image_size, i * image_size), captions[index], text_color, font=font)
    if down_factor is not None:
        grid_image = grid_image.resize((grid_image.size[0] // down_factor, grid_image.size[1] // down_factor))
    
    if draw_horizontal_arrow:
        draw = PIL.ImageDraw.Draw(grid_image)
        # draw it on the top of the image
        draw.line((0, 0, grid_image.size[0], 0), fill=(255, 0, 0), width=5)

    if draw_vertical_arrow:
        draw = PIL.ImageDraw.Draw(grid_image)
        # draw it on the left of the image
        draw.line((0, 0, 0, grid_image.size[1]), fill=(255, 0, 0), width=5)

    grid_image.save(image_path)

    if save_wandb and wandb.run is not None:
        if wandb_down_factor is not None:
            # resize for speed
            grid_image = grid_image.resize((grid_image.size[0] // wandb_down_factor, grid_image.size[1] // wandb_down_factor))
        wandb.log({"images/" + image_path.split("/")[-1]: wandb.Image(grid_image)})


def find_closest_factors(number):
    sqrt_number = int(math.sqrt(number))
    
    n = sqrt_number
    m = number // n
    
    while n * m != number:
        n += 1
        m = number // n

    return m, n
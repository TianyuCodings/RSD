def forward_fs(img, m):
    coil_imgs = img*m
    return bart(1, 'fft -u 3', coil_imgs.transpose(1, 2, 0)).transpose(2, 0, 1)

def normalization_const(s, gt):                   
    # Get normalization constant from undersampled RSS
    gt_maps_cropped = sp.resize(s, [s.shape[0], 384, 320])
    gt_ksp_cropped = forward_fs(gt[None,...], gt_maps_cropped)
    # zero out everything but ACS
    gt_ksp_acs_only = sp.resize(sp.resize(gt_ksp_cropped, (s.shape[0], ACS_size, ACS_size)), gt_ksp_cropped.shape)
    # make RCS img
    ACS_img = sp.rss(sp.ifft(gt_ksp_acs_only, axes =(-2,-1)), axes=(0,))
    norm_const_99 = np.percentile(np.abs(ACS_img), 99)
    
    return norm_const_99

def create_masks(R, delta_R, acs_lines, LENGTH):
    total_lines = LENGTH
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)
    mask = np.zeros((total_lines, total_lines))
    mask[:,center_line_idx] = 1.
    mask[:,random_line_idx] = 1.
    
    random.shuffle(random_line_idx)
    further_mask = mask.copy()
    further_mask[:, random_line_idx[0:delta_R]] = 0.

    mask = sp.resize(mask, [384, 320])
    further_mask = sp.resize(further_mask, [384, 320])
    mask[0:32] = mask[32:64]
    mask[352:384] = mask[32:64]
    further_mask[0:32] = further_mask[32:64]
    further_mask[352:384] = further_mask[32:64]

    return mask, further_mask

    

import sys
import os
os.environ['TOOLBOX_PATH'] = '/home/yasmin/projects/tianyu/ambient-diffusion-mri/bart'
sys.path.append('/home/yasmin/projects/tianyu/ambient-diffusion-mri/bart/python')
import numpy as np
import h5py
import sigpy as sp
import glob
import random
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm_base
def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)

from bart import bart
import torch
import json
from multiprocessing import Pool

verbose = True
center_slice     = 2
ACS_size         = 20
indexes = [i for i in range(2000)]
record_file = '/mnt/shared/yasmin/DATASETS/brain_multicoil_train_batch_0/record1.txt'

snr = "32dB"

if snr == "32dB":
    noise_amp = np.sqrt(0)
elif snr == "22dB":
    noise_amp = np.sqrt(10)
elif snr == "12dB":
    noise_amp = np.sqrt(100)

ksp_files_train = sorted(glob.glob("/mnt/shared/yasmin/DATASETS/brain_multicoil_train_batch_0/multicoil_train/**.h5"))
ksp_files = []
number = 100
save_root = '/mnt/shared/yasmin/DATASETS/brain_multicoil_train_batch_0/processed_' + str(snr) + "/"

for files in ksp_files_train:
    if 'AXT2' in files:
        ksp_files.append(files)

ksp_files = sorted(ksp_files)[0:number]

def task(i):
    try:
        idx = indexes[i]
        slice_idx  = center_slice

        # Load MRI samples and maps
        with h5py.File(ksp_files[idx], 'r') as contents:
            # Get k-space for specific slice
            ksp = np.asarray(contents['kspace'][slice_idx]).transpose(1, 2, 0)
        

        cimg = bart(1, 'fft -iu 3', ksp) # compare to `bart fft -iu 3 ksp cimg`
        noise = sp.resize(cimg, [396, cimg.shape[1], cimg.shape[2]])[0:30,0:30]
        noise_flat = np.reshape(noise, (-1, cimg.shape[2]))
        cimg = sp.resize(cimg, [384, 320, cimg.shape[2]])
        
        cimg_white = bart(1, 'whiten', cimg[:,:,None,:], noise_flat[:,None,None,:]).squeeze()
        cimg_white = cimg_white + (noise_amp / np.sqrt(2))*(np.random.normal(size=cimg_white.shape) + 1j * np.random.normal(size=cimg_white.shape))
        ksp_white = bart(1, 'fft -u 3', cimg_white) # (384, 320, 16)

        choice = 2
        if choice == 1:
            print('ksp_white.shape', ksp_white.shape)
            ref_white = sp.resize(ksp_white, [ksp_white.shape[0], 33,  ksp_white.shape[2]])   
            ksp_white = ksp_white[:,:,None,...]
            ref_white = ref_white[:,:,None,...] 
            cc_mat = bart(1, 'cc', ref_white)
            if (ksp_white.shape[-1] >= 4):
                ksp_white = bart(1, 'ccapply -p 4', ksp_white, cc_mat)
            print('ksp_white.shape', ksp_white.shape) # ksp_white.shape (384, 320, 1, 4)
            s_maps_white = bart(1, 'ecalib -m 1 -c0', ksp_white).squeeze()
            print('s_maps_white.shape', s_maps_white.shape)
        elif choice == 2:
            print('ksp_white.shape', ksp_white.shape)
            s_maps_white = bart(1, 'ecalib -m 1 -c0', ksp_white[:,:,None,:]).squeeze()
            print('s_maps_white.shape', s_maps_white.shape)
        
        gt_img_white_cropped = sp.resize(bart(1, 'pics -S -i 30', ksp_white[:,:,None,:], s_maps_white[:,:,None,:]), [384, 320])

        ksp_white = ksp_white.transpose(2, 0, 1)
        s_maps_white = s_maps_white.transpose(2, 0, 1)  
        cimg_white = cimg_white.transpose(2, 0, 1)  

        norm_const_99_white = normalization_const(s_maps_white, gt_img_white_cropped)
        ksp_white = ksp_white / norm_const_99_white

        choice = 1
        if choice == 1:
            print('ksp_white.shape', ksp_white.shape)
            ksp_white = ksp_white.transpose(1, 2, 0)
            ref_white = sp.resize(ksp_white, [ksp_white.shape[0], 33,  ksp_white.shape[2]])   
            ksp_white = ksp_white[:,:,None,...]
            ref_white = ref_white[:,:,None,...] 
            cc_mat = bart(1, 'cc', ref_white)
            if (ksp_white.shape[-1] >= 4):
                ksp_white = bart(1, 'ccapply -p 4', ksp_white, cc_mat)
            print('ksp_white.shape', ksp_white.shape) # ksp_white.shape (384, 320, 1, 4)
            s_maps_white = bart(1, 'ecalib -m 1 -c0', ksp_white).squeeze().transpose(2, 0, 1)
            ksp_white = ksp_white.squeeze().transpose(2, 0, 1)
            print('s_maps_white.shape', s_maps_white.shape)
        elif choice == 2:
            print('ksp_white.shape', ksp_white.shape)
            s_maps_white = bart(1, 'ecalib -m 1 -c0', ksp_white.transpose(1, 2, 0)[:,:,None,:]).squeeze().transpose(2, 0, 1)
            print('s_maps_white.shape', s_maps_white.shape)

        gt_img_white_cropped = sp.resize(bart(1, 'pics -S -i 30', ksp_white.transpose(1, 2, 0)[:,:,None,:], s_maps_white.transpose(1, 2, 0)[:,:,None,:]), [384, 320])
        cimg_white = bart(1, 'fft -iu 3', ksp_white.transpose(1, 2, 0)).transpose(2, 0, 1) # compare to `bart fft -iu 3 ksp cimg`
        var = np.var(cimg_white[:, 0:30, 0:30])

        

        mask_2, mask_delta_3 = create_masks(R=2, delta_R=54, acs_lines=20, LENGTH=320)
        mask_4, mask_delta_5 = create_masks(R=4, delta_R=16, acs_lines=20, LENGTH=320)
        mask_6, mask_delta_7 = create_masks(R=6, delta_R=8, acs_lines=20, LENGTH=320)
        mask_8, mask_delta_9 = create_masks(R=8, delta_R=5, acs_lines=20, LENGTH=320)
        
        gt = gt_img_white_cropped
        normalised_slices = np.stack((gt.real, gt.imag), axis=0)
        maps = s_maps_white
        if maps.shape[0] < 4:
            maps_zeros = np.zeros((4, 384, 320), dtype=maps.dtype)
            maps_zeros[:maps.shape[0], :, :] = maps
            maps = maps_zeros
        
        full_slices = np.zeros((2, 384, 384), dtype=normalised_slices.dtype)
        full_slices[:, :, 32:352] = normalised_slices


        # print('s_maps_white.shape', s_maps_white.shape)
        # if s_maps_white.shape[0] != 16:
        #     # write in record.txt
        #     with open(record_file, 'a') as f:
        #         f.write(str(idx) + ' ' +  ksp_files[idx] + '\n')

        if verbose:
            print('shapes')
            print('normalised_slices')
            print(normalised_slices.shape, normalised_slices.dtype)
            
            print('full_slices')
            print(full_slices.shape, full_slices.dtype)
            print('ksp_white')
            print(ksp_white.shape, ksp_white.dtype)
            print('s_maps_white')
            print(s_maps_white.shape, s_maps_white.dtype)
            pritn(s_maps_white[0][:5][:5])
            print('cimg_white')
            print(cimg_white.shape, cimg_white.dtype)
            print('mask_2')
            print(mask_2.shape, mask_2.dtype)


        
        cur_path = save_root
        cur_path = os.path.join(cur_path, 'slice_'+str(slice_idx), str(i))
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)
        
        slice_path_gt = os.path.join(cur_path, "gt.npy")
        slice_path_maps = os.path.join(cur_path, "maps.npy")
        
        slice_path_mask_2 = os.path.join(cur_path, "mask_2.npy")
        slice_path_mask_delta_3 = os.path.join(cur_path, "mask_delta_3.npy")
        slice_path_mask_4 = os.path.join(cur_path, "mask_4.npy")
        slice_path_mask_delta_5 = os.path.join(cur_path, "mask_delta_5.npy")
        slice_path_mask_6 = os.path.join(cur_path, "mask_6.npy")
        slice_path_mask_delta_7 = os.path.join(cur_path, "mask_delta_7.npy")
        slice_path_mask_8 = os.path.join(cur_path, "mask_8.npy")
        slice_path_mask_delta_9 = os.path.join(cur_path, "mask_delta_9.npy")
        
        np.save(slice_path_gt, full_slices)
        # relative_path_gt = slice_path_gt.split(save_root)[-1][0:]
        
        np.save(slice_path_maps, maps)
        # relative_path_maps = slice_path_maps.split(save_root)[-1][0:]

        np.save(slice_path_mask_2, mask_2)
        # relative_path_mask_2 = slice_path_mask_2.split(save_root)[-1][0:]
        np.save(slice_path_mask_delta_3, mask_delta_3)
        # relative_path_mask_delta_3 = slice_path_mask_delta_3.split(save_root)[-1][0:]

        np.save(slice_path_mask_4, mask_4)
        # relative_path_mask_4 = slice_path_mask_4.split(save_root)[-1][0:]
        np.save(slice_path_mask_delta_5, mask_delta_5)
        # relative_path_mask_delta_5 = slice_path_mask_delta_5.split(save_root)[-1][0:]

        np.save(slice_path_mask_6, mask_6)
        # relative_path_mask_6 = slice_path_mask_6.split(save_root)[-1][0:]
        np.save(slice_path_mask_delta_7, mask_delta_7)
        # relative_path_mask_delta_7 = slice_path_mask_delta_7.split(save_root)[-1][0:]

        np.save(slice_path_mask_8, mask_8)
        # relative_path_mask_8 = slice_path_mask_8.split(save_root)[-1][0:]
        np.save(slice_path_mask_delta_9, mask_delta_9)
        # relative_path_mask_delta_9 = slice_path_mask_delta_9.split(save_root)[-1][0:]
    except:
        pass

   




    
    

    # torch.save({'gt': torch.tensor(gt_img_white_cropped, dtype=torch.complex64),
    #             'ksp': torch.tensor(ksp_white, dtype=torch.complex64),
    #             's_map': torch.tensor(s_maps_white, dtype=torch.complex64),
    #             'mask_2': torch.tensor(mask_2),
    #             'mask_3': torch.tensor(mask_3),
    #             'mask_4': torch.tensor(mask_4),
    #             'mask_5': torch.tensor(mask_5),
    #             'mask_6': torch.tensor(mask_6),
    #             'mask_7': torch.tensor(mask_7),
    #             'mask_8': torch.tensor(mask_8),
    #             'mask_9': torch.tensor(mask_9),
    #             'mask_10': torch.tensor(mask_10),
    #             'norm_consts_99': norm_const_99_white,},
    #             path + 'sample_' + str(i) + '.pt')
    
    # torch.save({"noise_var_noisy": var},
            #    path + 'noise_var_' + str(i) + '.pt')

n_proc           = 20 # number of cpu cores to use, when possible
with Pool(n_proc) as p:
    for i in tqdm(p.imap(task, range(2000))):
        continue

class_dict = {}
json_output = {"labels": [[k, v] for k, v in class_dict.items()]}
j = json.dumps(json_output, indent=4)
with open(os.path.join(save_root, "dataset.json"), "w") as f:
    print(j, file=f)
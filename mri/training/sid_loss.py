# Copyright (c) 2024, Mingyuan Zhou. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt

import torch
from torch_utils import persistence
import subprocess
import torch.nn.functional as F

def print_gpu_memory_summary():
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
        encoding='utf-8'
    )
    lines = output.strip().split('\n')
    for i, line in enumerate(lines):
        used, total = map(int, line.split(','))
        print(f"GPU {i}: {used / 1024:.2f} GB / {total / 1024:.2f} GB used")

"""Loss functions used in the paper
"Score identity Distillation: Exponentially Fast Distillation of
Pretrained Diffusion Models for One-Step Generation"."""

def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0):
    # from https://github.com/crowsonkb/k-diffusion
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas

#----------------------------------------------------------------------------
@persistence.persistent_class
class SID_EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5,beta_d=19.9, beta_min=0.1, norm=2):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.norm = norm

        self.karras_sigmas = torch.flip(
            get_sigmas_karras(1000, sigma_max=80, sigma_min=0.002,
                              rho=7
                              ),
            dims=[0])

        self.min_step_percent = 0.02
        self.max_step_percent = 0.98
        self.num_train_timesteps = 1000
        self.min_step = int(self.min_step_percent * self.num_train_timesteps)
        self.max_step = int(self.max_step_percent * self.num_train_timesteps)

    def generator_loss(self, true_score, fake_score, images, hat_corruption_matrix, maps=None, labels=None, augment_pipe=None,alpha=1.2,tmax = 800):
        assert images.shape[1] == 2
        sigma_min = 0.002
        sigma_max = 80
        rho = 7.0
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        rnd_t = torch.rand([images.shape[0], 1, 1, 1], device=images.device)*tmax/1000
        sigma = (max_inv_rho + (1-rnd_t) * (min_inv_rho - max_inv_rho)) ** rho
       
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, torch.zeros(images.shape[0], 9).to(images.device))
        n = torch.randn_like(y) * sigma
    
        y_noisy = y + n
        y_noisy_cplx = y_noisy[:,0] + 1j*y_noisy[:,1]
        y_noisy_cplx = y_noisy_cplx[:,None,...]
        
        noisy_image = self.adjoint(self.mri_forward(y_noisy_cplx, maps, hat_corruption_matrix), maps, hat_corruption_matrix)
        noisy_image = torch.cat((noisy_image.real, noisy_image.imag), dim=1)
        
        hat_corruption_matrix_new = torch.ones_like(noisy_image).cuda()
        hat_corruption_matrix_new[:,0,:,:,] = hat_corruption_matrix[:,0]

        

        cat_input = torch.cat([noisy_image, hat_corruption_matrix_new], axis=1)#.clone().detach()

        #del y_noisy, y_noisy_cplx, n,  hat_corruption_matrix_new, noisy_image, images


        y_real = true_score(cat_input, sigma, labels, augment_labels=augment_labels)[:, :y.shape[1]]
        y_fake = fake_score(cat_input, sigma, labels, augment_labels=augment_labels)[:, :y.shape[1]]

        nan_mask_y = torch.isnan(y).flatten(start_dim=1).any(dim=1)
        nan_mask_y_real = torch.isnan(y_real).flatten(start_dim=1).any(dim=1)
        nan_mask_y_fake = torch.isnan(y_fake).flatten(start_dim=1).any(dim=1)
        nan_mask = nan_mask_y | nan_mask_y_real | nan_mask_y_fake

        # TODO: change it to hat_corruption_matrix_new
        corruption_mask = hat_corruption_matrix
        # Check if there are any NaN values present
        if nan_mask.any():
            # Invert the nan_mask to get a mask of samples without NaNs
            non_nan_mask = ~nan_mask
            # Filter out samples with NaNs from y_real and y_fake
            y = y[non_nan_mask]
            y_real = y_real[non_nan_mask]
            y_fake = y_fake[non_nan_mask]
            corruption_mask = corruption_mask[non_nan_mask]
            
        with torch.no_grad():
            weight_factor = (abs(y - y_real).to(torch.float32)).mean(dim=[1, 2, 3], keepdim=True).clip(min=0.00001)
        loss = ((y_real-y_fake)*((y_real-y)-alpha*(y_real-y_fake)))/weight_factor
        return loss

 

   

        # Centered, orthogonal fft in torch >= 1.7
    def fft(self, x):
        x = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
        return x

    # Centered, orthogonal ifft in torch >= 1.7
    def ifft(self, x):
        x = torch.fft.ifft2(x, dim=(-2, -1), norm='ortho')
        return x
    
    def mri_forward(self, image, maps, mask):
        coil_imgs = maps*image
        coil_ksp = self.fft(coil_imgs)
        sampled_ksp = mask*coil_ksp
        return sampled_ksp

    def adjoint(self, ksp, maps, mask):
        sampled_ksp = mask*ksp
        coil_imgs = self.ifft(sampled_ksp)
        img_out = torch.sum(torch.conj(maps)*coil_imgs,dim=1)[:,None,...] #sum over coil dimension
        return img_out
    
    def prepare_data(self, images, hat_corruption_matrix, maps=None, labels=None, augment_pipe=None, crop=True):
        if crop:
            images = images[:,:,:,32:352]
        # rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        # sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        # weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        # n = torch.randn_like(y) * sigma

        y_noisy = y 
        y_noisy_cplx = y_noisy[:,0] + 1j*y_noisy[:,1]
        y_noisy_cplx = y_noisy_cplx[:,None,...]
        
        noisy_image = self.adjoint(self.mri_forward(y_noisy_cplx, maps, hat_corruption_matrix), maps, hat_corruption_matrix)
        noisy_image = torch.cat((noisy_image.real, noisy_image.imag), dim=1)
        
        hat_corruption_matrix_new = torch.ones_like(noisy_image).cuda()
        hat_corruption_matrix_new[:,0,:,:,] = hat_corruption_matrix[:,0]

        cat_input = torch.cat([noisy_image, hat_corruption_matrix_new], axis=1)

        return cat_input


    def mri_loss(self, net, images, corruption_matrix, hat_corruption_matrix, maps=None, labels=None, augment_pipe=None):        
        # images = images[:,:,:,32:352]
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma

        y_noisy = y + n
        y_noisy_cplx = y_noisy[:,0] + 1j*y_noisy[:,1]
        y_noisy_cplx = y_noisy_cplx[:,None,...]
        
        noisy_image = self.adjoint(self.mri_forward(y_noisy_cplx, maps, hat_corruption_matrix), maps, hat_corruption_matrix)
        noisy_image = torch.cat((noisy_image.real, noisy_image.imag), dim=1)
        
        hat_corruption_matrix_new = torch.ones_like(noisy_image).cuda()
        hat_corruption_matrix_new[:,0,:,:,] = hat_corruption_matrix[:,0]

        cat_input = torch.cat([noisy_image, hat_corruption_matrix_new], axis=1).clone().detach()

        del noisy_image, hat_corruption_matrix_new, y_noisy_cplx, n, y_noisy, images,

        D_yn = net(cat_input, sigma, labels, augment_labels=augment_labels)[:, :y.shape[1]]

        D_yn_cplx = D_yn[:,0] + 1j*D_yn[:,1]
        D_yn_cplx = D_yn_cplx[:,None,...]
        masked_D_yn = self.adjoint(self.mri_forward(D_yn_cplx, maps, corruption_matrix), maps, corruption_matrix)
        masked_D_yn = torch.cat((masked_D_yn.real, masked_D_yn.imag), dim=1)
        masked_D_yn_hat = self.adjoint(self.mri_forward(D_yn_cplx, maps, hat_corruption_matrix), maps, hat_corruption_matrix)
        masked_D_yn_hat = torch.cat((masked_D_yn_hat.real, masked_D_yn_hat.imag), dim=1)
        y_cplx = y[:,0] + 1j*y[:,1]
        y_cplx = y_cplx[:,None,...]
        masked_y = self.adjoint(self.mri_forward(y_cplx, maps, corruption_matrix), maps, corruption_matrix)
        masked_y = torch.cat((masked_y.real, masked_y.imag), dim=1)
        masked_y_hat = self.adjoint(self.mri_forward(y_cplx, maps, hat_corruption_matrix), maps, hat_corruption_matrix)
        masked_y_hat = torch.cat((masked_y_hat.real, masked_y_hat.imag), dim=1)
        
        if self.norm == 2:
            train_loss = weight * ((masked_D_yn_hat - masked_y_hat) ** 2)
            val_loss = weight * ((masked_D_yn - masked_y) ** 2)
            test_loss = weight * ((D_yn - y) ** 2)
        elif self.norm == 1:
            # l1 loss
            train_loss = weight * (hat_corruption_matrix * torch.abs(D_yn - y))
            val_loss = weight * (corruption_matrix * torch.abs(D_yn - y))
            test_loss = weight * torch.abs(D_yn - y)
        else:
            # raise exception
            raise ValueError("Wrong norm type. Use 1 or 2.")
        return train_loss, val_loss, test_loss

    def dmd_generator_loss(self, true_score, fake_score, images, hat_corruption_matrix, maps=None, labels=None, augment_pipe=None,alpha=1.2,tmax = 800):
        assert images.shape[1] == 2
        batch_size = images.shape[0]
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)

        with torch.no_grad():
            timesteps = torch.randint(
                self.min_step,
                min(self.max_step+1, self.num_train_timesteps),
                [batch_size, 1, 1, 1],
                device="cpu",
                dtype=torch.long
            )
            noise = torch.randn_like(images)
            timestep_sigma = self.karras_sigmas[timesteps].to(images.device)

            y_noisy = y + timestep_sigma.reshape(-1, 1, 1, 1) * noise
            y_noisy_cplx = y_noisy[:,0] + 1j*y_noisy[:,1]
            y_noisy_cplx = y_noisy_cplx[:,None,...]
            
            noisy_image = self.adjoint(self.mri_forward(y_noisy_cplx, maps, hat_corruption_matrix), maps, hat_corruption_matrix)
            noisy_image = torch.cat((noisy_image.real, noisy_image.imag), dim=1)
            
            hat_corruption_matrix_new = torch.ones_like(noisy_image).cuda()
            hat_corruption_matrix_new[:,0,:,:,] = hat_corruption_matrix[:,0]

            cat_input = torch.cat([noisy_image, hat_corruption_matrix_new], axis=1).clone().detach()

            del y_noisy, y_noisy_cplx,  hat_corruption_matrix_new

            pred_real_image = true_score(cat_input, timestep_sigma, labels, augment_labels=augment_labels)[:, :y.shape[1]]
            pred_fake_image = fake_score(cat_input, timestep_sigma, labels, augment_labels=augment_labels)[:, :y.shape[1]]

            p_real = (images - pred_real_image)
            p_fake = (images - pred_fake_image)
            weight_factor = torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True)
            grad = (p_real - p_fake) / weight_factor
            grad = torch.nan_to_num(grad)
        loss = 0.5 * F.mse_loss(images, (images-grad).detach(), reduction="mean")
        return loss
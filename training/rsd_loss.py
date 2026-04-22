# Copyright (c) 2024, Mingyuan Zhou. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt

import torch
from torch_utils import persistence
import ambient_utils

"""Loss functions used in the paper
"Score identity Distillation: Exponentially Fast Distillation of
Pretrained Diffusion Models for One-Step Generation"."""

#----------------------------------------------------------------------------
@persistence.persistent_class
class RSD_EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5,beta_d=19.9, beta_min=0.1):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.beta_d = beta_d
        self.beta_min = beta_min

    def generator_loss(self, true_score, fake_score, images, labels=None, augment_pipe=None,alpha=1.2,tmax = 800):

        sigma_min = 0.002
        #TODO: change this sigma to 0.2, but not sure whether it is valid.
        #sigma_min=0.2 is bad, having fid about 18
        #sigma_min = 0.2
        sigma_max = 80
        rho = 7.0
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        rnd_t = torch.rand([images.shape[0], 1, 1, 1], device=images.device)*tmax/1000
        sigma = (max_inv_rho + (1-rnd_t) * (min_inv_rho - max_inv_rho)) ** rho
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, torch.zeros(images.shape[0], 9).to(images.device))
        n = torch.randn_like(y) * sigma
        y_real = true_score(y + n, sigma, labels, augment_labels=augment_labels)
        y_fake = fake_score(y + n, sigma, labels, augment_labels=augment_labels)

        nan_mask_y = torch.isnan(y).flatten(start_dim=1).any(dim=1)
        nan_mask_y_real = torch.isnan(y_real).flatten(start_dim=1).any(dim=1)
        nan_mask_y_fake = torch.isnan(y_fake).flatten(start_dim=1).any(dim=1)
        nan_mask = nan_mask_y | nan_mask_y_real | nan_mask_y_fake

        # Check if there are any NaN values present
        if nan_mask.any():
            # Invert the nan_mask to get a mask of samples without NaNs
            non_nan_mask = ~nan_mask
            # Filter out samples with NaNs from y_real and y_fake
            y = y[non_nan_mask]
            y_real = y_real[non_nan_mask]
            y_fake = y_fake[non_nan_mask]

        with torch.no_grad():
            weight_factor = abs(y - y_real).to(torch.float32).mean(dim=[1, 2, 3], keepdim=True).clip(min=0.00001)
        loss = (y_real-y_fake)*( (y_real-y)-alpha*(y_real-y_fake) )/weight_factor
        return loss

    def __call__(self, fake_score, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        y_fake = fake_score(y + n, sigma, labels, augment_labels=augment_labels)
        nan_mask = torch.isnan(y).flatten(start_dim=1).any(dim=1) | torch.isnan(y_fake).flatten(start_dim=1).any(dim=1)
        if nan_mask.any():
            # Invert the nan_mask to get a mask of samples without NaNs
            non_nan_mask = ~nan_mask
            # Filter out samples with NaNs from y_real and y_fake
            y_fake = y_fake[non_nan_mask]
            y = y[non_nan_mask]
            weight=weight[non_nan_mask]
        loss = weight * ((y_fake - y) ** 2)
        return loss


    def ambient_loss(self, net, images, labels=None, current_sigma=0.0, augment_pipe=None):

        #net._set_static_graph()
        current_sigma = current_sigma.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        # sample a sigma in [current_sigma, sigma_T]
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        sigma = torch.clamp(sigma, min=current_sigma)
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)

        # add additional noise to reach the level sigma
        n = torch.randn_like(y) * torch.sqrt(sigma ** 2 - current_sigma ** 2)
        noisy_input = y + n
        x0_pred = net(noisy_input, sigma, labels, augment_labels=augment_labels)
        # make it xtn prediction
        D_yn = ambient_utils.from_x0_pred_to_xnature_pred_ve_to_ve(x0_pred, noisy_input, sigma, current_sigma)

        # loss weight depends on sigma
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        nan_mask = torch.isnan(y).flatten(start_dim=1).any(dim=1) | torch.isnan(D_yn).flatten(start_dim=1).any(dim=1)
        if nan_mask.any():
            # Invert the nan_mask to get a mask of samples without NaNs
            non_nan_mask = ~nan_mask
            # Filter out samples with NaNs from y_real and y_fake
            D_yn = D_yn[non_nan_mask]
            y = y[non_nan_mask]
            weight=weight[non_nan_mask]

        loss = weight * ((D_yn - y) ** 2)
        return loss
# Copyright (c) 2023
# Mingxuan Cai - patch-based 3D-RI-based spatially-varying aberration correction

import torch
import numpy as np
import os, cv2, scipy, time, argparse

import scipy.io as sio
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import opticaltomography.forward_3dri as forward
import opticaltomography.loss as opt_loss
import opticaltomography.opticsutils_3dri as utils

# DEVICE = 'cuda:0'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir',default='.', type=str)
    parser.add_argument('--data_dir',default='Dataset', type=str)
    parser.add_argument('--save_dir',default='Result', type=str)
    parser.add_argument('--num_epochs', default=1200, type=int)
    parser.add_argument('--init_lr', default=1e-2, type=float)
    parser.add_argument('--final_lr', default=1e-3, type=float)
    parser.add_argument('--patch_gap', default=2, type=float)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--tv_weight', default=0.01, type=float)
    parser.add_argument('--l1_weight', default=1e-8, type=float)

    args = parser.parse_args()

    # Load PSF matrix
    result_dir = f'{args.root_dir}/{args.save_dir}/'
    predict_psf_meas_origin = np.load(result_dir+'predict_psf_usaf_100x100.npy')
    predict_psf_meas = predict_psf_meas_origin.reshape((100,100,200,200))

    if args.patch_gap == 1:
        patch_psf = np.zeros((100,100,80,80))
        for i in range(100):
            for j in range(100):
                patch_psf[j,i] = np.roll(predict_psf_meas[i,j], shift=(50-i, 50-j), axis=(1,0))[60:140,60:140]
    elif args.patch_gap == 2:
        patch_psf = np.zeros((50,50,80,80))
        for i in range(50):
            for j in range(50):
                patch_psf[j,i] = np.roll(predict_psf_meas[i*2,j*2], shift=(50-i*2, 50-j*2), axis=(1,0))[60:140,60:140]

    # Load USAF target
    dataset_dir = f'{args.root_dir}/{args.data_dir}/'
    usaf = np.array(plt.imread(dataset_dir+'usaf_s.jpeg'))
    usaf = sk.color.rgb2gray(cv2.resize(usaf, (50, 50)))
    usaf = np.where(usaf > 0.8, 1, 0)
    usaf = np.pad(usaf, ((75, 75), (75, 75)), mode='constant', constant_values=0)

    # Load wide-field measurement
    meas = np.load(result_dir+'meas.npy')

    # Load parameters to GPU
    device = args.device
    print(device)

    recon_obj = torch.zeros((200,200), requires_grad=True, device=device)
    patch_psf_tensor = torch.tensor(patch_psf, device=device)
    usaf_tensor = torch.tensor(usaf, device=device)
    meas_tensor = torch.tensor(meas, device=device)

    # Set optimization parameters
    opt = torch.optim.Adam([recon_obj], lr=args.init_lr)
    # opt_sche = torch.optim.lr_scheduler

    loss_list = []

    for epoch in range(args.num_epochs):

        opt.zero_grad()
        pred_forward = utils.patch_interp_conv22(recon_obj, patch_psf_tensor, device=device)
        
        total = opt_loss.total_variation_loss_2d(pred_forward, args.tv_weight, device=device)
        mseloss = opt_loss.mse_loss(pred_forward, meas_tensor)
        l1loss = opt_loss.L1_loss(pred_forward, args.l1_weight)
        loss = total+mseloss+l1loss
        
        loss_list.append(loss)

        loss.backward()
        opt.step()
        
        """
        if epoch % 100 == 0:
            with torch.no_grad():
                print(f'Reconstruction after iteration {epoch}, loss {loss:.4e}')
        """
        
        with torch.no_grad():
            print(f'Reconstruction after iteration {epoch}, loss {mseloss:.4e}, l1 {l1loss:.4e}, tv {total:.4e}')

    # Save reconstruction result
    save_recon_dir = result_dir + 'patch_bsed/patch_'+str(args.patch_gap)+'_tv'+str(args.tv_weight)+'_l1'+str(args.l1_weight)+'_epoch'+str(args.num_epochs)
    
    if not os.path.exists(save_recon_dir):
        os.mkdir(save_recon_dir)

    recon_cpu = recon_obj.detach().cpu()
    np.save(save_recon_dir + '/recon_cpu.npy', recon_cpu)
    losslist_cpu = [tensor.detach().cpu() for tensor in loss_list]
    np.save(save_recon_dir + '/losslist_cpu.npy',losslist_cpu)
    skio.imsave(save_recon_dir+'recon.jpg', (utils.norm_np(np.array(recon_cpu[50:150,50:150]))*255).astype(np.uint8))

    # Print evaluation
    psnr_value = psnr(np.array(recon_cpu[50:150,50:150]).astype(np.float64), usaf[50:150, 50:150].astype(np.float64))
    print(f'PSNR: {psnr_value:.4e}')
    ssim_value = ssim(np.array(recon_cpu[50:150, 50:150]).astype(np.float64), usaf[50:150, 50:150].astype(np.float64))
    print(f'SSIM: {ssim_value:.4e}')







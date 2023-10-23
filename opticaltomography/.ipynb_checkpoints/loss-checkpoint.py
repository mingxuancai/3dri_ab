import torch
import numpy as np
import torch.nn as nn



def mse_loss(predict, obj, device='cpu'):
    
    y, x = obj.shape
    mse = torch.pow(predict-obj, 2).sum()/(torch.tensor(y)*torch.tensor(x))
    # print(mse.device)
    return mse

def mse_loss_np(predict, obj):
    
    y, x = obj.shape
    mse = np.sum(np.power(predict - obj, 2)) / (y * x)

    return mse

def mse_loss_vec(predict, obj, device='cpu'):
    
    y = obj.shape[0]
    mse = torch.pow(predict-obj, 2).sum()/torch.tensor(y)
    # print(mse.device)
    return mse

def sparsity_loss(obj, weight, device='cpu'):
    loss = nn.L1Loss()
    # print(obj.device)
    zero = torch.zeros(obj.shape).to(device)
    # print(zero.device)
    return loss(obj, zero)*weight

def L1_loss(obj, weight, device='cpu'):
    loss = weight*torch.sum(torch.abs(obj))
    
    return loss

def total_variation_loss(obj, weight, device='cpu'):
    y, x, z = obj.size()
    tv_h = torch.abs(obj[1,:,:] - obj[-1,:,:]).sum()
    tv_w = torch.abs(obj[:,1,:] - obj[:,-1,:]).sum()
    tv_z = torch.abs(obj[:,:,1] - obj[:,:,-1]).sum()
    
    loss = weight*(tv_h+tv_w+tv_z)/(y*x*z)
    
    return loss

def total_variation_loss_2d(obj_in, weight, device='cpu'):
    # Differences between adjacent pixels in the horizontal direction
    tv_h = torch.sum(torch.abs(obj_in[1:, :] - obj_in[:-1, :]))
    
    # Differences between adjacent pixels in the vertical direction
    tv_w = torch.sum(torch.abs(obj_in[:, 1:] - obj_in[:, :-1]))
    
    # Combine the two components and normalize by image size
    loss = torch.mean(weight * (tv_h + tv_w))
    
    return loss



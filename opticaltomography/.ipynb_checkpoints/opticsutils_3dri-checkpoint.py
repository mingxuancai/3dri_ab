"""
opticsutils_3dri.py - Description:
Optical utility functions for optical phase tomography with pytorch. This script heavily references the code from Michael Chen and David Ren.
Created by Mingxuan Cai on Sep 05, 2023
Contact: mingxuan_cai@berkeley.edu
"""
import numpy as np
import torch
import scipy

np_f32 = np.float32
np_c64 = np.complex64
t_f32 = torch.float32
t_c64 = torch.complex64

def propKernel(shape, pixel_size, wavelength, prop_distance, NA=None, RI=1.0, band_limited=True):
    assert len(shape) == 2, "pupil should be two dimensional!"
    fxlin = genGrid(shape[1], 1 / (pixel_size * shape[1]), flag_shift=True)
    fylin = genGrid(shape[0], 1 / (pixel_size * shape[0]), flag_shift=True)
    fxlin = np.tile(fxlin, (shape[0], 1))
    fylin = np.tile(fylin, (1, shape[1]))

    if band_limited:
        assert NA is not None, "need to provide numerical aperture of the system!"
        Pcrop = genPupil(shape, pixel_size, NA, wavelength)
    else:
        Pcrop = 1.0

    prop_kernel = Pcrop * np.exp(1.0j * 2.0 * np.pi * abs(prop_distance) * Pcrop * \
                                  ((RI / wavelength) ** 2 - fxlin ** 2 - fylin ** 2) ** 0.5)
    prop_kernel = np.conjugate(prop_kernel) if prop_distance < 0 else prop_kernel
    return prop_kernel

def genPupil(shape, pixel_size, NA, wavelength):
    assert len(shape) == 2, "pupil should be two dimensional"
    fxlin = genGrid(shape[1], 1 / (pixel_size * shape[1]), flag_shift=True)
    fylin = genGrid(shape[0], 1 / (pixel_size * shape[0]), flag_shift=True)
    
    fxlin = np.tile(fxlin.T, (shape[0], 1))
    fylin = np.tile(fylin, (1, shape[1]))
    
    pupil_radius = NA / wavelength
    pupil = (fxlin**2 + fylin**2 <= pupil_radius**2)
        
    pupil_mask = (fxlin**2 + fylin**2 <= np.max(fxlin)**2)
    pupil *= pupil_mask
    
    return pupil
    
    
def genGrid(size, dx, flag_shift=False):
    """
    This function generates 1D Fourier grid, centered at the middle of the array.
    
    Inputs:
        size       - length of the array
        dx         - pixel size
        
    Optional parameters:
        flag_shift - flag indicating whether the final array is circularly shifted
                     should be False when computing real space coordinates
                     should be True when computing Fourier coordinates
    
    Outputs:
        xlin       - 1D Fourier grid
    """
    xlin = (np.arange(size) - size//2) * dx
    if flag_shift:
        xlin = np.roll(xlin, -1 * size//2)
    xlin = xlin[:,np.newaxis]
    return xlin # return torch.complex64

def norm(img):
    return (img - torch.amin(img)) / (torch.amax(img) - torch.amin(img))

def norm_np(img):
    return (img - np.amin(img)) / (np.amax(img) - np.amin(img))

def patch_matrix(predict_psf_meas, patch_matrix_size=100, patch_psf_size=80, gap=1):
    """
    This function generates the patched PSF matrix
    
    Inputs:
        predict_psf_meas: full psf matrix, (x, y)
        patch_matrix_size: the final psf matrix size: how many patched psfs
        patch_psf_size: the size of psf that can be cropped to
        gap: interpolation size: 1x1 and 2x2
    
    return:
        patch_psf: numpy.array, (y, x)
        patch_psf_index: return the according index of the psf (y, x) in the original image
    """
        
    patch_psf = np.zeros((patch_matrix_size,patch_matrix_size,patch_psf_size,patch_psf_size))
    patch_psf_index = np.zeros((patch_matrix_size,patch_matrix_size, 2))  # (y,x)
    origin_dim = predict_psf_meas.shape[-1]
    crop_dim = (origin_dim - patch_psf_size)//2
    
    for i in range(patch_matrix_size):
        for j in range(patch_matrix_size):
            patch_psf[j, i] = np.roll(predict_psf_meas[i*gap, j*gap], shift=((patch_matrix_size-1)//2 - i, (patch_matrix_size-1)//2 - j), axis=(1,0))[crop_dim:(origin_dim - crop_dim), crop_dim:(origin_dim - crop_dim)]
            patch_psf_index[j, i] =  [crop_dim+(j*gap), crop_dim+(i*gap)]

    return patch_psf, patch_psf_index

def patch_convolution_100(recon_obj, patch_psf, device='cpu'):
    # patch gap = 1
    pred_forward = torch.zeros((200,200), device=device)
    for i in range(100):
        for j in range(100):
            
            pred_forward[(10+i):(90+i),(10+j):(90+j)] = pred_forward[(10+i):(90+i),(10+j):(90+j)] + recon_obj[50+i,50+j]*patch_psf[i,j]
            
    return pred_forward

def patch_convolution_50(recon_obj, patch_psf, device='cpu'):
    # patch gap = 2
    pred_forward = torch.zeros((200,200), device=device)
    for i in range(50):
        for j in range(50):
            for yy in range(2):
                for xx in range(2):
                    pred_forward[(10+yy+i*2):(90+yy+i*2),(10+xx+j*2):(90+xx+j*2)] = pred_forward[(10+yy+i*2):(90+yy+i*2),(10+xx+j*2):(90+xx+j*2)] + recon_obj[50+yy+2*i,50+xx+2*j]*patch_psf[i,j]
            
    return pred_forward

def patch_interp_conv22(recon_obj, patch_psf, device='cpu'):
    """
    This function generates the forward prediction with 2x2 patched psf
    
    Inputs:
        recon_obj: reconstructed object, torch.tensor
        patch_psf: patched psf, torch.sensor
    
    return:
        pred_forward: forward prediction with convolution, torch.sensor
    """
    pred_forward = torch.zeros((200,200),device=device)
    
    for i in range(50):
        for j in range(50):
            if i!= 49 and j!=49:
                for yy in range(2):
                    for xx in range(2):
                        if yy == 0 and xx==0:
                            psf = patch_psf[i,j]
                        elif yy == 1 and xx==0:
                            psf = 0.5 * patch_psf[i,j] + 0.5 * patch_psf[i+1,j]
                        elif yy == 0 and xx==1:
                            psf = 0.5 * patch_psf[i,j] + 0.5 * patch_psf[i,j+1]
                        elif yy == 1 and xx==1:
                            psf = 0.25 * patch_psf[i,j] + 0.25 * patch_psf[i,j+1] + 0.25 * patch_psf[i+1,j] + 0.25 * patch_psf[i+1,j+1]
                        pred_forward[(10+yy+i*2):(90+yy+i*2),(10+xx+j*2):(90+xx+j*2)] = pred_forward[(10+yy+i*2):(90+yy+i*2),(10+xx+j*2):(90+xx+j*2)] + recon_obj[50+yy+2*i,50+xx+2*j]*psf
                        
            elif i!=49 and j==49:
                for xx in range(2):
                    for yy in range(2):
                        if yy == 0 and xx==0:
                            psf = patch_psf[i,j]
                        elif yy == 1 and xx==0:
                            psf = 0.5 * patch_psf[i,j] + 0.5 * patch_psf[i+1,j]
                        elif yy == 0 and xx==1:
                            psf = patch_psf[i,j]
                        elif yy == 1 and xx==1:
                            psf = 0.5 * patch_psf[i,j] + 0.5 * patch_psf[i+1,j]
                        pred_forward[(10+yy+i*2):(90+yy+i*2),(10+xx+j*2):(90+xx+j*2)] = pred_forward[(10+yy+i*2):(90+yy+i*2),(10+xx+j*2):(90+xx+j*2)] + recon_obj[50+yy+2*i,50+xx+2*j]*psf
                        
            elif i==49 and j!=49:
                for yy in range(2):
                    for xx in range(2):
                        if yy == 0 and xx==0:
                            psf = patch_psf[i,j]
                        elif yy == 1 and xx==0:
                            psf = patch_psf[i,j]
                        elif yy == 0 and xx==1:
                            psf = 0.5 * patch_psf[i,j] + 0.5 * patch_psf[i,j+1]
                        elif yy == 1 and xx==1:
                            psf = 0.5 * patch_psf[i,j] + 0.5 * patch_psf[i,j+1]
                        pred_forward[(10+yy+i*2):(90+yy+i*2),(10+xx+j*2):(90+xx+j*2)] = pred_forward[(10+yy+i*2):(90+yy+i*2),(10+xx+j*2):(90+xx+j*2)] + recon_obj[50+yy+2*i,50+xx+2*j]*psf
            
            else:
                for xx in range(2):
                    for yy in range(2):
                        pred_forward[(10+yy+i*2):(90+yy+i*2),(10+xx+j*2):(90+xx+j*2)] = pred_forward[(10+yy+i*2):(90+yy+i*2),(10+xx+j*2):(90+xx+j*2)] + recon_obj[50+yy+2*i,50+xx+2*j]*psf
       
    return pred_forward

def gen_patch_psf_22(patch_psf):
    full_patch = np.zeros((100,100,80,80))
    
    for i in range(50):
        for j in range(50):
            if i!= 49 and j!=49:
                for yy in range(2):
                    for xx in range(2):
                        if yy == 0 and xx==0:
                            psf = patch_psf[i,j]
                        elif yy == 1 and xx==0:
                            psf = 0.5 * patch_psf[i,j] + 0.5 * patch_psf[i+1,j]
                        elif yy == 0 and xx==1:
                            psf = 0.5 * patch_psf[i,j] + 0.5 * patch_psf[i,j+1]
                        elif yy == 1 and xx==1:
                            psf = 0.25 * patch_psf[i,j] + 0.25 * patch_psf[i,j+1] + 0.25 * patch_psf[i+1,j] + 0.25 * patch_psf[i+1,j+1]
                        full_patch[i*2+yy,j*2+xx] = psf
                        
            elif i!=49 and j==49:
                for xx in range(2):
                    for yy in range(2):
                        if yy == 0 and xx==0:
                            psf = patch_psf[i,j]
                        elif yy == 1 and xx==0:
                            psf = 0.5 * patch_psf[i,j] + 0.5 * patch_psf[i+1,j]
                        elif yy == 0 and xx==1:
                            psf = patch_psf[i,j]
                        elif yy == 1 and xx==1:
                            psf = 0.5 * patch_psf[i,j] + 0.5 * patch_psf[i+1,j]
                        full_patch[i*2+yy,j*2+xx] = psf
                        
            elif i==49 and j!=49:
                for yy in range(2):
                    for xx in range(2):
                        if yy == 0 and xx==0:
                            psf = patch_psf[i,j]
                        elif yy == 1 and xx==0:
                            psf = patch_psf[i,j]
                        elif yy == 0 and xx==1:
                            psf = 0.5 * patch_psf[i,j] + 0.5 * patch_psf[i,j+1]
                        elif yy == 1 and xx==1:
                            psf = 0.5 * patch_psf[i,j] + 0.5 * patch_psf[i,j+1]
                        full_patch[i*2+yy,j*2+xx] = psf
            
            else:
                for xx in range(2):
                    for yy in range(2):
                        full_patch[i*2+yy,j*2+xx] = psf
    return full_patch
    

def find_min_index(img):
    min_index = np.argmin(img)
    min_coord = np.unravel_index(min_index, img.shape)
    return min_coord

def compute_psf_matrix_row(fy, fx, num_column):
    return fy * num_column + fx

def PSF_matrix(predict_psf_meas, ymin, ymax, xmin, xmax, shape):
    # gap = 2 not appliable
    dim_y = shape[0]
    dim_x = shape[1]
    psf_matrix = np.zeros((dim_y ** 2, dim_x ** 2))
    psf_matrix_truncate = np.zeros((dim_y ** 2, (ymax - ymin) * (xmax - xmin)))
    
    fy_varying_coord = np.arange(ymin, ymax, 1)
    fx_varying_coord = np.arange(xmin, xmax, 1)
    row_return = []
    
    for i in range(xmax - xmin):
        for ii in range(ymax - ymin):
            x_coord = fx_varying_coord[i]
            y_coord = fy_varying_coord[ii]
            
            row = compute_psf_matrix_row(y_coord, x_coord, dim_x)
            row_return.append(row)
            psf_matrix[:, row] = predict_psf_meas[i*(xmax-xmin)+ii].reshape((-1,))
            
            row_truncate = i * (xmax - xmin) + ii
            psf_matrix_truncate[:,row_truncate] = predict_psf_meas[row_truncate].reshape((-1,))
    
    return psf_matrix, psf_matrix_truncate, row_return


def PSF_matrix_truncate2full(psf_matrix_truncate, row_return, ymin, ymax, xmin, xmax, shape=(200,200)):
    dim_y = shape[0]
    dim_x = shape[1]
    psf_matrix = np.zeros((dim_y ** 2, dim_x ** 2))
    
    for i in range(xmax - xmin):
        for ii in range(ymax - ymin):
            row_truncate = i * (xmax - xmin) + ii
            psf_matrix[:, row_return[row_truncate]] = psf_matrix_truncate[:,row_truncate]
    
    return psf_matrix
    

def SVD_truncate(matrix, num_singular_value):
    U, S, Vt = np.linalg.svd(matrix)
    
    # Only pick up the num_singular_value
    U_truncated = U[:, :num_singular_value]
    S_truncated = np.diag(S[:num_singular_value])
    Vt_truncated = Vt[:num_singular_value, :]
    
    return U_truncated, S_truncated, Vt_truncated

def SVD_sparse(matrix, num_singular_value):
    U, S, Vt = scipy.sparse.linalg.svds(matrix, num_singular_value)
    D = np.diag(S)
    
    return U, D, Vt
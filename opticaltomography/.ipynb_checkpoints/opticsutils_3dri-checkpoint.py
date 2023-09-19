"""
opticsutils_3dri.py - Description:
Optical utility functions for optical phase tomography with pytorch. This script heavily references the code from Michael Chen and David Ren.
Created by Mingxuan Cai on Sep 05, 2023
Contact: mingxuan_cai@berkeley.edu
"""
import numpy as np
import torch

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

def find_min_index(img):
    min_index = np.argmin(img)
    min_coord = np.unravel_index(min_index, img.shape)
    return min_coord

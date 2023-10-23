"""
opticsmodel_3dri.py - Description:
Implement optics algorithms for optical phase tomography using GPU related functions with pytorch. This script heavily references the code from Michael Chen and David Ren.
Created by Mingxuan Cai on Sep 05, 2023
Contact: mingxuan_cai@berkeley.edu
"""
import numpy as np
import torch
import scipy
from opticaltomography.opticsutils_3dri import genGrid, genPupil, propKernel

np_f32 = np.float32
np_c64 = np.complex64
t_f32 = torch.float32
t_c64 = torch.complex64


class Aperture:
    """
    Class for optical aperture (general)
    """
    def __init__(self, shape, pixel_size, na, pad = True, pad_size = None, **kwargs):
        """
        shape:          shape of object (y,x,z)
        pixel_size:     pixel size of the system
        na:             NA of the system
        pad:            boolean variable to pad the reconstruction
        pad_size:       if pad is true, default pad_size is shape//2. Takes a tuple, pad size in dimensions (y, x)
        """
        self.shape = shape
        self.pixel_size = pixel_size
        self.na = na
        self.pad = pad
        if self.pad:
            self.pad_size = pad_size
            if self.pad_size == None:
                self.pad_size = (self.shape[0]//4, self.shape[1]//4)
            self.row_crop = slice(self.pad_size[0], self.shape[0] - self.pad_size[0])
            self.col_crop = slice(self.pad_size[1], self.shape[1] - self.pad_size[1])
        else:
            self.row_crop = slice(0, self.shape[0])
            self.col_crop = slice(0, self.shape[1])
    
    def forward(self):
        pass
    

class Aberration(Aperture):
    """
    Aberration class, here just for diffraction-limit simulation
    """
    def __init__(self, shape, pixel_size, wavelength, na, pad = True, **kwargs):
        super().__init__(shape, pixel_size, na, pad, **kwargs)
        self.pupil = genPupil((self.shape[0], self.shape[1]), self.pixel_size, self.na, wavelength)
        self.wavelength = wavelength
        self.pupil_support = self.pupil.copy()
    
    def forward(self, field, device='cpu'):
        """Apply pupil"""
        field_f = torch.fft.fft2(field)
        # self.test_f = field_f
        field_f_limited = field_f * torch.from_numpy(self.pupil)
        # self.test_f_limited = field_f_limited
        field_r = torch.fft.ifft2(field_f_limited)
        # self.test_real = field_r
        
        return field_r
    

class Defocus(Aperture):
    """
    Defocus class for tomography
    """
    def __init__(self, shape, pixel_size, wavelength, na, RI_measure = 1.0, pad = True, **kwargs):
        """
        Initialization of the class

        RI_measure: refractive index on the detection side (example: oil immersion objectives)
        """
        super().__init__(shape, pixel_size, na, pad, **kwargs)
        fxlin, fylin = self._genFrequencyGrid()
        self.pupil = genPupil(self.shape, self.pixel_size, self.na, wavelength)
        self.pupilstop = (fxlin**2 + fylin**2 <= np.max(fxlin)**2).astype(np_c64)
        self.fzlin = ((self.RI / self.wavelength) ** 2 - fxlin**2 - fylin**2) ** 0.5
        self.prop_kernel_phase = torch.from_numpy(1.0j * 2.0 * np.pi * self.pupil * self.pupilstop * fzlin)
     
    def forward(self, field, propagation_distances):
        """
        defocus with angular spectrum
        """
        field_defocus = self.pupil * torch.fft.fft2(field)
        field_defocus = field_defocus.repeat(1, 1, len(propagation_distances))
        
        for z_idx, propagation_distance in enumerate(propagation_distances):
            propagation_kernel = torch.exp(self.prop_kernel_phase * propagation_distance[z_idx])
            field_defocus[..., z_idx] *= propagation_kernel
        
        field_defocus = torch.fft.ifft2(field_defocus)
        
        return field_defocus[self.row_crop, self.col_crop]
    
    def _genFrequencyGrid(self):
        fxlin = genGrid(self.shape[1], 1.0 / self.pixel_size / self.shape[1], flag_shift=True)
        fylin = genGrid(self.shape[0], 1.0 / self.pixel_size / self.shape[0], flag_shift=True)
        fxlin = np.tile(fxlin.T, (self.shape[0], 1))
        fylin = np.tile(fylin, (1, self.shape[1]))
        
        return fxlin, fylin
        
        

class ScatteringModels:
    """
    Core of the scattering model
    """
    def __init__(self, phase_obj_3d, wavelength, **kwargs):
        """
        Initialization of the class
        
        phase_obj_3d:      object of the class PhaseObject3D
        wavelength:        wavelength of the light
        """
        
        self.shape = phase_obj_3d.shape
        self.RI = phase_obj_3d.RI
        self.wavelength = wavelength
        self.pixel_size = phase_obj_3d.pixel_size
        self.pixel_size_z = phase_obj_3d.pixel_size_z
        fxlin, fylin = self._genFrequencyGrid() # shifted
        self.fzlin = ((self.RI / self.wavelength) ** 2 - fxlin**2 - fylin**2) ** 0.5
        self.pupilstop = (fxlin**2 + fylin**2 <= np.max(fxlin)**2).astype(np_f32)
        self.prop_kernel_phase = torch.from_numpy(1.0j * 2.0 * np.pi * self.pupilstop * self.fzlin)
        
    
    def _genFrequencyGrid(self):
        fxlin = genGrid(self.shape[1], 1.0 / self.pixel_size / self.shape[1], flag_shift=True)
        fylin = genGrid(self.shape[0], 1.0 / self.pixel_size / self.shape[0], flag_shift=True)
        fxlin = np.tile(fxlin.T, (self.shape[0], 1))
        fylin = np.tile(fylin, (1, self.shape[1]))
        
        return fxlin, fylin
    
    def _genRealGrid(self, flag_shift = False):
        xlin = genGrid(self.shape[1], self.pixel_size, flag_shift = flag_shift)
        ylin = genGrid(self.shape[0], self.pixel_size, flag_shift = flag_shift)
        xlin = np.tile(xlin.T, (self.shape[0], 1))
        ylin = np.tile(ylin, (1, self.shape[1]))
        
        return xlin, ylin
    
    def _setIlluminationOnGrid(self, fy_illu, fx_illu, device='cpu'):
        
        fx_source_on_grid = np.round(fx_illu*self.shape[1]/self.pixel_size)*self.pixel_size/self.shape[1]
        fy_source_on_grid = np.round(fy_illu*self.shape[0]/self.pixel_size)*self.pixel_size/self.shape[0]
        
        return fy_source_on_grid, fx_source_on_grid
    
    
    def _genRealGrid_no_ps(self, flag_shift = False):
        
        xlin = genGrid(self.shape[1], 1, flag_shift = flag_shift)
        ylin = genGrid(self.shape[0], 1, flag_shift = flag_shift)
        xlin = np.tile(xlin.T, (self.shape[0], 1))
        ylin = np.tile(ylin, (1, self.shape[1]))
        
        return xlin, ylin
    
    
    def _setIlluminationOnGrid_no_ps(self, fy_illu, fx_illu):
        
        fx_source_on_grid = np.round(fx_illu*self.shape[1])/self.shape[1]
        fy_source_on_grid = np.round(fy_illu*self.shape[0])/self.shape[0]
        
        return fy_source_on_grid, fx_source_on_grid
    
    
    def _genSphericalWave(self, fy_illu, fx_illu, fz_depth, prop_distance, device='cpu'):
        
        xlin, ylin = self._genRealGrid_no_ps() # 200x200
        fy_source, fx_source = self._setIlluminationOnGrid_no_ps(fy_illu, fx_illu) # set source on the 200x200 grid
        
        fz_source = self.RI / self.wavelength 
        
        r_on_grid = self.pixel_size * ((xlin - fx_source*self.shape[1])**2 + (ylin - fy_source*self.shape[0])**2)**0.5

        if fz_depth != 0:
            dz_prop_distance = self.pixel_size_z + (np.ceil(fz_depth) - fz_depth) * self.pixel_size_z # compute the spherical wave at the next slice and plus one slice to avoid small value
        
            r = (r_on_grid**2 + (dz_prop_distance)**2)**0.5
        else:
            r = (r_on_grid**2 + (prop_distance)**2)**0.5
        
        source_xy = 1.0 * torch.exp(torch.from_numpy(1.0j * 2.0 * np.pi / self.wavelength * r))/r
        
        return source_xy, fy_source, fx_source, fz_source
    
    
    def _genSphericalkernel(self, obj_recon, fz_depth, prop_distance):
        
        xlin, ylin = self._genRealGrid_no_ps()
        fy_source, fx_source = self._setIlluminationOnGrid_no_ps(0, 0)
        
        fz_source = self.RI / self.wavelength 
        
        r_on_grid = self.pixel_size * ((xlin - fx_source*self.shape[1])**2 + (ylin - fy_source*self.shape[0])**2)**0.5
        
        if fz_depth != 0:
            dz_prop_distance = self.pixel_size_z + (np.ceil(fz_depth) - fz_depth) * self.pixel_size_z 
            r = (r_on_grid**2 + (dz_prop_distance)**2)**0.5
        else:
            r = (r_on_grid**2 + (prop_distance)**2)**0.5
        
        source_xy_kernel = 1.0 * torch.exp(torch.from_numpy(1.0j * 2.0 * np.pi / self.wavelength * r))/r
        
        source_xy = scipy.signal.convolve2d(obj_recon, source_xy_kernel, mode='same', boundary='symm')
        
        source_xy = torch.from_numpy(source_xy).to(torch.complex64)
        
        return source_xy, source_xy_kernel
        
    
    def _propagationInplace(self, field, propagation_distance, adjoint=False, in_real=True):
        # print(np.max(np.abs(self.prop_kernel_phase)))
        # print(torch.mean(torch.abs(self.prop_kernel_phase)))
        if in_real:
            field = torch.fft.fft2(field)  # why this dont do fftshift
        if adjoint: # to inverse direction
            field *= torch.conj(torch.exp(self.prop_kernel_phase * propagation_distance))
        else:
            field *= torch.exp(self.prop_kernel_phase * propagation_distance)
        if in_real:
            field = torch.fft.ifft2(field)
        
        return field
    
    
class MultiPhaseContrast(ScatteringModels):
    """
    MultiPhaseContrast scattering model. This class also serves as a parent class for all multi-slice scattering methods
    """
    def __init__(self, phase_obj_3d, wavelength, sigma = 1, **kwargs):
        super().__init__(phase_obj_3d, wavelength, **kwargs)
        self.sigma = sigma
        self.slice_separation = phase_obj_3d.slice_separation
        # self.test = []
        self.test_prop = []
        self.prop_kernel_phase = self.prop_kernel_phase
    
    def forward(self, contrast_obj, fy_illu, fx_illu, fz_source):
        
        obj = contrast_obj
        
        Nz = obj.shape[2]
        
        field, _, _, fz_illu = self._genSphericalWave(fy_illu, fx_illu, fz_source, prop_distance=self.pixel_size_z*2) 
        self.test_new = field
        
        transmittance = torch.exp(1.0j * self.sigma * obj)
        
        
        if fz_source != 0:
            Nz -= (np.ceil(fz_source))
            Nz = int(Nz)
        
        for zz in range(Nz):
            if fz_source != 0:
                zz += np.ceil(fz_source)
                zz = int(zz)
            
            field *= transmittance[:,:,zz]
            # print(torch.mean(torch.real(transmittance[:,:,zz])))
            
            if zz < Nz - 1:
                field = self._propagationInplace(field, self.slice_separation[zz])
                # print(torch.mean(torch.abs(field)))
        # self.test.append(field)
            
        back_to_center = self.pixel_size_z * (Nz - 1)/2 # focus at the center
        
        field = self._propagationInplace(field, back_to_center, adjoint = True)
        # self.test_prop.append(field)
        
        return field
    
    
    def forward_model_2d(self, obj_recon, contrast_obj, fz_source):
        
        obj = contrast_obj
        
        Nz = obj.shape[2]
        
        field, kernel = self._genSphericalkernel(obj_recon, fz_source, self.pixel_size_z)
        self.test_2d = field
        
        transmittance = torch.exp(1.0j * self.sigma * obj)
        
        
        if fz_source != 0:
            Nz -= (np.ceil(fz_source))
            Nz = int(Nz)
        
        for zz in range(Nz):
            if fz_source != 0:
                zz += np.ceil(fz_source)
                zz = int(zz)
            
            field *= transmittance[:,:,zz]
            
            if zz < Nz - 1:
                field = self._propagationInplace(field, self.slice_separation[zz])
            
        back_to_center = self.pixel_size_z * (Nz - 1)/2 # focus at the center
        
        field = self._propagationInplace(field, back_to_center, adjoint = True)
        
        return field
        

    
class MultiBorn(MultiPhaseContrast):
    """
    Multislice algorithm that computes scattered field with first Born approximation at every slice
    """
    def __init__(self, phase_obj_3d, wavelength, **kwargs):
        super().__init__(phase_obj_3d, wavelength, **kwargs)
        
        # generate green's function convolution kernel
        fxlin, fylin = self._genFrequencyGrid()
        kernel_mask = (self.RI/self.wavelength)**2 > 1.01*(fxlin * np.conj(fxlin) - fylin * np.conj(fylin))
        self.green_kernel_2d = -0.25j * torch.exp(torch.from_numpy(2.0j * np.pi * self.fzlin * self.pixel_size_z)) / np.pi / self.fzlin
        self.green_kernel_2d *= kernel_mask
        self.green_kernel_2d[torch.isnan(self.green_kernel_2d)==1] = 0.0
    
    def forward(self, V_obj, fy_illu, fx_illu, fz_source):
        
        obj = V_obj
        
        Nz = obj.shape[2]
        
        field, _, _, fz_illu = self._genSphericalWave(fy_illu, fx_illu, fz_source, prop_distance=self.pixel_size_z)
        
        field_layer_in = torch.zeros(obj.shape, dtype=t_c64)
        
        if fz_source != 0:
            Nz -= (np.ceil(fz_source))
            Nz = int(Nz)
        
        for zz in range(Nz):
            if fz_source != 0:
                zz += np.ceil(fz_source)
                zz = int(zz)
                
            field_layer_in[:,:,zz] = field
            field = self._propagationInplace(field, self.slice_separation[zz])
            field_scat = torch.fft.ifft2(torch.fft.fft2(field_layer_in[:,:,zz] * obj[:,:,zz])*self.green_kernel_2d) * self.pixel_size_z
            field += field_scat
        
        back_to_center = self.pixel_size_z * (Nz - 1)/2
        field = self._propagationInplace(field, back_to_center, adjoint = True)
        
        return field
    
    def forward_model_2d(self, obj_recon, V_obj, fz_source):
        
        obj = V_obj
        
        Nz = obj.shape[2]
        
        field, _ = self._genSphericalkernel(obj_recon, fz_source, self.pixel_size_z)
        
        field_layer_in = torch.zeros(obj.shape, dtype=t_c64)
        
        if fz_source != 0:
            Nz -= (np.ceil(fz_source))
            Nz = int(Nz)
        
        for zz in range(Nz):
            if fz_source != 0:
                zz += np.ceil(fz_source)
                zz = int(zz)
                
            field_layer_in[:,:,zz] = field
            field = self._propagationInplace(field, self.slice_separation[zz])
            field_scat = torch.fft.ifft2(torch.fft.fft2(field_layer_in[:,:,zz] * obj[:,:,zz])*self.green_kernel_2d) * self.pixel_size_z
            field += field_scat
        
        back_to_center = self.pixel_size_z * (Nz - 1)/2
        field = self._propagationInplace(field, back_to_center, adjoint = True)
        
        return field
            
            
            
            
        
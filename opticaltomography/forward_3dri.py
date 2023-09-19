"""
forward_3dri.py - Description:
Implement optics algorithms for forward prediction of optical phase tomography using GPU related functions with pytorch. This script heavily references the code from Michael Chen and David Ren.
Created by Mingxuan Cai on Sep 05, 2023
Contact: mingxuan_cai@berkeley.edu
"""
import numpy as np
import torch
from opticaltomography.opticsmodel_3dri import MultiPhaseContrast, MultiBorn
from opticaltomography.opticsmodel_3dri import Aberration

np_f32 = np.float32
np_c64 = np.complex64
t_f32 = torch.float32
t_c64 = torch.complex64


class PhaseObject3D:
    """
    Class created for 3D objects.
    Depending on the scattering model, one of the following quantities will be used:
    - Refractive index (RI)
    - PhaseContrast
    - Scattering potential (V)
    
    shape:              shape of object to be reconstructed in (x, y, z), tuple
    RI_obj:             refractive index of object(Optional) - 1.33
    RI:                 background refractive index (Optional) - 1.33
    voxel_size:         size of each voxel in (y,x,z), tuple
    slice_separation:   how far apart are slices separated, array
    """

    def __init__(self, shape, voxel_size, RI_obj=None, RI=1.0):
        assert len(shape) == 3, "shape should be 3 dimensional"
        self.RI_obj = RI * torch.ones(shape, dtype=t_f32) if RI_obj is None else RI_obj.type(t_f32)
        self.shape = shape
        self.RI = RI
        self.pixel_size = voxel_size[0]
        self.pixel_size_z = voxel_size[2]

        self.slice_separation = self.pixel_size_z * torch.ones((shape[2],), dtype=t_f32)

    def convertRItoPhaseContrast(self):
        self.contrast_obj = self.RI_obj - self.RI
    
    def convertRItoV(self, wavelength):
        k0 = 2.0 * np.pi / wavelength
        self.V_obj = k0**2 * (self.RI**2 - self.RI_obj**2)
        # self.RI_obj = None

class TomographySolver:
    """
    Highest level solver object for tomography problem
    
    phase_obj_3d:        phase_obj_3d object defined from class PhaseObject3D
    fx_illu_list:        illumination coordinate in x, default = [0] (on axis)
    fy_illu_list:        illumination coordinate in y
    fz_illu_list:        illumination coordinate in z
    """
    def __init__(self, phase_obj_3d, fx_illu_list=[0], fy_illu_list=[0], fz_illu_list=[0], device='cpu', **kwargs):
        self.phase_obj_3d = phase_obj_3d
        self.wavelength = kwargs["wavelength"]
        
        # illumination source coordinate
        assert len(fx_illu_list) == len(fy_illu_list), "fx dimension not equal to fy"
        self.fx_illu_list = fx_illu_list
        self.fy_illu_list = fy_illu_list
        self.fz_illu_list = fz_illu_list
        self.number_illum = len(self.fx_illu_list)  # number of inner sources
        
        # Aberration object: multiply with pupil
        self._aberration_obj = Aberration(phase_obj_3d.shape, phase_obj_3d.pixel_size, self.wavelength, kwargs["na"], pad = False)
        
        # Scattering models and algorithms
        self._opticsmodel    = {"MultiPhaseContrast":          MultiPhaseContrast,
                                "MultiBorn":                   MultiBorn,
                                }
        self.scat_model_args = kwargs
        
    def setScatteringMethod(self, model="MultiPhaseContrast"):
        if model == "MultiPhaseContrast":
            if not hasattr(self.phase_obj_3d, 'contrast_obj'):
                self.phase_obj_3d.convertRItoPhaseContrast()
            self._x = self.phase_obj_3d.contrast_obj
        elif model == "MultiBorn":
            if not hasattr(self.phase_obj_3d, 'V_obj'):
                self.phase_obj_3d.convertRItoV(self.wavelength)
            self._x = self.phase_obj_3d.V_obj
        
        # Set scattering model
        self._scattering_obj = self._opticsmodel[model](self.phase_obj_3d, **self.scat_model_args)
        self.scat_model = model
    
    def forwardPredict(self, obj, device='cpu'):
        
        obj = self._x
        
        forward_scattered_predict = torch.zeros((self.number_illum, self.phase_obj_3d.shape[0], self.phase_obj_3d.shape[1]), dtype=t_f32) # store the scattered field
        
        for illu_idx in range(self.number_illum):  # number of emission
            fx_source = self.fx_illu_list[illu_idx]
            fy_source = self.fy_illu_list[illu_idx]
            fz_source_layer = self.fz_illu_list[illu_idx]
            fields = self._forwardMeasure(fy_source, fx_source, fz_source_layer, obj, device)
            # print(torch.mean(torch.abs(fields)))
            # print(torch.max(torch.abs(fields)))
            
            # transform field to intensity
            est_intensity = torch.abs(fields)
            
            # self.abs_field = est_intensity
            intensity = est_intensity * est_intensity
            
            forward_scattered_predict[illu_idx, :, :] = intensity
        
        return forward_scattered_predict, fields
    
    def _forwardMeasure(self, fy_illu, fx_illu, fz_illu_layer, obj, device='cpu'):
        """
        From an inner emitting source, this function computes the exit wave
        fy_source, fy_source, fz_source: source position in y, x, z
        obj: phase object to be solved
        """
        fields = self._scattering_obj.forward(obj, fy_illu, fx_illu, fz_illu_layer)
        # print(torch.abs(fields[100, 100]))
        
        field_pupil = self._aberration_obj.forward(fields)
        # print(torch.abs(field_pupil[100, 100]))
        
        return field_pupil
    
    
            
        
        
        
    
    
        

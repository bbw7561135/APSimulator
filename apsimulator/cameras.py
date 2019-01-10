# -*- coding: utf-8 -*-
# This file is part of the APSimulator API.
# Author: Lars Frogner
import numpy as np
import filters


class Camera:

    def __init__(self, gain=1, quantum_efficiency=1, filter_set=[], seed=None):
        self.set_gain(gain)
        self.set_filter_set(filter_set)
        self.set_quantum_efficiency(quantum_efficiency)
        self.set_seed(seed)
        self.has_signal_field = False
        self.has_captured_signal_field = False

    def set_gain(self, gain):
        self.gain = float(gain)

    def set_filter_set(self, filter_set):
        if isinstance(filter_set, filters.FilterSet):
            self.filter_set = filter_set
        else:
            self.filter_set = filters.FilterSet(*list(filter_set))

    def set_quantum_efficiency(self, quantum_efficiency):
        if isinstance(quantum_efficiency, filters.Filter):
            self.quantum_efficiency = quantum_efficiency
        else:
            self.quantum_efficiency = filters.Filter('quantum_efficiency',
                                                     *self.filter_set.get_full_wavelength_range(),
                                                     transmittances=float(quantum_efficiency))

    def set_pixel_extents(self, pixel_extent_x, pixel_extent_y):
        self.pixel_extent_x = float(pixel_extent_x)
        self.pixel_extent_y = float(pixel_extent_y)
        self.pixel_area = self.pixel_extent_x*self.pixel_extent_y

    def set_seed(self, seed):
        self.seed = None if seed is None else int(seed)
        self.random_generator = np.random.RandomState(seed=self.seed)

    def compute_signal_field(self, image_field, use_memmap=False):
        '''
        The camera signal field is the spatial distribution of photon rates
        detected by the camera in the image plane. The signal field has one
        component for each filter in the filter set, giving the total photon
        rates detected through each filter. It has units of photons/s/m^2.

        It is found by weighing the spectral fluxes with the product of the
        filter transmittance and the camera's quantum efficiency for each
        wavelength. The weighted spectral fluxes are then converted to spectral
        photon rates, which are integrated to yield the total photon rates.
        '''
        self.filter_set_with_quantum_efficiency = self.filter_set.copy_with_added_filter_on_top(self.quantum_efficiency)
        self.signal_field = self.filter_set_with_quantum_efficiency.compute_filtered_image_field(image_field, convert_to_photon_rates=True, use_memmap=use_memmap)
        self.has_signal_field = True

    def compute_captured_signal_field(self, exposure_time, use_memmap=False):
        '''
        The captured signal field has units of photons/pixel.
        '''
        assert self.has_signal_field
        self.captured_signal_field = self.signal_field.multiplied(exposure_time*self.pixel_area, use_memmap=use_memmap)
        self.captured_signal_field.set_values(self.random_generator.poisson(lam=self.captured_signal_field.values), copy=False)
        self.has_captured_signal_field = True

    def get_signal_field(self):
        assert self.has_signal_field
        return self.signal_field

    def get_captured_signal_field(self):
        assert self.has_captured_signal_field
        return self.captured_signal_field

    def get_gain(self):
        return self.gain

    def get_quantum_efficiency(self):
        return self.quantum_efficiency

    def get_filter_set(self):
        return self.filter_set

    def get_pixel_extents(self):
        return self.pixel_extent_x, self.pixel_extent_y

    def get_pixel_area(self):
        return self.pixel_area



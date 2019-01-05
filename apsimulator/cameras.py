# -*- coding: utf-8 -*-
# This file is part of the APSimulator API.
# Author: Lars Frogner
import numpy as np
import filters


class Camera:

    def __init__(self, gain=1, quantum_efficiency=1, filter_set=[]):
        self.set_gain(gain)
        self.set_filter_set(filter_set)
        self.set_quantum_efficiency(quantum_efficiency)

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
                                                     transmittance=float(quantum_efficiency))

    def compute_filtered_image_field(self, image_field):
        return self.filter_set.copy_with_filter_on_top(self.quantum_efficiency).compute_filtered_image_field(image_field)

    def get_gain(self):
        return self.gain

    def get_quantum_efficiency(self):
        return self.quantum_efficiency

    def get_filter_set(self):
        return self.filter_set



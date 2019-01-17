# -*- coding: utf-8 -*-
# This file is part of the APSimulator API.
# Author: Lars Frogner
import os
import parallel_utils
import procedural_noise


class ProceduralNoisePattern:

    def __init__(self, number_of_octaves=1,
                       initial_frequency=0,
                       frequency_scale=1,
                       spectral_index=1,
                       exponential_factor=1,
                       invert_octaves=False,
                       seed=None):

        self.set_number_of_octaves(number_of_octaves)
        self.set_initial_frequency(initial_frequency)
        self.set_frequency_scale(frequency_scale)
        self.set_spectral_index(spectral_index)
        self.set_exponential_factor(exponential_factor)
        self.set_invert_octaves(invert_octaves)
        self.set_seed(seed)

    def set_number_of_octaves(self, number_of_octaves):
        self.number_of_octaves = int(number_of_octaves)
        assert self.number_of_octaves > 0

    def set_initial_frequency(self, initial_frequency):
        self.initial_frequency = float(initial_frequency)
        assert self.initial_frequency >= 0

    def set_frequency_scale(self, frequency_scale):
        self.frequency_scale = float(frequency_scale)
        assert self.frequency_scale > 0

    def set_spectral_index(self, spectral_index):
        self.spectral_index = float(spectral_index)
        assert self.spectral_index > 0

    def set_exponential_factor(self, exponential_factor):
        self.exponential_factor = float(exponential_factor)
        assert self.exponential_factor != 0

    def set_invert_octaves(self, invert_octaves):
        self.invert_octaves = bool(invert_octaves)

    def set_seed(self, seed):
        if seed is None:
            # Get random integer from operating system
            self.seed = int.from_bytes(os.urandom(4), byteorder='big')
        else:
            self.seed = int(seed)

    def compute(self, x_coordinates, y_coordinates, z_coordinates):

        assert isinstance(x_coordinates, np.ndarray)
        assert isinstance(y_coordinates, np.ndarray)
        assert isinstance(z_coordinates, np.ndarray)

        pattern_values = np.empty((z_coordinates.size, y_coordinates.size, x_coordinates.size), dtype='float64')

        procedural_noise.generate_noise_pattern(pattern_values,
                                                x_coordinates.astype('float64'),
                                                y_coordinates.astype('float64'),
                                                z_coordinates.astype('float64'),
                                                self.number_of_octaves,
                                                self.initial_frequency,
                                                self.frequency_scale,
                                                self.spectral_index,
                                                self.exponential_factor,
                                                int(self.invert_octaves),
                                                self.seed,
                                                parallel_utils.get_number_of_threads())

        return pattern_values

# -*- coding: utf-8 -*-
# This file is part of the APSimulator API.
# Author: Lars Frogner
import numpy as np
import os
import parallel_utils
import procedural_noise
import math_utils


class FractalNoisePattern:

    def __init__(self, number_of_octaves=1,
                       initial_frequency=1,
                       persistence=1,
                       seed=None):

        self.set_number_of_octaves(number_of_octaves)
        self.set_initial_frequency(initial_frequency)
        self.set_persistence(persistence)
        self.set_seed(seed)

    def set_number_of_octaves(self, number_of_octaves):
        self.number_of_octaves = int(number_of_octaves)
        assert self.number_of_octaves > 0

    def set_initial_frequency(self, initial_frequency):
        self.initial_frequency = float(initial_frequency)
        assert self.initial_frequency >= 0

    def set_persistence(self, persistence):
        self.persistence = float(persistence)
        assert self.persistence > 0

    def set_seed(self, seed):
        if seed is None:
            # Get random integer from operating system
            self.seed = int.from_bytes(os.urandom(2), byteorder='big')
        else:
            self.seed = int(seed)

    def compute(self, x_coordinates, y_coordinates, z_coordinates, smallest_scale=None, largest_scale=None, mask=None):

        assert isinstance(x_coordinates, np.ndarray)
        assert isinstance(y_coordinates, np.ndarray)
        assert isinstance(z_coordinates, np.ndarray)
        assert x_coordinates.dtype == np.dtype('float32')
        assert y_coordinates.dtype == np.dtype('float32')
        assert z_coordinates.dtype == np.dtype('float32')

        if x_coordinates.ndim == 1 and y_coordinates.ndim == 1 and z_coordinates.ndim == 1:
            x_coordinates, y_coordinates, z_coordinates = np.meshgrid(x_coordinates, y_coordinates, z_coordinates, indexing='ij')
        else:
            assert x_coordinates.ndim == 3 and y_coordinates.ndim == 3 and z_coordinates.ndim == 3
            assert x_coordinates.shape == y_coordinates.shape
            assert x_coordinates.shape == z_coordinates.shape

        noise_values = np.zeros(x_coordinates.shape, dtype='float32')

        if mask is None:
            mask = np.ones(noise_values.shape, dtype='uint8')
        else:
            assert mask.shape == noise_values.shape
            mask = mask.astype('uint8')

        if largest_scale is not None and smallest_scale is not None:
            # Limit the number of octaves to avoid wasteful noise generation at subgrid scales
            number_of_octaves = min(self.number_of_octaves, self.compute_max_number_of_octaves(smallest_scale, largest_scale))
        else:
            number_of_octaves = self.number_of_octaves

        procedural_noise.generate_fractal_noise(noise_values,
                                                x_coordinates,
                                                y_coordinates,
                                                z_coordinates,
                                                mask,
                                                number_of_octaves,
                                                self.initial_frequency,
                                                self.persistence,
                                                self.seed,
                                                parallel_utils.get_number_of_threads())

        return noise_values

    def compute_smallest_noise_extent(self, largest_grid_extent):
        return 1/(largest_grid_extent*self.initial_frequency*2**(self.number_of_octaves - 1))

    def compute_max_number_of_octaves(self, smallest_grid_cell_extent, largest_grid_extent):
        return 1 + math_utils.nearest_lower_power_of_2_exponent(1/(smallest_grid_cell_extent*largest_grid_extent*self.initial_frequency))

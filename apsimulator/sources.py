# -*- coding: utf-8 -*-
import numpy as np
import fields
import field_combination
import math_utils


class StarField(field_combination.FieldGenerator):

    def __init__(self, star_density, seed=None):
        self.star_density = float(star_density) # Average number of stars per solid angle
        self.random_generator = np.random.RandomState(seed=seed)

    def generate_star_field(self):

        field_of_view_x = math_utils.polar_angle_from_direction_vector_extent(self.grid.extent_x)
        field_of_view_y = math_utils.polar_angle_from_direction_vector_extent(self.grid.extent_y)
        field_of_view_area = field_of_view_x*field_of_view_y
        n_stars = int(self.star_density*field_of_view_area)

        total_window_size = self.grid.compute_total_window_size()
        star_indices = self.random_generator.randint(low=0, high=total_window_size, size=n_stars)
        star_field = np.zeros((self.n_wavelengths, total_window_size), dtype='float64')
        star_field[:, star_indices] = 1

        return star_field.reshape((self.n_wavelengths, *self.grid.window.shape))

    def apply(self, combined_field):
        combined_field.add_within_window(self.generate_star_field())

    def generate_field(self, use_memmap=False):

        self.generated_field = fields.SpectralField(self.grid, self.wavelengths,
                                                    initial_value=0,
                                                    dtype='float64',
                                                    use_memmap=use_memmap)

        self.generated_field.set_values_inside_window(self.generate_star_field())

    def apply_generated_field(self, combined_field):
        combined_field.add_within_window(self.generated_field.values)

# -*- coding: utf-8 -*-
import numpy as np
import fields
import field_combination


class CircularAperture(field_combination.FieldGenerator):

    def __init__(self, diameter, inner_diameter=0, spider_wane_width=0):
        self.set_diameter(diameter)
        self.set_inner_diameter(inner_diameter)
        self.set_spider_wane_width(spider_wane_width)

    def set_diameter(self, diameter):
        self.diameter = float(diameter)
        self.squared_radius = (self.diameter/2)**2

    def set_inner_diameter(self, inner_diameter):
        self.inner_diameter = float(inner_diameter)
        self.squared_inner_radius = (self.inner_diameter/2)**2

    def set_spider_wane_width(self, spider_wane_width):
        self.spider_wane_width = float(spider_wane_width)
        self.spider_wane_halfwidth = self.spider_wane_width/2

    def compute_blocking_mask(self):

        normalized_x_coordinates, normalized_y_coordinates = self.grid.get_coordinate_meshes_within_window()
        squared_normalized_distances = self.grid.compute_squared_distances_within_window()

        abs_x_coordinates = np.multiply.outer(self.wavelengths, np.abs(normalized_x_coordinates))
        abs_y_coordinates = np.multiply.outer(self.wavelengths, np.abs(normalized_y_coordinates))
        squared_distances = np.multiply.outer(self.wavelengths**2, squared_normalized_distances)

        blocking_mask = np.logical_and.reduce((squared_distances <= self.squared_radius,
                                               squared_distances >= self.squared_inner_radius,
                                               abs_x_coordinates >= self.spider_wane_halfwidth,
                                               abs_y_coordinates >= self.spider_wane_halfwidth))

        return blocking_mask

    def apply(self, combined_field):
        combined_field.multiply_within_window(self.compute_blocking_mask())

    def generate_field(self, use_memmap=False):

        self.generated_field = fields.SpectralField(self.grid, self.wavelengths,
                                                    initial_value=False,
                                                    dtype='bool',
                                                    use_memmap=use_memmap)

        self.generated_field.set_values_inside_window(self.compute_blocking_mask())

    def apply_generated_field(self, combined_field):
        combined_field.multiply_within_window(self.generated_field.values)

# -*- coding: utf-8 -*-
import numpy as np
import fields
import math_utils


class FraunhoferImager:

    def __init__(self, aperture_diameter, focal_length):
        self.set_aperture_diameter(aperture_diameter)
        self.set_focal_length(focal_length)

    def set_aperture_diameter(self, aperture_diameter):
        self.aperture_diameter = float(aperture_diameter)

    def set_focal_length(self, focal_length):
        self.focal_length = float(focal_length)

    def get_aperture_diameter(self):
        return self.aperture_diameter

    def get_focal_length(self):
        return self.focal_length

    def initialize_image_field(self, aperture_grid, wavelengths, field_of_view_x, field_of_view_y, use_memmap=False):

        self.wavelengths = wavelengths
        self.image_grid = aperture_grid.to_spatial_frequency_grid().scaled(self.focal_length, grid_type='image')

        # Define a window in the grid covering the specified field of view.
        half_field_of_view_extent_x = np.sin(field_of_view_x/2)*self.focal_length
        half_field_of_view_extent_y = np.sin(field_of_view_y/2)*self.focal_length
        self.image_grid.define_window((-half_field_of_view_extent_x, half_field_of_view_extent_x),
                                      (-half_field_of_view_extent_y, half_field_of_view_extent_y))

        self.flux_scales = (wavelengths*(aperture_grid.compute_cell_area()/self.focal_length))**2

        self.image_field = fields.SpectralField(self.image_grid, self.wavelengths,
                                                initial_value=0,
                                                dtype='float64',
                                                use_memmap=use_memmap)

    def compute_image_field(self, modulated_aperture_field):

        # Compute the Fourier transform of the modulated aperture field
        fourier_coefficients = modulated_aperture_field.fourier_transformed_values()

        # Construct complex field of Fourier coefficients on the image grid
        fourier_transformed_field = fields.SpectralField(self.image_grid, self.wavelengths,
                                                         initial_value=fourier_coefficients,
                                                         dtype='complex128',
                                                         use_memmap=False,
                                                         copy_initial_array=False)

        # Convert Fourier coefficients within the field of view window to image fluxes
        fourier_coefficients_inside_window = fourier_transformed_field.get_values_inside_window()
        spectral_fluxes_inside_window = self.flux_scales[:, np.newaxis, np.newaxis]*math_utils.abs2(fourier_coefficients_inside_window)

        # Assign new fluxes within the field of view window of the image field
        self.image_field.set_values_inside_window(spectral_fluxes_inside_window)

    def visualize(self, **plot_kwargs):
        fields.visualize_field(self.image_field, **plot_kwargs)

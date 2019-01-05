# -*- coding: utf-8 -*-
# This file is part of the APSimulator API.
# Author: Lars Frogner
import numpy as np
import fields
import math_utils


class FraunhoferImager:
    '''
    This class forms an image from an aperture field by computing the Fraunhofer diffracted
    field. This approach assumes that the image plane coincides with the focal plance, and will
    thus produce a perfectly focused image. Out-of-focus effects, i.e. when the image plane and
    focal plane differ, require Fresnel diffraction.
    '''
    def __init__(self, aperture_diameter=1, focal_length=1):
        self.set_aperture_diameter(aperture_diameter) # The diameter of the aperture [m]
        self.set_focal_length(focal_length) # The distance from the aperture plane to the focal (and image) plane [m]
        self.has_image_field = False

    def set_aperture_diameter(self, aperture_diameter):
        self.aperture_diameter = float(aperture_diameter)

    def set_focal_length(self, focal_length):
        self.focal_length = float(focal_length)

    def initialize_image_field(self, aperture_grid, wavelengths, field_of_view_x, field_of_view_y, use_memmap=False):
        '''
        Constructs the grid for the image field and initializes the field.

        The image field is the spatial and spectral distribution of energy flux in the image
        plane. The field values have units of W/m^2/m.

        The image field grid is defined in terms of the spatial x- and y-coordinates with
        repect to the optical axis in the image plane, divided by the focal length. The
        coordinates are found by computing the spatial frequencies of the aperture grid.
        '''
        self.wavelengths = wavelengths
        self.image_grid = aperture_grid.to_spatial_frequency_grid(grid_type='image')

        # Define a window in the grid covering the specified field of view.
        half_field_of_view_extent_x = np.sin(field_of_view_x/2)
        half_field_of_view_extent_y = np.sin(field_of_view_y/2)
        self.image_grid.define_window((-half_field_of_view_extent_x, half_field_of_view_extent_x),
                                      (-half_field_of_view_extent_y, half_field_of_view_extent_y))

        # Compute the scale factors required to obtain image fluxes from the squared Fourier coefficients
        # of the aperture field.
        self.flux_scales = (wavelengths*(aperture_grid.get_cell_area()/self.focal_length))**2

        self.image_field = fields.SpectralField(self.image_grid, self.wavelengths,
                                                initial_value=0,
                                                dtype='float64',
                                                use_memmap=use_memmap)

        self.has_image_field = True

    def compute_image_field(self, modulated_aperture_field):
        '''
        The Fraunhofed diffracted image field is found by taking the Fourier transform of the
        aperture field, squaring and multiplying with the appropriate scale factors.
        '''
        assert self.has_image_field

        # Compute the Fourier transform of the modulated aperture field
        fourier_coefficients = modulated_aperture_field.compute_fourier_transformed_values()

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

    def compute_spectral_powers_of_image_field(self, image_field):
        return np.sum(image_field.get_values_inside_window(), axis=(1, 2))*self.focal_length**2*self.image_grid.get_cell_area()

    def compute_spectral_powers(self):
        assert self.has_image_field
        return self.compute_spectral_powers_of_image_field(self.image_field)

    def get_image_field(self):
        assert self.has_image_field
        return self.image_field

    def get_aperture_diameter(self):
        return self.aperture_diameter

    def get_focal_length(self):
        return self.focal_length

# -*- coding: utf-8 -*-
import numpy as np
import math_utils
import grids
import fields
import field_combination


class ImagingSystem:

    def __init__(self, field_of_view_x, field_of_view_y, angular_coarseness, wavelengths):
        self.field_of_view_x = float(field_of_view_x)
        self.field_of_view_y = float(field_of_view_y)
        self.angular_coarseness = float(angular_coarseness)

        self.wavelengths = np.asfarray(wavelengths)
        self.n_wavelengths = len(self.wavelengths)
        assert(self.wavelengths.ndim == 1)

        self.source_field_combiner = field_combination.FieldCombiner(initial_field_value=0, dtype='float64')
        self.aperture_modulation_combiner = field_combination.FieldCombiner(initial_field_value=1, dtype='complex128')

        self.initialize_fields()

        self.has_imager = False
        self.has_aperture = False

    def initialize_fields(self):
        self.initialize_source_field()
        self.initialize_aperture_field()

    def set_imager(self, imager):
        self.imager = imager
        self.has_imager = True
        self.initialize_aperture_grid_window()
        self.initialize_image_field()

    def set_aperture(self, aperture):
        self.add_aperture_modulator('aperture', aperture)
        self.has_aperture = True

    def initialize_source_field(self, use_memmap=False):
        '''
        Constructs the grid for the source field and initializes the field.

        The source field grid is defined in terms of the x- and y-components of
        the unit direction vector from the center of the aperture to the source.
        '''

        # Construct square grid with a power of 2 size, covering the field of view and with
        # the given angular coarseness
        max_field_of_view = max(self.field_of_view_x, self.field_of_view_y)
        min_source_grid_extent = 2*math_utils.direction_vector_extent_from_polar_angle(max_field_of_view/2)
        grid_cell_extent = math_utils.direction_vector_extent_from_polar_angle(self.angular_coarseness)
        min_source_grid_size = min_source_grid_extent/grid_cell_extent
        source_grid_size_exponent = math_utils.nearest_higher_power_of_2_exponent(min_source_grid_size)
        source_grid_extent = grid_cell_extent*2**source_grid_size_exponent
        self.source_grid = grids.FFTGrid(source_grid_size_exponent, source_grid_size_exponent,
                                         source_grid_extent, source_grid_extent,
                                         is_centered=True, grid_type='source')

        # Define a window in the grid covering the specified field of view.
        # This can be utilized when we only want to operate on the part of the field that will be visible.
        half_field_of_view_extent_x = math_utils.direction_vector_extent_from_polar_angle(self.field_of_view_x/2)
        half_field_of_view_extent_y = math_utils.direction_vector_extent_from_polar_angle(self.field_of_view_y/2)
        self.source_grid.define_window((-half_field_of_view_extent_x, half_field_of_view_extent_x),
                                       (-half_field_of_view_extent_y, half_field_of_view_extent_y))

        # Initialize the object holding the sources
        self.source_field_combiner.initialize_combined_field(self.source_grid, self.wavelengths,
                                                             use_memmap=use_memmap)

    def initialize_aperture_field(self, use_memmap=False):
        '''
        Constructs the grid for the aperture field and initializes the field.

        The aperture field grid is defined in terms of the spatial x- and y-coordinates
        with respect to the aperture center in the plane of the aperture. However, the
        grid uses the spatial coordinates divided by the wavelength ("normalized coordinates"),
        which results in a one-to-one correspondence with the angular grids. In fact, the normalized
        coordinates are simply the spatial frequencies of the angular source grid coordinates.
        '''
        self.aperture_grid = self.source_grid.to_spatial_frequency_grid(grid_type='aperture')

        self.aperture_field = fields.SpectralField(self.aperture_grid, self.wavelengths,
                                                   initial_value=0,
                                                   dtype='complex128',
                                                   use_memmap=use_memmap)

        # Initialize the object holding the aperture field modulations
        self.aperture_modulation_combiner.initialize_combined_field(self.aperture_grid, self.wavelengths,
                                                                    use_memmap=use_memmap)

    def initialize_aperture_grid_window(self):
        assert self.has_imager
        # Define a window in the grid covering the maximum aperture diameter.
        max_normalized_aperture_radius = self.imager.get_aperture_diameter()/(2*np.min(self.wavelengths))
        self.aperture_grid.define_window((-max_normalized_aperture_radius, max_normalized_aperture_radius),
                                         (-max_normalized_aperture_radius, max_normalized_aperture_radius))

    def initialize_image_field(self, use_memmap=False):
        assert self.has_imager
        self.imager.initialize_image_field(self.aperture_grid, self.wavelengths,
                                           self.field_of_view_x, self.field_of_view_y,
                                           use_memmap=use_memmap)

    def add_source(self, name, source, keep_field=False):
        self.source_field_combiner.add(name, source, keep_field=keep_field)

    def add_aperture_modulator(self, name, aperture_modulator, keep_field=False):
        self.aperture_modulation_combiner.add(name, aperture_modulator, keep_field=keep_field)

    def run_full_propagation(self):
        self.compute_source_field()
        self.compute_aperture_field()
        self.compute_aperture_modulation()
        self.compute_image_field()

    def compute_source_field(self):
        self.source_field_combiner.compute_combined_field()

    def compute_aperture_field(self):
        '''
        Computes the aperture field by taking the Fourier transform of the source field.
        This corresponds to summing up the plane waves incident on the aperture plane
        from each direction in the source grid. The plane waves are assumed to have no
        initial phase difference, and the high-frequency temporal phase shifts due to the
        oscillations of the electromagnetic field are neglected (since these will average
        out over a very short time).
        '''
        combined_source_field = self.source_field_combiner.get_combined_field()
        self.aperture_field.set_values_inside_window(combined_source_field.fourier_transformed_values())

    def compute_aperture_modulation(self):
        self.aperture_modulation_combiner.compute_combined_field()

    def compute_modulated_aperture_field(self):
        combined_modulation_field = self.aperture_modulation_combiner.get_combined_field()
        modulated_aperture_field = self.aperture_field.multiplied_within_window(combined_modulation_field.values)
        return modulated_aperture_field

    def compute_image_field(self):
        assert self.has_imager
        self.imager.compute_image_field(self.compute_modulated_aperture_field())

    def get_source(self, name):
        return self.source_field_combiner.get(name)

    def get_aperture_modulator(self, name):
        return self.aperture_modulation_combiner.get(name)

    def get_imager(self):
        assert self.has_imager
        return self.imager

    def get_aperture(self):
        assert self.has_aperture
        return self.aperture_modulation_combiner.get('aperture')

    def visualize_source_field(self, name, **plot_kwargs):
        self.source_field_combiner.visualize(name, **plot_kwargs)

    def visualize_aperture_modulation_field(self, name, **plot_kwargs):
        self.aperture_modulation_combiner.visualize(name, **plot_kwargs)

    def visualize_combined_source_field(self, **plot_kwargs):
        fields.visualize_field(self.source_field_combiner.get_combined_field(), **plot_kwargs)

    def visualize_aperture_field(self, **plot_kwargs):
        fields.visualize_field(self.aperture_field, **plot_kwargs)

    def visualize_combined_aperture_modulation_field(self, **plot_kwargs):
        fields.visualize_field(self.aperture_modulation_combiner.get_combined_field(), **plot_kwargs)

    def visualize_modulated_aperture_field(self, **plot_kwargs):
        fields.visualize_field(self.compute_modulated_aperture_field(), **plot_kwargs)

    def visualize_image_field(self, **plot_kwargs):
        assert self.has_imager
        self.imager.visualize(**plot_kwargs)

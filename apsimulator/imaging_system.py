# -*- coding: utf-8 -*-
# This file is part of the APSimulator API.
# Author: Lars Frogner
import numpy as np
import matplotlib.pyplot as plt
import math_utils
import grids
import fields
import field_processing
import filters


class ImagingSystem:
    '''
    This class organizes the image formation pipeline. It is responsible for holding all
    the objects used at various stages of the image generation process and communicating
    intermediate results between them.
    '''
    def __init__(self, field_of_view_x, field_of_view_y, angular_coarseness, wavelengths, use_memmaps=False):
        self.field_of_view_x = float(field_of_view_x) # Field of view in the x-direction [rad]
        self.field_of_view_y = float(field_of_view_y) # Field of view in the y-direction [rad]
        self.angular_coarseness = float(angular_coarseness) # Angle subtended by a pixel in the center of the image plane [rad]
        self.wavelengths = np.asfarray(wavelengths) # Array of wavelengths for the incident light [m]
        self.use_memmaps = bool(use_memmaps) # Whether to conserve memory at the cost of performance by storing fields in memory mapped files

        assert(self.wavelengths.ndim == 1)
        self.n_wavelengths = self.wavelengths.size

        self.initialize_fields()

        self.has_imager = False
        self.has_aperture = False
        self.has_filters = False
        self.has_filtered_image_field = False

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

    def set_filter_set(self, filter_set):
        self.filter_set = filter_set
        self.has_filters = True

    def initialize_source_field(self):
        '''
        Constructs the grid for the source field and initializes the field.

        The source field is the angular and spectral distribution of amplitudes
        of the light field incident on the aperture, and has units of sqrt(W/m^2/m).

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

        source_field = fields.SpectralField(self.source_grid, self.wavelengths,
                                            dtype='float64',
                                            initial_value=0,
                                            use_memmap=self.use_memmaps)

        # Initialize the pipeline object for adding source fluxes to the source field
        self.source_pipeline = field_processing.FieldProcessingPipeline(source_field)

    def initialize_aperture_field(self):
        '''
        Constructs the grid for the aperture field and initializes the field.

        The aperture field is the spatial and spectral distribution of the total light
        field incident on the aperture from all sources. The field values are complex,
        representing both field strength (in the form of amplitudes) and field phases.
        The field values have units of sqrt(W/m^2/m).

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
                                                   use_memmap=self.use_memmaps)

        # Initialize the pipeline object for modulating the aperture field
        self.aperture_modulation_pipeline = field_processing.FieldProcessingPipeline(self.aperture_field)

    def initialize_aperture_grid_window(self):
        assert self.has_imager
        # Define a window in the grid covering the maximum aperture diameter.
        max_normalized_aperture_radius = self.imager.get_aperture_diameter()/(2*np.min(self.wavelengths))
        self.aperture_grid.define_window((-max_normalized_aperture_radius, max_normalized_aperture_radius),
                                         (-max_normalized_aperture_radius, max_normalized_aperture_radius))

    def initialize_image_field(self):
        assert self.has_imager

        self.imager.initialize_image_field(self.aperture_grid, self.wavelengths,
                                           self.field_of_view_x, self.field_of_view_y,
                                           use_memmap=self.use_memmaps)

        self.image_postprocessing_pipeline = field_processing.FieldProcessingPipeline(self.imager.get_image_field().create_window_field(copy_values=False))

    def add_source(self, label, source, store_field=False):
        self.source_pipeline.add_field_processor(label, source, store_process_field_if_possible=store_field)

    def add_aperture_modulator(self, label, aperture_modulator, store_field=False):
        self.aperture_modulation_pipeline.add_field_processor(label, aperture_modulator, store_process_field_if_possible=store_field)

    def add_image_postprocessor(self, label, image_postprocessor):
        assert self.has_imager
        self.image_postprocessing_pipeline.add_field_processor(label, image_postprocessor)

    def run_full_propagation(self):
        self.compute_source_field()
        self.compute_aperture_field()
        self.compute_modulated_aperture_field()
        self.compute_image_field()
        self.compute_postprocessed_image_field()
        self.compute_filtered_image_field()

    def compute_source_field(self):
        self.source_pipeline.compute_processed_field()

    def compute_aperture_field(self):
        '''
        Computes the aperture field by taking the Fourier transform of the source field.
        This corresponds to summing up the plane waves incident on the aperture plane
        from each direction in the source grid. The plane waves are assumed to have no
        initial phase difference, and the high-frequency temporal phase shifts due to the
        oscillations of the electromagnetic field are neglected (since these will average
        out over a very short time).
        '''
        total_source_field = self.get_source_field()
        self.aperture_field.set_values_inside_window(total_source_field.compute_fourier_transformed_values())

    def compute_modulated_aperture_field(self):
        self.aperture_modulation_pipeline.compute_processed_field()

    def compute_transmitted_aperture_field(self):
        assert self.has_aperture
        aperture = self.aperture_modulation_pipeline.get_processor('aperture')
        transmitted_aperture_field = self.get_aperture_field().copy()
        aperture.process(transmitted_aperture_field)
        return transmitted_aperture_field

    def compute_image_field(self):
        assert self.has_imager
        self.imager.compute_image_field(self.get_modulated_aperture_field())

    def compute_postprocessed_image_field(self):
        assert self.has_imager
        self.image_postprocessing_pipeline.compute_processed_field()

    def compute_filtered_image_field(self):
        assert self.has_filters
        self.filtered_image_field = self.filter_set.compute_filtered_image_field(self.get_postprocessed_image_field())
        self.has_filtered_image_field = True

    def get_source(self, label):
        return self.source_pipeline.get(label)

    def get_aperture_modulator(self, label):
        return self.aperture_modulation_pipeline.get(label)

    def get_imager(self):
        assert self.has_imager
        return self.imager

    def get_aperture(self):
        assert self.has_aperture
        return self.aperture_modulation_pipeline.get('aperture')

    def get_source_field(self):
        return self.source_pipeline.get_processed_field()

    def get_aperture_field(self):
        return self.aperture_field

    def get_modulated_aperture_field(self):
        return self.aperture_modulation_pipeline.get_processed_field()

    def get_image_field(self):
        assert self.has_imager
        return self.imager.get_image_field()

    def get_postprocessed_image_field(self):
        assert self.has_imager
        return self.image_postprocessing_pipeline.get_processed_field()

    def get_filtered_image_field(self):
        assert self.has_filtered_image_field
        return self.filtered_image_field

    def compute_rayleigh_limit(self, wavelength):
        assert self.has_imager
        return 1.22*wavelength/self.imager.get_aperture_diameter()

    def compute_spectral_powers_of_aperture_field(self, aperture_field):
        return np.sum(math_utils.abs2(aperture_field.get_values_inside_window()), axis=(1, 2))*self.wavelengths**2*self.aperture_grid.get_cell_area()

    def compute_incident_spectral_powers(self):
        return self.compute_spectral_powers_of_aperture_field(self.compute_transmitted_aperture_field())

    def compute_modulated_spectral_powers(self):
        return self.compute_spectral_powers_of_aperture_field(self.get_modulated_aperture_field())

    def visualize_energy_conservation(self):
        assert self.has_imager
        incident_spectral_powers = self.compute_incident_spectral_powers()
        modulated_spectral_powers = self.compute_modulated_spectral_powers()
        image_spectral_powers = self.imager.compute_spectral_powers()
        postprocessed_image_spectral_powers = self.imager.compute_spectral_powers_of_image_field(self.get_postprocessed_image_field())

        incident_power = np.trapz(incident_spectral_powers, x=self.wavelengths)
        modulated_power = np.trapz(modulated_spectral_powers, x=self.wavelengths)
        image_power = np.trapz(image_spectral_powers, x=self.wavelengths)
        postprocessed_image_power = np.trapz(postprocessed_image_spectral_powers, x=self.wavelengths)

        fig, ax = plt.subplots()
        alpha = 1
        ax.plot(self.wavelengths, incident_spectral_powers, '-', alpha=alpha, label='Incident ({:g} W total)'.format(incident_power))
        ax.plot(self.wavelengths, modulated_spectral_powers, '--', alpha=alpha, label='Modulated ({:g} W total)'.format(modulated_power))
        ax.plot(self.wavelengths, image_spectral_powers, ':', alpha=alpha, label='Image ({:g} W total)'.format(image_power))
        ax.plot(self.wavelengths, postprocessed_image_spectral_powers, '-.', alpha=alpha, label='Postprocessed image ({:g} W total)'.format(postprocessed_image_power))
        ax.set_title('Energy conservation')
        ax.set_xlabel('Wavelength [m]')
        ax.set_ylabel('Spectral power [W/m]')
        ax.legend(loc='best')
        plt.show()

    def visualize_source_field(self, label, **plot_kwargs):
        self.source_pipeline.visualize_process_field(label, **plot_kwargs)

    def visualize_aperture_modulation_field(self, label, **plot_kwargs):
        self.aperture_modulation_pipeline.visualize_process_field(label, **plot_kwargs)

    def visualize_total_source_field(self, **plot_kwargs):
        fields.visualize_field(self.get_source_field(), **plot_kwargs)

    def visualize_aperture_field(self, **plot_kwargs):
        fields.visualize_field(self.get_aperture_field(), **plot_kwargs)

    def visualize_modulated_aperture_field(self, **plot_kwargs):
        fields.visualize_field(self.get_modulated_aperture_field(), **plot_kwargs)

    def visualize_image_field(self, **plot_kwargs):
        fields.visualize_field(self.get_image_field(), **plot_kwargs)

    def visualize_postprocessed_image_field(self, **plot_kwargs):
        fields.visualize_field(self.get_postprocessed_image_field(), **plot_kwargs)

    def visualize_filtered_image_field(self, **plot_kwargs):
        fields.visualize_field(self.get_filtered_image_field(), **plot_kwargs)

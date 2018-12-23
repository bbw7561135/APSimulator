# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math_utils
import plot_utils


def compute_scaled_fried_parameter(reference_value, reference_wavelength, reference_zenith_angle, wavelength, zenith_angle):
    '''
    Scales the given Fried parameter value to the given wavelength and zenith angle.
    '''
    return reference_value*(wavelength/reference_wavelength)**(6/5)*(np.cos(zenith_angle)/np.cos(reference_zenith_angle))**(3/5)


class TurbulencePhaseScreen:
    '''
    Class for computing the phase perturbations of the incident light field across the telescope aperture
    due to turbulece in the atmosphere (seeing). The subharmonic method is based on Johansson & Gavel (1994).
    '''
    def __init__(self, fried_parameter_zenith, fried_parameter_wavelength, zenith_angle, n_subharmonic_levels, wavelengths, outer_scale=np.inf):

        # The Fried parameter is the aperture diameter for which the RMS phase perturbation would be equal to 1 radian [m].
        # The given reference value applies to light with the given wavelength (fried_parameter_wavelength) in the direction of the zenith
        self.fried_parameter = compute_scaled_fried_parameter(fried_parameter_zenith, fried_parameter_wavelength, 0, fried_parameter_wavelength, zenith_angle)

        self.fried_parameter_wavelength = fried_parameter_wavelength # Wavelength for which the given Fried parameter applies
        self.zenith_angle = zenith_angle # Angle of view direction with respect to the zenith (straight up) [rad]
        self.outer_scale = outer_scale # Largest size of the turbulent eddies [m]
        self.n_subharmonic_levels = n_subharmonic_levels # Number of subharmonic grids to use for improving large-scale accuracy
        self.wavelengths = wavelengths # Array of wavelengths [m]
        self.n_wavelengths = len(self.wavelengths)

    def get_fried_parameter(self, wavelength, zenith_angle):
        return compute_scaled_fried_parameter(self.fried_parameter, self.fried_parameter_wavelength, self.zenith_angle, wavelength, zenith_angle)

    def initialize_grids(self, optical_system):

        # Set up grids
        self.setup_aperture_grid(optical_system)
        self.setup_phase_screen_grid(1)

    def initialize(self, optical_system):

        self.initialize_grids(optical_system)

        # Precompute constant quantities for use with the filter functions
        self.compute_filter_function_constants()

        # Compute filter functions
        self.filter_functions = self.compute_filter_functions()

        if self.n_subharmonic_levels > 0:
            self.subharmonic_filter_functions = self.compute_subharmonic_filter_functions()

        # Initialize phase screen
        self.phase_screen_generator = self.generate_phase_screen()
        self.setup_phase_screen_canvas()

    def setup_aperture_grid(self, optical_system):

        self.aperture_diameter = optical_system.get_aperture_diameter() # [m]
        self.n_aperture_grid_cells = optical_system.get_n_aperture_grid_cells()
        self.n_pad_grid_cells = optical_system.get_n_pad_grid_cells()
        self.full_image_angular_coordinates = optical_system.get_full_image_angular_coordinates()
        self.normalized_grid_cell_extent = optical_system.get_normalized_aperture_grid_cell_extent() # [wavelengths]

        self.normalized_coordinates = np.arange(-self.n_aperture_grid_cells//2, self.n_aperture_grid_cells//2)*self.normalized_grid_cell_extent
        self.normalized_x_coordinate_mesh, self.normalized_y_coordinate_mesh = np.meshgrid(self.normalized_coordinates, self.normalized_coordinates, indexing='xy')
        self.normalized_distances = np.sqrt(self.normalized_x_coordinate_mesh**2 + self.normalized_y_coordinate_mesh**2)

        self.aperture_shift = 0 # Number of pixels that the aperture has shifted relative to the phase screen

    def setup_phase_screen_grid(self, aspect_number):

        self.n_screen_grid_cells_y = math_utils.nearest_higher_power_of_2(self.n_aperture_grid_cells)
        self.half_n_screen_grid_cells_y = self.n_screen_grid_cells_y//2
        self.normalized_screen_extent_y = self.n_screen_grid_cells_y*self.normalized_grid_cell_extent

        self.n_screen_grid_cells_x = int(aspect_number)*self.n_screen_grid_cells_y
        self.half_n_screen_grid_cells_x = self.n_screen_grid_cells_x//2
        self.normalized_screen_extent_x = self.n_screen_grid_cells_x*self.normalized_grid_cell_extent

    def setup_phase_screen_canvas(self):
        self.phase_screen_canvas = next(self.phase_screen_generator)

    def compute_filter_function_constants(self):
        '''
        Computes quantities that will be used repeatedly for computing filter functions.
        '''
        self.filter_function_scale = 2*np.pi*np.sqrt(0.00058)*self.fried_parameter_wavelength\
                                        /(self.fried_parameter**(5/6)*np.sqrt(self.normalized_screen_extent_x*self.normalized_screen_extent_y))
        self.outer_scale_frequency_squared = 1/self.outer_scale**2
        self.inverse_wavelengths_squared = 1/self.wavelengths**2

        if self.n_subharmonic_levels > 0:

            self.subharmonic_indices = np.arange(-3, 3) + 0.5 # Fractional indices in the subharmonic grids
            self.subharmonic_level_scales = 1/3**(np.arange(self.n_subharmonic_levels) + 1) # Length scaling for the subharmonic grids

            # Precompute the complex exponentials that will be modulated with the subharmonic filter function
            # in the computation of the low-frequency phase perturbations

            unit_x_mesh, unit_y_mesh = np.meshgrid(np.arange(self.n_screen_grid_cells_x)/self.n_screen_grid_cells_x,
                                                   np.arange(self.n_screen_grid_cells_y)/self.n_screen_grid_cells_y, indexing='xy')

            subharmonic_index_x_mesh, subharmonic_index_y_mesh = np.meshgrid(self.subharmonic_indices, self.subharmonic_indices, indexing='xy')

            unscaled_subharmonic_phase_shifts = np.multiply.outer(2*np.pi*subharmonic_index_x_mesh, unit_x_mesh) + \
                                                np.multiply.outer(2*np.pi*subharmonic_index_y_mesh, unit_y_mesh)

            subharmonic_phase_shifts = np.multiply.outer(self.subharmonic_level_scales, unscaled_subharmonic_phase_shifts)

            self.subharmonic_phases = np.cos(subharmonic_phase_shifts) + 1j*np.sin(subharmonic_phase_shifts) # Shape: (n_subharmonic_levels, 6, 6, n_grid_cells_y, n_grid_cells_x)

    def compute_filter_functions(self):
        '''
        Computes the filtering function used in the Kolmogorov turbulence model to generate
        phase perturbations. The filtering function is related to the power spectrum of the
        phase perturbations.
        '''
        normalized_x_frequencies = np.arange(-self.half_n_screen_grid_cells_x, self.half_n_screen_grid_cells_x)/self.normalized_screen_extent_x
        normalized_y_frequencies = np.arange(-self.half_n_screen_grid_cells_y, self.half_n_screen_grid_cells_y)/self.normalized_screen_extent_y
        normalized_x_frequency_mesh, normalized_y_frequency_mesh = np.meshgrid(normalized_x_frequencies, normalized_y_frequencies, indexing='xy')

        normalized_distance_frequencies_squared = normalized_x_frequency_mesh**2 + normalized_y_frequency_mesh**2
        normalized_distance_frequencies_squared[self.half_n_screen_grid_cells_y, self.half_n_screen_grid_cells_x] = 1 # Mask zero frequency to avoid division by zero

        distance_frequencies_squared = np.multiply.outer(self.inverse_wavelengths_squared, normalized_distance_frequencies_squared)

        filter_functions = self.filter_function_scale*self.inverse_wavelengths_squared[:, np.newaxis, np.newaxis]/(distance_frequencies_squared + self.outer_scale_frequency_squared)**(11/12)
        filter_functions[:, self.half_n_screen_grid_cells_y, self.half_n_screen_grid_cells_x] = 0 # Average phase perturbation (corresponds to zero frequency) should be zero

        # Scale to account for overlap of subharmonic grid
        if self.n_subharmonic_levels > 0:
            for offset in [-1, 1]:
                filter_functions[:, self.half_n_screen_grid_cells_y + offset, self.half_n_screen_grid_cells_x]          *= 0.5
                filter_functions[:, self.half_n_screen_grid_cells_y,          self.half_n_screen_grid_cells_x + offset] *= 0.5
                filter_functions[:, self.half_n_screen_grid_cells_y + offset, self.half_n_screen_grid_cells_x + offset] *= 0.75
                filter_functions[:, self.half_n_screen_grid_cells_y + offset, self.half_n_screen_grid_cells_x - offset] *= 0.75

        return filter_functions

    def compute_subharmonic_filter_functions(self):
        '''
        Computes the subharmonic filtering function used in the model of Vorontsov et al. (2008) to generate
        additional low-frequency phase perturbations. This yields phase screens with statistics that more
        accurately follows the analytical Kolmogorov power spectrum.
        '''
        unscaled_normalized_x_frequencies = self.subharmonic_indices/self.normalized_screen_extent_x
        unscaled_normalized_y_frequencies = self.subharmonic_indices/self.normalized_screen_extent_y

        unscaled_normalized_x_frequency_mesh, \
            unscaled_normalized_y_frequency_mesh = np.meshgrid(unscaled_normalized_x_frequencies,
                                                               unscaled_normalized_y_frequencies, indexing='xy')

        unscaled_normalized_distance_frequencies_squared = unscaled_normalized_x_frequency_mesh**2 + unscaled_normalized_y_frequency_mesh**2

        normalized_distance_frequencies_squared = np.multiply.outer(self.subharmonic_level_scales**2, unscaled_normalized_distance_frequencies_squared)

        distance_frequencies_squared = np.multiply.outer(self.inverse_wavelengths_squared, normalized_distance_frequencies_squared)

        subharmonic_filter_functions = self.filter_function_scale*self.subharmonic_level_scales[np.newaxis, :, np.newaxis, np.newaxis]\
                                       *self.inverse_wavelengths_squared[:, np.newaxis, np.newaxis, np.newaxis]/(distance_frequencies_squared + self.outer_scale_frequency_squared)**(11/12)
        subharmonic_filter_functions[:, :, 2:4, 2:4] = 0 # Average phase perturbation (corresponds to zero frequency) should be zero

        return subharmonic_filter_functions # Shape: (n_wavelengths, n_subharmonic_levels, 6, 6)

    def generate_white_noise(self, size):
        std_dev = 1/np.sqrt(2)
        return    np.random.normal(loc=0, scale=std_dev, size=size) + \
               1j*np.random.normal(loc=0, scale=std_dev, size=size)

    def generate_high_frequency_phase_perturbations(self):
        '''
        Generates a random phase screen by modulating complex white noise with the Kolmogorov filtering
        function and taking the inverse Fourier transform.
        '''
        noise = self.generate_white_noise((self.n_screen_grid_cells_y, self.n_screen_grid_cells_x))
        return (self.n_screen_grid_cells_x*self.n_screen_grid_cells_y)*np.fft.ifft2(np.fft.ifftshift(noise[np.newaxis, :, :]*self.filter_functions,
                                                                                                     axes=(1, 2)),
                                                                                    axes=(1, 2)).real

    def generate_low_frequency_phase_perturbations(self):
        '''
        Generates a random phase screen with perturbations at larger scales than in the standard approach.
        Adding this to the original phase screen provides a better fit to the Kolmogorov power spectrum.
        '''
        noise = self.generate_white_noise((self.n_subharmonic_levels, 6, 6))
        filtered_noise = self.subharmonic_filter_functions*noise[np.newaxis, :, :, :]
        phase_perturbations = np.tensordot(filtered_noise, self.subharmonic_phases, axes=([1, 2, 3], [0, 1, 2]))
        return phase_perturbations.real - np.mean(phase_perturbations.real, axis=(1, 2))[:, np.newaxis, np.newaxis]

    def generate_phase_screen(self):
        '''
        Generator method that computes a new phase screen and possibly adds some additional low-frequency
        perturbations.
        '''
        while True:
            phase_perturbations = self.generate_high_frequency_phase_perturbations()
            if self.n_subharmonic_levels > 0:
                phase_perturbations += self.generate_low_frequency_phase_perturbations()
            yield phase_perturbations

    def get_phase_screen_covering_aperture(self):
        '''
        Returns a view into the part of the phase screen canvas currently covering the aperture window.
        '''
        return self.phase_screen_canvas[:, :self.n_aperture_grid_cells, self.aperture_shift:self.aperture_shift+self.n_aperture_grid_cells]

    def get_monochromatic_phase_screen_covering_aperture(self, wavelength_idx):
        '''
        Like get_phase_screen_covering_aperture, but only returns the screen for a given wavelength index.
        '''
        return self.phase_screen_canvas[wavelength_idx, :self.n_aperture_grid_cells, self.aperture_shift:self.aperture_shift+self.n_aperture_grid_cells]

    def compute_perturbation_field(self):
        '''
        Generates plane waves on the aperture window for the current phase perturbations.
        '''
        phase_screen_covering_aperture = self.get_phase_screen_covering_aperture()
        return np.cos(phase_screen_covering_aperture) + 1j*np.sin(phase_screen_covering_aperture)

    def compute_high_frequency_autocorrelation(self):
        '''
        Computes the analytical autocorrelation function for the part of the model without any subharmonic
        components.
        '''
        return (self.n_screen_grid_cells_x*self.n_screen_grid_cells_y)*np.fft.ifft2(np.fft.ifftshift(self.filter_functions**2,
                                                                                                     axes=(1, 2)),
                                                                                    axes=(1, 2)).real

    def compute_low_frequency_autocorrelation(self):
        '''
        Computes the analytical autocorrelation function for the subharmonic components.
        '''
        return np.tensordot(self.subharmonic_filter_functions**2, self.subharmonic_phases, axes=([1, 2, 3], [0, 1, 2])).real

    def compute_structure_function(self):
        '''
        Computes the structure function describing the statistics of the phase screens.
        '''
        autocorrelation = self.compute_high_frequency_autocorrelation()
        if self.n_subharmonic_levels > 0:
            autocorrelation += self.compute_low_frequency_autocorrelation()
        return 2*(autocorrelation[:, 0:1, 0:1] - autocorrelation)

    def compute_analytical_structure_function(self, distance, fried_parameter):
        '''
        Computes the analytical structure function for the Kolmogorov turbulence model.
        Warning: Currently assumes the outer scale to be infinite.
        '''
        return 6.88*(distance/fried_parameter)**(5/3)

    def compute_analytical_modulation_transfer_function(self):
        '''
        Computes the analytical modulation transfer function for the phase screens, which
        corresponds to the Fourier transform of the time-averaged point spread function.
        '''
        distances = np.multiply.outer(self.wavelengths, self.normalized_distances)
        fried_parameters = self.get_fried_parameter(self.wavelengths, self.zenith_angle)[:, np.newaxis, np.newaxis]
        modulation_transfer_function = np.exp(-0.5*self.compute_analytical_structure_function(distances, fried_parameters))

        return modulation_transfer_function

    def compute_approximate_time_averaged_FWHM(self, wavelength, zenith_angle):
        return 0.98*wavelength/self.get_fried_parameter(wavelength, zenith_angle)

    def compute_time_averaged_point_spread_function(self, extent=3):
        '''
        Computes the long-exposure point spread function due for the Kolmogorov turbulence
        model.
        '''
        modulation_transfer_function = self.compute_analytical_modulation_transfer_function()

        padded_MTF = np.pad(modulation_transfer_function,
                            ((0, 0),
                             (self.n_pad_grid_cells, self.n_pad_grid_cells),
                             (self.n_pad_grid_cells, self.n_pad_grid_cells)),
                            'constant')

        fourier_coefficients = np.fft.fftshift(np.fft.fft2(padded_MTF, axes=(1, 2)), axes=(1, 2)).real

        max_FWHM = self.compute_approximate_time_averaged_FWHM(np.min(self.wavelengths), self.zenith_angle)
        start_idx = np.searchsorted(self.full_image_angular_coordinates, -extent*max_FWHM/2)
        end_idx = len(self.full_image_angular_coordinates) - start_idx

        point_spread_function = np.abs(fourier_coefficients[:, start_idx:end_idx, start_idx:end_idx])
        point_spread_function /= np.sum(point_spread_function, axis=(1, 2))[:, np.newaxis, np.newaxis]

        return point_spread_function

    def plot_monochromatic_point_spread_function(self, approximate_wavelength, extent=3):

        wavelength_idx = np.argmin(np.abs(self.wavelengths - approximate_wavelength))
        wavelength = self.wavelengths[wavelength_idx]

        point_spread_function = self.compute_time_averaged_point_spread_function(extent=extent)[wavelength_idx, :, :]

        pad = (len(self.full_image_angular_coordinates) - point_spread_function.shape[-1])//2
        angular_extent = np.array([self.full_image_angular_coordinates[pad], self.full_image_angular_coordinates[-pad-1],
                                   self.full_image_angular_coordinates[pad], self.full_image_angular_coordinates[-pad-1]])

        print(math_utils.arcsec_from_radian(self.compute_approximate_time_averaged_FWHM(wavelength, self.zenith_angle)))

        fig, ax = plt.subplots()

        ax.set_xlabel(r'$\alpha_x$ [arcsec]')
        ax.set_ylabel(r'$\alpha_y$ [arcsec]')
        ax.set_title(r'Time averaged atmospheric point spread function ($\lambda = {:g}$ nm)'.format(wavelength*1e9))

        image = ax.imshow(point_spread_function,
                          extent=math_utils.arcsec_from_radian(angular_extent),
                          origin='lower',
                          interpolation='none',
                          cmap=plt.get_cmap('gray'))

        plot_utils.add_colorbar(fig, ax, image, label='Weight')

        plt.show()

    def plot_color_point_spread_function(self, filter_set, clipping_factor=1, extent=3):

        point_spread_function = self.compute_time_averaged_point_spread_function(extent=extent)

        filtered_fluxes = filter_set.compute_filtered_fluxes(self.wavelengths, point_spread_function)
        filtered_flux_array = filter_set.construct_filtered_flux_array(filtered_fluxes, ['red', 'green', 'blue'], [0, 1, 2], color_axis=2)
        vmin = np.min(filtered_flux_array)
        vmax = np.max(filtered_flux_array)*clipping_factor
        scaled_color_image = (filtered_flux_array - vmin)/(vmax - vmin)

        pad = (len(self.full_image_angular_coordinates) - point_spread_function.shape[-1])//2
        angular_extent = np.array([self.full_image_angular_coordinates[pad], self.full_image_angular_coordinates[-pad-1],
                                   self.full_image_angular_coordinates[pad], self.full_image_angular_coordinates[-pad-1]])

        fig, ax = plt.subplots()

        ax.set_xlabel(r'$\alpha_x$ [arcsec]')
        ax.set_ylabel(r'$\alpha_y$ [arcsec]')
        ax.set_title('Time averaged atmospheric point spread function')

        ax.imshow(scaled_color_image,
                  extent=math_utils.arcsec_from_radian(angular_extent),
                  origin='lower',
                  interpolation='none')

        plt.show()

    def plot_structure_function_comparison(self, approximate_wavelength):
        '''
        Compares the expected average structure function of the generated phase screen with
        the analytical structure function of the Kolmogorov model.
        '''
        wavelength_idx = np.argmin(np.abs(self.wavelengths - approximate_wavelength))
        wavelength = self.wavelengths[wavelength_idx]

        distances = wavelength*np.arange(self.half_n_screen_grid_cells_x)*self.normalized_grid_cell_extent
        fried_parameter = self.get_fried_parameter(wavelength, self.zenith_angle)
        analytical_structure_function = self.compute_analytical_structure_function(distances, fried_parameter)

        structure_function = self.compute_structure_function()

        fig, ax = plt.subplots()
        ax.plot(distances, structure_function[wavelength_idx, 0, :self.half_n_screen_grid_cells_x], label='Measured')
        ax.plot(distances, analytical_structure_function, label='Analytical')
        ax.set_xlabel(r'$r$ [m]')
        ax.set_ylabel(r'$D_\phi$')
        ax.set_title(r'Structure function ($\lambda = {:g}$ nm)'.format(wavelength*1e9))
        ax.legend(loc='best')
        plt.show()


class MovingTurbulencePhaseScreen(TurbulencePhaseScreen):
    '''
    Extension of the phase screen class that incorporates temporal evolution due to wind speed.
    The phase screen moves across the aperture with the wind speed, with new phase screens being
    generated on demand and spliced together. The splicing method is based on Vorontsov et al. (2008).
    '''
    def __init__(self, fried_parameter_zenith, fried_parameter_wavelength, zenith_angle, wind_speed, n_subharmonic_levels, wavelengths, outer_scale=np.inf):

        super().__init__(fried_parameter_zenith, fried_parameter_wavelength, zenith_angle, n_subharmonic_levels, wavelengths, outer_scale=outer_scale)

        self.wind_speed = wind_speed

    def initialize_grids(self, optical_system):

        # Set up grids
        self.setup_aperture_grid(optical_system)
        self.setup_phase_screen_grid(2)

    def initialize(self, optical_system):

        self.initialize_grids(optical_system)

        self.compute_temporal_quantities()

        # Precompute constant quantities for use with the filter functions
        self.compute_filter_function_constants()

        # Compute filter functions
        self.filter_functions = self.compute_filter_functions()

        if self.n_subharmonic_levels > 0:
            self.subharmonic_filter_functions = self.compute_subharmonic_filter_functions()

        self.tapering_function = self.compute_tapering_function()

        # Initialize phase screen
        self.phase_screen_generator = self.generate_phase_screen()
        self.setup_phase_screen_canvas()

    def compute_temporal_quantities(self):

        # Approximate time scale for which image changes due to turbulence become significant (for the reference wavelength) [s]
        self.coherence_time = 0.31*self.fried_parameter/self.wind_speed

        # Speed with which to move the normalized grid across the aperture [1/s]
        self.normalized_wind_speed = self.wind_speed/self.fried_parameter_wavelength

        # Speed with which to move the normalized grid across the aperture [grid cells/s]
        self.grid_speed = self.normalized_wind_speed/self.normalized_grid_cell_extent

        # Time steps smaller than this will not give improved time resolution
        self.min_time_step = self.normalized_grid_cell_extent/self.normalized_wind_speed

        # Time for a point on the normalized grid to traverse the aperture [s]
        self.aperture_crossing_duration = self.aperture_diameter/self.wind_speed

        # Elapsed time [s]
        self.time = 0

    def compute_tapering_function(self):
        '''
        Computes sine function that can be used to taper the edges of phase screens in the x-direction.
        Two tapered phase screens can then be overlapped to produce a longer phase screen while still
        maintaining the correct statistics in the overlapping region.
        '''
        return np.sin(np.pi*np.arange(self.n_screen_grid_cells_x)/(self.n_screen_grid_cells_x-1))[np.newaxis, np.newaxis, :]

    def setup_phase_screen_canvas(self):
        '''
        Creates an array ("canvas") for storing a phase screen, with additional room in the x-direction
        for adding a new phase screen with overlap.
        '''

        # Create canvas that fits 1.5 phase screens in width
        self.phase_screen_canvas = np.zeros((self.n_wavelengths, self.n_screen_grid_cells_y, self.half_n_screen_grid_cells_x*3))

        # Insert the first phase screen into the leftmost two thirds of the canvas
        self.phase_screen_canvas[:, :, :self.n_screen_grid_cells_x] = next(self.phase_screen_generator)

        # Scale the initial phase screen so that it tapers off to the right
        self.phase_screen_canvas[:, :, self.half_n_screen_grid_cells_x:self.half_n_screen_grid_cells_x*2] *= self.tapering_function[:, :, self.half_n_screen_grid_cells_x:]

        # Generate the next phase screen, taper it in both ends and add to the rightmost two thirds of the canvas,
        # merging with the initial phase screen in the middle
        self.phase_screen_canvas[:, :, self.half_n_screen_grid_cells_x:] += next(self.phase_screen_generator)*self.tapering_function

    def move_phase_screen(self, time_step):
        '''
        Emulates movement of the phase screen across the aperture in the x-direction. An index for the aperture window
        into the canvas is updated according to the wind speed. When the rightmost end of the aperture window crosses
        the middle of the canvas, the content of the canvas is shifted to put the aperture window at the beginning
        again and a new phase screen is spliced to the end of the old one.
        '''
        assert time_step < self.aperture_crossing_duration

        time_step = max(time_step, self.min_time_step) # Put floor on time step

        # Compute number of grid cells to shift the aperture
        grid_cell_offset = int(round(self.grid_speed*time_step))

        # Update current shift
        self.aperture_shift += grid_cell_offset
        self.time += time_step

        # Handle shifting past the middle of the canvas
        if self.aperture_shift + self.n_aperture_grid_cells >= 2*self.half_n_screen_grid_cells_x:

            # Generate an empty canvas
            new_phase_screen_canvas = np.zeros(self.phase_screen_canvas.shape)

            # Insert the phase screen occupying the rightmost two thirds of the old canvas into the leftmost two thirds of the new canvas
            new_phase_screen_canvas[:, :, :self.half_n_screen_grid_cells_x*2] = self.phase_screen_canvas[:, :, self.half_n_screen_grid_cells_x:]

            # Generate the next phase screen, taper it in both ends and add to the rightmost two thirds of the canvas,
            # merging with the existing phase screen in the middle
            new_phase_screen_canvas[:, :, self.half_n_screen_grid_cells_x:] += next(self.phase_screen_generator)*self.tapering_function

            # Use the new canvas
            self.phase_screen_canvas = new_phase_screen_canvas

            # Move aperture back one third of the canvas
            self.aperture_shift -= self.half_n_screen_grid_cells_x

    def get_coherence_time(self):
        return self.coherence_time

    def animate(self, time_step_scale, duration, approximate_wavelength, output_path=False):
        '''
        Generates a movie showing the time evolution of the phase perturbations over the aperture.
        '''

        # Find index of closest wavelength
        wavelength_idx = np.argmin(np.abs(self.wavelengths - approximate_wavelength))
        wavelength = self.wavelengths[wavelength_idx]

        time_step = time_step_scale*self.coherence_time

        monochromatic_phase_screen = self.get_monochromatic_phase_screen_covering_aperture(wavelength_idx)

        spatial_extent = np.array([self.normalized_coordinates[0], self.normalized_coordinates[-1],
                                   self.normalized_coordinates[0], self.normalized_coordinates[-1]])*wavelength

        n_time_steps = int(np.ceil(duration/time_step)) + 1
        times = np.arange(n_time_steps)*time_step

        vmin = min(np.min(monochromatic_phase_screen), -np.max(monochromatic_phase_screen))
        vmax = -vmin

        title = ''

        figheight = 6
        aspect = 1.3

        fig = plt.figure(figsize=(aspect*figheight, figheight))
        ax = fig.add_subplot(111)

        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(r'$y$ [m]')
        ax.set_title('{}'.format(title))

        image = ax.imshow(monochromatic_phase_screen,
                          extent=spatial_extent,
                          origin='lower',
                          #aspect='auto',
                          vmin=vmin, vmax=vmax,
                          interpolation='none',
                          animated=True)

        #plot_utils.set_axis_aspect(1

        plot_utils.add_colorbar(fig, ax, image, label='Phase perturbation [rad]')

        time_text = ax.text(0.01, 0.99, '', color='white', ha='left', va='top', transform=ax.transAxes)

        plt.tight_layout(pad=1)

        def init():
            return image, time_text

        def update(time_idx):
            print('Frame {:d}/{:d}'.format(time_idx, n_time_steps))
            time = times[time_idx]
            self.move_phase_screen(time_step)
            image.set_array(self.get_monochromatic_phase_screen_covering_aperture(wavelength_idx))
            time_text.set_text('time: {:g} s'.format(time))

            return image, time_text

        anim = animation.FuncAnimation(fig, update, init_func=init, blit=True, frames=np.arange(n_time_steps))

        if output_path:
            anim.save(output_path, writer=animation.FFMpegWriter(fps=30,
                                                                 bitrate=3200,
                                                                 extra_args=['-vcodec', 'libx264']))
        else:
            plt.show()

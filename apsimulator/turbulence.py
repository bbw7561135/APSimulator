# -*- coding: utf-8 -*-
# This file is part of the APSimulator API.
# Author: Lars Frogner
import numpy as np
import scipy.signal
import field_processing
import grids
import fields
import math_utils
import parallel_utils


def compute_scaled_fried_parameter(reference_value, reference_wavelength, reference_zenith_angle, wavelength, zenith_angle):
    '''
    Scales the given Fried parameter value to the given wavelength and zenith angle.
    '''
    return reference_value*(wavelength/reference_wavelength)**(6/5)*(np.cos(zenith_angle)/np.cos(reference_zenith_angle))**(3/5)

def compute_zenith_angle_scaled_fried_parameter(reference_value, reference_zenith_angle, zenith_angle):
    return reference_value*(np.cos(zenith_angle)/np.cos(reference_zenith_angle))**(3/5)


def compute_wavelength_scaled_fried_parameter(reference_value, reference_wavelength, wavelength):
    return reference_value*(wavelength/reference_wavelength)**(6/5)


class KolmogorovTurbulence:
    '''
    Represents the Kolmogorov model of atmospheric seeing, where light passing through the
    atmosphere is phase shifted due to differences in refractive indices caused by turbulence.
    '''
    def __init__(self, reference_fried_parameter=0.1, reference_wavelength=500e-9, reference_zenith_angle=0, zenith_angle=0):

        # The Fried parameter is the aperture diameter for which the RMS phase perturbation would be equal to 1 radian [m].
        # The given reference value applies to light with the given wavelength (fried_parameter_wavelength) for the given zenith angle
        self.set_reference_fried_parameter(reference_fried_parameter, reference_wavelength, reference_zenith_angle, has_zenith_angle=False)
        self.set_zenith_angle(zenith_angle) # Angle of view direction with respect to the zenith (straight up) [rad]

    def set_reference_fried_parameter(self, reference_fried_parameter, reference_wavelength, reference_zenith_angle, has_zenith_angle=True):
        self.reference_fried_parameter = float(reference_fried_parameter)
        self.reference_wavelength = float(reference_wavelength) # Wavelength for which the given Fried parameter applies
        self.reference_zenith_angle = float(reference_zenith_angle)

        if has_zenith_angle:
            self.fried_parameter_at_zenith_angle = compute_zenith_angle_scaled_fried_parameter(self.reference_fried_parameter,
                                                                                               self.reference_zenith_angle, self.zenith_angle)

    def set_zenith_angle(self, zenith_angle):
        self.zenith_angle = float(zenith_angle)

        self.fried_parameter_at_zenith_angle = compute_zenith_angle_scaled_fried_parameter(self.reference_fried_parameter,
                                                                                           self.reference_zenith_angle, self.zenith_angle)

    def compute_fried_parameter(self, wavelength):
        return compute_wavelength_scaled_fried_parameter(self.fried_parameter_at_zenith_angle, self.reference_wavelength, wavelength)

    def compute_analytical_structure_function(self, distance, wavelength):
        return 6.88*(distance/self.compute_fried_parameter(wavelength))**(5/3)

    def compute_analytical_modulation_transfer_function(self, distance, wavelength):
        '''
        Computes the analytical modulation transfer function for the phase screens, which
        corresponds to the Fourier transform of the time-averaged point spread function.
        '''
        return np.exp(-0.5*self.compute_analytical_structure_function(distance, wavelength))

    def compute_approximate_time_averaged_FWHM(self, wavelength):
        return 0.976*wavelength/self.compute_fried_parameter(wavelength)

    def compute_moffat_PSF_fit(self, wavelengths, angular_distances, beta=4):

        HW = self.compute_approximate_time_averaged_FWHM(wavelengths)/2
        one_over_HW_squared = 1/HW**2
        theta_over_HW_squared = np.multiply.outer(one_over_HW_squared, angular_distances**2)

        constant_1 = 2**(1/beta) - 1
        constant_2 = (beta - 1)*constant_1**(1 - beta)/np.pi

        return constant_2*one_over_HW_squared[:, np.newaxis, np.newaxis]/(1/constant_1 + theta_over_HW_squared)**beta

    def compute_dual_moffat_PSF_fit(self, wavelengths, angular_distances, beta_1=7, beta_2=2):

        HW = self.compute_approximate_time_averaged_FWHM(wavelengths)/2
        one_over_HW_squared = 1/HW**2
        theta_over_HW_squared = np.multiply.outer(one_over_HW_squared, angular_distances**2)

        constant_1 = 2**(1/beta_1) - 1
        constant_2 = (beta_1 - 1)*constant_1**(1 - beta_1)/(2*np.pi)
        constant_3 = 2**(1/beta_2) - 1
        constant_4 = (beta_2 - 1)*constant_3**(1 - beta_2)/(2*np.pi)

        return one_over_HW_squared[:, np.newaxis, np.newaxis]*\
               (constant_2/(1/constant_1 + theta_over_HW_squared)**beta_1 +
                constant_4/(1/constant_3 + theta_over_HW_squared)**beta_2)

    def get_zenith_angle(self):
        return self.zenith_angle


class AveragedKolmogorovTurbulence(KolmogorovTurbulence, field_processing.FieldProcessor):
    '''
    Class for convolving the image field with the long-exposure point spread function for
    the Kolmogorov turbulence model.
    '''
    def __init__(self, reference_fried_parameter=0.1, reference_wavelength=500e-9, reference_zenith_angle=0, zenith_angle=0, minimum_psf_extent=None):

        super().__init__(reference_fried_parameter=reference_fried_parameter,
                         reference_wavelength=reference_wavelength,
                         reference_zenith_angle=reference_zenith_angle,
                         zenith_angle=zenith_angle)

        self.set_minimum_psf_extent(minimum_psf_extent)

    def set_minimum_psf_extent(self, minimum_psf_extent):
        self.minimum_psf_extent = None if minimum_psf_extent is None else float(minimum_psf_extent)

    def determine_optimal_psf_size(self):
        minimum_angular_width = self.minimum_psf_extent*np.max(self.compute_approximate_time_averaged_FWHM(self.wavelengths))
        minimum_width = 2*np.sin(minimum_angular_width/2)
        minimum_size = minimum_width/min(self.grid.cell_extent_x, self.grid.cell_extent_y)
        size = math_utils.nearest_higher_power_of_2(minimum_size)
        return min(size, min(self.grid.size_x, self.grid.size_y))

    def compute_point_spread_function(self):

        # Compute mesh of angular distances
        angular_distances = math_utils.polar_angle_from_direction_vector(*self.grid.get_coordinate_meshes())

        # Approximate the long-exposure Kolmogorov PSF as the sum of two Moffat functions
        point_spread_function = self.compute_dual_moffat_PSF_fit(self.wavelengths, angular_distances)

        # Normalize to ensure energy conservation
        point_spread_function /= np.sum(point_spread_function, axis=(1,2))[:, np.newaxis, np.newaxis]

        # Trim edges to reduce the size of the PSF array, by only keeping a central region covering the given number of FWHMs
        if self.minimum_psf_extent is not None:
            psf_size = self.determine_optimal_psf_size()
            x_index_range = (self.grid.size_x//2 - psf_size//2, self.grid.size_x//2 + psf_size//2)
            y_index_range = (self.grid.size_y//2 - psf_size//2, self.grid.size_y//2 + psf_size//2)
            point_spread_function = point_spread_function[:, y_index_range[0]:y_index_range[1], x_index_range[0]:x_index_range[1]]

        return point_spread_function

    def process(self, field):
        '''
        Implements the FieldProcessor method for convolving the given field with the
        time-averaged turbulence point spread function.
        '''
        point_spread_function = self.compute_point_spread_function()
        convolved_field = parallel_utils.parallel_fftconvolve(field.get_values_inside_window(), point_spread_function)
        field.set_values_inside_window(convolved_field)


class KolmogorovPhaseScreen(KolmogorovTurbulence, field_processing.MultiplicativeFieldProcessor):
    '''
    Class for computing the phase perturbations of the incident light field across the telescope aperture
    due to turbulece in the atmosphere (seeing). The subharmonic method is based on Johansson & Gavel (1994).
    '''
    def __init__(self, reference_fried_parameter=0.1,
                 reference_wavelength=500e-9, reference_zenith_angle=0, zenith_angle=0,
                 n_subharmonic_levels=0, outer_scale=np.inf):

        super().__init__(reference_fried_parameter=reference_fried_parameter,
                         reference_wavelength=reference_wavelength,
                         reference_zenith_angle=reference_zenith_angle,
                         zenith_angle=zenith_angle)

        self.set_n_subharmonic_levels(n_subharmonic_levels) # Number of subharmonic grids to use for improving large-scale accuracy
        self.set_outer_scale(outer_scale) # Largest size of the turbulent eddies [m]

    def set_n_subharmonic_levels(self, n_subharmonic_levels):
        self.n_subharmonic_levels = int(n_subharmonic_levels)

    def set_outer_scale(self, outer_scale):
        self.outer_scale = float(outer_scale)

    def precompute_processing_quantities(self):
        '''
        Implements the FieldProcessor method called after recieveing the properties of the aperture field.
        '''
        self.initialize_phase_screen()

    def initialize_phase_screen(self,  width_doublings=0):

        self.setup_phase_screen_grid(width_doublings)

        # Precompute constant quantities for use with the filter functions
        self.compute_filter_function_constants()

        # Compute filter functions
        self.filter_functions = self.compute_filter_functions()

        if self.n_subharmonic_levels > 0:
            self.subharmonic_filter_functions = self.compute_subharmonic_filter_functions()

        # Initialize phase screen
        self.phase_screen_generator = self.generate_phase_screen()
        self.setup_phase_screen_canvas()

        self.aperture_shift = 0

    def setup_phase_screen_grid(self, width_doublings):

        max_window_size = max(self.grid.window.shape)

        grid_size_exponent_y = math_utils.nearest_higher_power_of_2_exponent(max_window_size)
        normalized_screen_extent_y = self.grid.cell_extent_y*2**grid_size_exponent_y

        grid_size_exponent_x = grid_size_exponent_y + int(width_doublings)
        normalized_screen_extent_x = self.grid.cell_extent_x*2**grid_size_exponent_x

        self.screen_grid = grids.FFTGrid(grid_size_exponent_x, grid_size_exponent_y,
                                         normalized_screen_extent_x, normalized_screen_extent_y,
                                         is_centered=True, grid_type='aperture')

    def setup_phase_screen_canvas(self):
        self.phase_screen_canvas = next(self.phase_screen_generator)

    def compute_filter_function_constants(self):
        '''
        Computes quantities that will be used repeatedly for computing filter functions.
        '''
        self.filter_function_scale = 2*np.pi*np.sqrt(0.00058)*self.reference_wavelength\
                                        /(self.reference_fried_parameter**(5/6)*np.sqrt(self.screen_grid.get_area()))
        self.outer_scale_frequency_squared = 1/self.outer_scale**2
        self.inverse_wavelengths_squared = 1/self.wavelengths**2

        if self.n_subharmonic_levels > 0:

            self.subharmonic_indices = np.arange(-3, 3) + 0.5 # Fractional indices in the subharmonic grids
            self.subharmonic_level_scales = 1/3**(np.arange(self.n_subharmonic_levels) + 1) # Length scaling for the subharmonic grids

            # Precompute the complex exponentials that will be modulated with the subharmonic filter function
            # in the computation of the low-frequency phase perturbations

            unit_x_mesh, unit_y_mesh = np.meshgrid(np.arange(self.screen_grid.size_x)/self.screen_grid.size_x,
                                                   np.arange(self.screen_grid.size_y)/self.screen_grid.size_y, indexing='xy')

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
        normalized_x_frequencies = self.screen_grid.cell_numbers_x/self.screen_grid.extent_x
        normalized_y_frequencies = self.screen_grid.cell_numbers_y/self.screen_grid.extent_y
        normalized_x_frequency_mesh, normalized_y_frequency_mesh = np.meshgrid(normalized_x_frequencies, normalized_y_frequencies, indexing='xy')

        normalized_distance_frequencies_squared = normalized_x_frequency_mesh**2 + normalized_y_frequency_mesh**2
        normalized_distance_frequencies_squared[self.screen_grid.shift_y, self.screen_grid.shift_x] = 1 # Mask zero frequency to avoid division by zero

        distance_frequencies_squared = np.multiply.outer(self.inverse_wavelengths_squared, normalized_distance_frequencies_squared)

        filter_functions = self.filter_function_scale*self.inverse_wavelengths_squared[:, np.newaxis, np.newaxis]/(distance_frequencies_squared + self.outer_scale_frequency_squared)**(11/12)
        filter_functions[:, self.screen_grid.shift_y, self.screen_grid.shift_x] = 0 # Average phase perturbation (corresponds to zero frequency) should be zero

        # Scale to account for overlap of subharmonic grid
        if self.n_subharmonic_levels > 0:
            for offset in [-1, 1]:
                filter_functions[:, self.screen_grid.shift_y + offset, self.screen_grid.shift_x]          *= 0.5
                filter_functions[:, self.screen_grid.shift_y,          self.screen_grid.shift_x + offset] *= 0.5
                filter_functions[:, self.screen_grid.shift_y + offset, self.screen_grid.shift_x + offset] *= 0.75
                filter_functions[:, self.screen_grid.shift_y + offset, self.screen_grid.shift_x - offset] *= 0.75

        return filter_functions

    def compute_subharmonic_filter_functions(self):
        '''
        Computes the subharmonic filtering function used in the model of Vorontsov et al. (2008) to generate
        additional low-frequency phase perturbations. This yields phase screens with statistics that more
        accurately follows the analytical Kolmogorov power spectrum.
        '''
        unscaled_normalized_x_frequencies = self.subharmonic_indices/self.screen_grid.extent_x
        unscaled_normalized_y_frequencies = self.subharmonic_indices/self.screen_grid.extent_y

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
        noise = self.generate_white_noise((self.screen_grid.size_y, self.screen_grid.size_x))
        return self.screen_grid.get_total_size()*np.fft.ifft2(np.fft.ifftshift(noise[np.newaxis, :, :]*self.filter_functions,
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
        return self.phase_screen_canvas[:, :self.grid.window.size_y, self.aperture_shift:self.aperture_shift+self.grid.window.size_x]

    def get_monochromatic_phase_screen_covering_aperture(self, wavelength_idx):
        '''
        Like get_phase_screen_covering_aperture, but only returns the screen for a given wavelength index.
        '''
        return self.phase_screen_canvas[wavelength_idx, :self.grid.window.size_y, self.aperture_shift:self.aperture_shift+self.grid.window.size_x]

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
        return self.screen_grid.get_total_size()*np.fft.ifft2(np.fft.ifftshift(self.filter_functions**2,
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

    def process(self, field):
        '''
        Implements the FieldProcessor method for modulating the given field with the
        turbulence modulation field.
        '''
        phase_perturbations = self.get_phase_screen_covering_aperture()
        modulation_field_values = np.cos(phase_perturbations) + 1j*np.sin(phase_perturbations)
        field.multiply_within_window(modulation_field_values)


class MovingKolmogorovPhaseScreen(KolmogorovPhaseScreen):
    '''
    Extension of the phase screen class that incorporates temporal evolution due to wind speed.
    The phase screen moves across the aperture with the wind speed, with new phase screens being
    generated on demand and spliced together. The splicing method is based on Vorontsov et al. (2008).
    '''

    def __init__(self, reference_fried_parameter=0.1, wind_speed=10,
                 reference_wavelength=500e-9, reference_zenith_angle=0, zenith_angle=0,
                 n_subharmonic_levels=0, outer_scale=np.inf):

        super().__init__(reference_fried_parameter=reference_fried_parameter,
                         reference_wavelength=reference_wavelength,
                         reference_zenith_angle=reference_zenith_angle,
                         zenith_angle=zenith_angle)

        self.set_wind_speed(wind_speed) # Speed at which the phase screen moves across the aperture [m/s]

    def set_wind_speed(self, wind_speed):
        self.wind_speed = float(wind_speed)

    def precompute_processing_quantities(self):
        '''
        Implements the FieldProcessor method called after recieveing the properties of the aperture field.
        '''
        self.initialize_phase_screen(width_doublings=1)

    def compute_temporal_quantities(self):

        # Approximate time scale for which image changes due to turbulence become significant (for the reference wavelength) [s]
        self.coherence_time = 0.31*self.reference_fried_parameter/self.wind_speed

        # Speed with which to move the normalized grid across the aperture [1/s]
        self.normalized_wind_speed = self.wind_speed/self.reference_wavelength

        # Speed with which to move the normalized grid across the aperture [grid cells/s]
        self.grid_speed = self.normalized_wind_speed/self.grid.cell_extent_x

        # Time steps smaller than this will not give improved time resolution
        self.min_time_step = self.grid.cell_extent_x/self.normalized_wind_speed

        # Time for a point on the normalized grid to traverse the aperture [s]
        self.aperture_crossing_duration = self.grid.window.size_x*self.grid.cell_extent_x/self.normalized_wind_speed

        # Elapsed time [s]
        self.time = 0

    def compute_tapering_function(self):
        '''
        Computes sine function that can be used to taper the edges of phase screens in the x-direction.
        Two tapered phase screens can then be overlapped to produce a longer phase screen while still
        maintaining the correct statistics in the overlapping region.
        '''
        return np.sin(np.pi*np.arange(self.screen_grid.size_x)/(self.screen_grid.size_x-1))[np.newaxis, np.newaxis, :]

    def setup_phase_screen_canvas(self):
        '''
        Creates an array ("canvas") for storing a phase screen, with additional room in the x-direction
        for adding a new phase screen with overlap.
        '''

        # Create canvas that fits 1.5 phase screens in width
        self.phase_screen_canvas = np.zeros((self.n_wavelengths, self.n_screen_grid_cells_y, self.screen_grid.shift_x*3))

        # Insert the first phase screen into the leftmost two thirds of the canvas
        self.phase_screen_canvas[:, :, :self.screen_grid.size_x] = next(self.phase_screen_generator)

        # Scale the initial phase screen so that it tapers off to the right
        self.phase_screen_canvas[:, :, self.screen_grid.shift_x:self.screen_grid.shift_x*2] *= self.tapering_function[:, :, self.screen_grid.shift_x:]

        # Generate the next phase screen, taper it in both ends and add to the rightmost two thirds of the canvas,
        # merging with the initial phase screen in the middle
        self.phase_screen_canvas[:, :, self.screen_grid.shift_x:] += next(self.phase_screen_generator)*self.tapering_function

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
        if self.aperture_shift + self.grid.window.size_x >= 2*self.screen_grid.shift_x:

            # Generate an empty canvas
            new_phase_screen_canvas = np.zeros(self.phase_screen_canvas.shape)

            # Insert the phase screen occupying the rightmost two thirds of the old canvas into the leftmost two thirds of the new canvas
            new_phase_screen_canvas[:, :, :self.screen_grid.shift_x*2] = self.phase_screen_canvas[:, :, self.screen_grid.shift_x:]

            # Generate the next phase screen, taper it in both ends and add to the rightmost two thirds of the canvas,
            # merging with the existing phase screen in the middle
            new_phase_screen_canvas[:, :, self.screen_grid.shift_x:] += next(self.phase_screen_generator)*self.tapering_function

            # Use the new canvas
            self.phase_screen_canvas = new_phase_screen_canvas

            # Move aperture back one third of the canvas
            self.aperture_shift -= self.screen_grid.shift_x

    def get_coherence_time(self):
        return self.coherence_time

    def get_wind_speed(self):
        return self.wind_speed

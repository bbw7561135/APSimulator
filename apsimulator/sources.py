# -*- coding: utf-8 -*-
# This file is part of the APSimulator API.
# Author: Lars Frogner
import numpy as np
import fields
import filters
import field_processing
import math_utils
import physics_utils


class UniformStarField(field_processing.AdditiveFieldProcessor):

    def __init__(self, stellar_density=0.14, near_distance=2, far_distance=15000,
                       red_giant_fraction=0.01, red_supergiant_fraction=0.001,
                       temperature_variance_scale=0.1, luminosity_variance_scale=0.1,
                       combine_overlapping_stars=False,
                       seed=None):
        self.set_stellar_density(stellar_density) # Average number of stars per volume [1/pc^3]
        self.set_near_distance(near_distance) # Distance to where the uniform star field begins [pc]
        self.set_far_distance(far_distance) # Distance to where the uniform star field ends [pc]
        self.initialize_star_population()
        self.set_star_type_fractions(red_giant_fraction, red_supergiant_fraction)
        self.set_variance_scales(temperature_variance_scale, luminosity_variance_scale)
        self.set_seed(seed)
        self.set_combine_overlapping_stars(combine_overlapping_stars) # Whether to sum the fluxes of stars generated at the same position

    def initialize_star_population(self):
        self.star_population = physics_utils.StarPopulation()

    def set_stellar_density(self, stellar_density):
        self.stellar_density = math_utils.inverse_cubic_meters_from_inverse_cubic_parsecs(float(stellar_density)) # [1/m^3]

    def set_near_distance(self, near_distance):
        self.near_distance = math_utils.meters_from_parsecs(float(near_distance)) # [m]
        self.near_distance_cubed = self.near_distance**3

    def set_far_distance(self, far_distance):
        self.far_distance = math_utils.meters_from_parsecs(float(far_distance)) # [m]
        self.far_distance_cubed = self.far_distance**3

    def set_distance_range(self, near_distance, far_distance):
        self.set_near_distance(near_distance)
        self.set_far_distance(far_distance)

    def set_star_type_fractions(self, red_giant_fraction=0, red_supergiant_fraction=0):
        self.star_population.set_star_type_fractions(red_giant_fraction, red_supergiant_fraction)

    def set_variance_scales(self, temperature_variance_scale=0, luminosity_variance_scale=0):
        self.star_population.set_variance_scales(temperature_variance_scale, luminosity_variance_scale)

    def set_combine_overlapping_stars(self, combine_overlapping_stars):
        self.combine_overlapping_stars = bool(combine_overlapping_stars)

    def set_seed(self, seed):
        self.seed = None if seed is None else int(seed)
        self.random_generator = np.random.RandomState(seed=self.seed)
        self.star_population.set_seed(None if self.seed is None else self.seed + 1)

    def compute_visible_star_field_volume(self, field_of_view_x, field_of_view_y):
        return (3/4)*np.tan(field_of_view_x/2)*np.tan(field_of_view_y/2)*(self.far_distance_cubed - self.near_distance_cubed)

    def compute_average_number_of_visible_stars(self, field_of_view_x, field_of_view_y):
        return self.stellar_density*self.compute_visible_star_field_volume(field_of_view_x, field_of_view_y)

    def generate_star_count(self):

        extent_x, extent_y = self.grid.get_window_extents()

        field_of_view_x = math_utils.polar_angle_from_direction_vector_extent(extent_x)
        field_of_view_y = math_utils.polar_angle_from_direction_vector_extent(extent_y)

        average_number_of_visible_stars = self.compute_average_number_of_visible_stars(field_of_view_x, field_of_view_y)

        # Draw number of stars to generate from a Poisson distribution
        number_of_stars = self.random_generator.poisson(lam=average_number_of_visible_stars)

        return number_of_stars

    def generate_distances(self, number_of_stars):
        '''
        Samples distances to the stars using that the probability of finding
        a distance r is proportional to r^2.
        '''
        random_fractions = self.random_generator.random_sample(size=number_of_stars)
        return np.cbrt((self.far_distance_cubed - self.near_distance_cubed)*random_fractions + self.near_distance_cubed)

    def generate_temperatures_and_luminosities(self, number_of_stars):
        #temperatures = np.full(number_of_stars, 6000)
        #luminosities = np.full(number_of_stars, physics_utils.solar_luminosity)
        temperatures, luminosities = self.star_population.generate_temperatures_and_luminosities(number_of_stars)
        return temperatures, luminosities

    def generate_stars(self, number_of_stars):
        distances = self.generate_distances(number_of_stars)
        temperatures, luminosities = self.generate_temperatures_and_luminosities(number_of_stars)
        stars = physics_utils.BlackbodyStars(self.wavelengths, distances, temperatures, luminosities)
        return stars

    def generate_star_field(self):

        number_of_stars = self.generate_star_count()

        total_number_of_grid_cells = self.grid.get_total_window_size()

        # Generate 1D index into the image array for each star, using a uniform distribution
        star_indices = self.random_generator.randint(low=0, high=total_number_of_grid_cells, size=number_of_stars)

        stars = self.generate_stars(number_of_stars)

        # Compute flux spectrum recieved from each star
        spectral_fluxes = stars.compute_recieved_spectral_fluxes()

        if self.combine_overlapping_stars:
            # Handle multiple stars in the same grid cells by summing the spectral fluxes with the same indices
            star_indices, spectral_fluxes = self.sum_overlapping_values(star_indices, spectral_fluxes)

        star_field = np.zeros((self.n_wavelengths, total_number_of_grid_cells), dtype=self.dtype)
        star_field[:, star_indices] = spectral_fluxes
        star_field = star_field.reshape((self.n_wavelengths, *self.grid.window.shape))

        return star_field

    def sum_overlapping_values(self, positions, values):
        assert positions.ndim == 1
        assert values.ndim == 2
        assert values.shape[1] == positions.size

        indices_of_positions_when_sorted = np.argsort(positions)
        sorted_positions = positions[indices_of_positions_when_sorted]
        unique_positions, start_indices_of_unique_positions = np.unique(sorted_positions, return_index=True)
        indices_of_all_occurances_of_each_position = np.split(indices_of_positions_when_sorted, start_indices_of_unique_positions[1:])

        summed_values = np.empty((values.shape[0], unique_positions.size))

        for i in range(unique_positions.size):
            summed_values[:, i] = np.sum(values[:, indices_of_all_occurances_of_each_position[i]], axis=1)

        return unique_positions, summed_values

    def process(self, field):
        '''
        Implements the FieldProcessor method for adding the star field source field
        to the given field.
        '''
        field.add_within_window(self.generate_star_field())

    def plot_HR_diagram(self, absolute=False, output_path=False):

        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter

        stars = self.generate_stars(self.generate_star_count())
        spectral_fluxes = stars.compute_recieved_spectral_fluxes_at_distance(math_utils.meters_from_parsecs(10)) if absolute else \
                          stars.compute_recieved_spectral_fluxes()
        V_band_filter = filters.get_V_band_filter()
        V_band_fluxes = V_band_filter.compute_integrated_flux(self.wavelengths, spectral_fluxes)
        V_band_magnitudes = physics_utils.V_band_magnitude_from_flux(V_band_fluxes)

        fig, ax = plt.subplots()
        ax.scatter(stars.temperatures, V_band_magnitudes, c='b', s=0.01, alpha=0.5)
        ax.set_xlabel('Temperature [K]')
        ax.set_ylabel('{} visual magnitude'.format('Absolute' if absolute else 'Apparent'))
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_xscale('log')
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_formatter(ScalarFormatter())
        ax.set_xticks([2000, 4000, 8000, 16000, 32000])

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()

    def get_stellar_density(self):
        return self.stellar_density

    def get_near_distance(self):
        return self.near_distance

    def get_far_distance(self):
        return self.far_distance

    def get_distance_range(self):
        return self.near_distance, self.far_distance

    def get_red_giant_fraction(self):
        return self.star_population.get_red_giant_fraction()

    def get_red_supergiant_fraction(self):
        return self.star_population.get_red_supergiant_fraction()

    def get_main_sequence_fraction(self):
        return self.star_population.get_main_sequence_fraction()

    def get_temperature_variance_scale(self):
        return self.star_population.get_temperature_variance_scale()

    def get_luminosity_variance_scale(self):
        return self.star_population.get_luminosity_variance_scale()


class UniformEmissionLineSkyglow(field_processing.AdditiveFieldProcessor):

    def __init__(self, bortle_class=3, emission_wavelengths=[500e-9], relative_emission_strengths=[1]):
        self.set_bortle_class(bortle_class)
        self.set_emission_wavelengths(emission_wavelengths)
        self.set_relative_emission_strengths(relative_emission_strengths)

    def set_bortle_class(self, bortle_class):
        self.bortle_class = float(bortle_class)

    def set_emission_wavelengths(self, emission_wavelengths):
        self.emission_wavelengths = np.asfarray(emission_wavelengths)
        assert self.emission_wavelengths.ndim == 1

    def set_relative_emission_strengths(self, relative_emission_strengths):
        self.relative_emission_strengths = np.asfarray(relative_emission_strengths)
        assert self.relative_emission_strengths.ndim == 1

    def generate_skyglow(self):

        included_emission_mask = np.logical_and(self.emission_wavelengths >= self.wavelengths[0], self.emission_wavelengths <= self.wavelengths[-1])
        included_emission_wavelengths = self.emission_wavelengths[included_emission_mask]
        included_relative_emission_strengths = self.relative_emission_strengths[included_emission_mask]

        # Estimate the solid angle of the field of view
        extent_x, extent_y = self.grid.get_window_extents()
        field_of_view_x = math_utils.polar_angle_from_direction_vector_extent(extent_x)
        field_of_view_y = math_utils.polar_angle_from_direction_vector_extent(extent_y)
        field_of_view_solid_angle = field_of_view_x*field_of_view_y

        # Model emission line skyglow by setting non-zero spectral fluxes for the wavelengths closest to the emission wavelengths
        spectral_fluxes = np.zeros(self.wavelengths.size)
        for i in range(included_emission_wavelengths.size):
            closest_wavelength_idx = np.argmin(np.abs(self.wavelengths - included_emission_wavelengths[i]))
            spectral_fluxes[closest_wavelength_idx] = included_relative_emission_strengths[i]

        # Compute the total flux in the visual band that the skyglow should produce
        target_V_band_magnitude = physics_utils.V_band_magnitude_from_bortle_class(self.bortle_class, field_of_view_solid_angle)
        target_V_band_flux = physics_utils.V_band_flux_from_magnitude(target_V_band_magnitude)

        # Compute the scale for the spectral flux values that will produce the wanted visual band flux
        V_band_filter = filters.get_V_band_filter()
        integrated_V_band_spectral_flux = V_band_filter.compute_integrated_flux(self.wavelengths, spectral_fluxes)
        flux_scale = target_V_band_flux/integrated_V_band_spectral_flux if integrated_V_band_spectral_flux > 0 else 0

        # Divide the fluxes by the total number of grid cells so that summing the flux over all grid cells
        # yields the correct total flux
        flux_scale /= self.grid.get_total_window_size()

        spectral_fluxes *= flux_scale

        return spectral_fluxes

    def process(self, field):
        '''
        Implements the FieldProcessor method for adding the skyglow source field
        to the given field.
        '''
        field += self.generate_skyglow()[:, np.newaxis, np.newaxis]

    def apply_process_field(self, field, process_field):
        field += process_field.values

    def get_bortle_class(self):
        return self.bortle_class

    def get_emission_wavelengths(self):
        return self.emission_wavelengths

    def get_relative_emission_strengths(self):
        return self.relative_emission_strengths


class UniformBlackbodySkyglow(field_processing.AdditiveFieldProcessor):

    def __init__(self, bortle_class=3, color_temperature=5000):
        self.set_bortle_class(bortle_class)
        self.set_color_temperature(color_temperature)

    def set_bortle_class(self, bortle_class):
        self.bortle_class = float(bortle_class)

    def set_color_temperature(self, color_temperature):
        self.color_temperature = float(color_temperature)

    def generate_skyglow(self):

        # Estimate the solid angle of the field of view
        extent_x, extent_y = self.grid.get_window_extents()
        field_of_view_x = math_utils.polar_angle_from_direction_vector_extent(extent_x)
        field_of_view_y = math_utils.polar_angle_from_direction_vector_extent(extent_y)
        field_of_view_solid_angle = field_of_view_x*field_of_view_y

        # Model spectral flux distribution of the skyglow as a blackbody spectrum of the specified temperature
        spectral_fluxes = physics_utils.compute_blackbody_spectral_fluxes(self.wavelengths, self.color_temperature)

        # Compute the total flux in the visual band that the skyglow should produce
        target_V_band_magnitude = physics_utils.V_band_magnitude_from_bortle_class(self.bortle_class, field_of_view_solid_angle)
        target_V_band_flux = physics_utils.V_band_flux_from_magnitude(target_V_band_magnitude)

        # Compute the scale for the spectral flux values that will produce the wanted visual band flux
        V_band_filter = filters.get_V_band_filter()
        integrated_V_band_spectral_flux = V_band_filter.compute_integrated_flux(self.wavelengths, spectral_fluxes)
        flux_scale = target_V_band_flux/integrated_V_band_spectral_flux

        # Divide the fluxes by the total number of grid cells so that summing the flux over all grid cells
        # yields the correct total flux
        flux_scale /= self.grid.get_total_window_size()

        spectral_fluxes *= flux_scale

        return spectral_fluxes

    def process(self, field):
        '''
        Implements the FieldProcessor method for adding the skyglow source field
        to the given field.
        '''
        field += self.generate_skyglow()[:, np.newaxis, np.newaxis]

    def apply_process_field(self, field, process_field):
        field += process_field.values

    def get_bortle_class(self):
        return self.bortle_class

    def get_color_temperature(self):
        return self.color_temperature


class MoonSkyglow(UniformBlackbodySkyglow):
    # Possible model: Bradley Schaefer (1998) (used in Stellarioum)
    def __init__(self, illumination_percentage, altitude_angle, relative_polar_angle):
        self.illumination_percentage = float(illumination_percentage)
        self.relative_polar_angle = float(relative_polar_angle)

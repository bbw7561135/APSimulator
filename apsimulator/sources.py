# -*- coding: utf-8 -*-
# This file is part of the APSimulator API.
# Author: Lars Frogner
import numpy as np
import fields
import field_processing
import math_utils
import physics_utils


def sum_overlapping_values(positions, values):
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


class UniformStarField(field_processing.AdditiveFieldProcessor):

    def __init__(self, stellar_density, near_distance, far_distance, combine_overlapping_stars=False, seed=None):
        self.set_stellar_density(stellar_density) # Average number of stars per volume [1/m^3]
        self.set_near_distance(near_distance) # Distance to where the uniform star field begins [m]
        self.set_far_distance(far_distance) # Distance to where the uniform star field ends [m]
        self.set_combine_overlapping_stars(combine_overlapping_stars) # Whether to sum the fluxes of stars generated at the same position
        self.set_seed(seed)

    def set_stellar_density(self, stellar_density):
        self.stellar_density = float(stellar_density)

    def set_near_distance(self, near_distance):
        self.near_distance = float(near_distance)
        self.near_distance_cubed = self.near_distance**3

    def set_far_distance(self, far_distance):
        self.far_distance = float(far_distance)
        self.far_distance_cubed = self.far_distance**3

    def set_distance_range(self, near_distance, far_distance):
        self.set_near_distance(near_distance)
        self.set_far_distance(far_distance)

    def set_combine_overlapping_stars(self, combine_overlapping_stars):
        self.combine_overlapping_stars = bool(combine_overlapping_stars)

    def set_seed(self, seed):
        self.seed = None if seed is None else int(seed)
        self.random_generator = np.random.RandomState(seed=self.seed)

    def compute_visible_star_field_volume(self, field_of_view_x, field_of_view_y):
        return (3/4)*np.tan(field_of_view_x/2)*np.tan(field_of_view_y/2)*(self.far_distance_cubed - self.near_distance_cubed)

    def compute_average_number_of_visible_stars(self, field_of_view_x, field_of_view_y):
        return self.stellar_density*self.compute_visible_star_field_volume(field_of_view_x, field_of_view_y)

    def generate_distances(self, n_stars):
        '''
        Samples distances to the stars using that the probability of finding
        a distance r is proportional to r^2.
        '''
        random_fractions = self.random_generator.random_sample(size=n_stars)
        return np.cbrt((self.far_distance_cubed - self.near_distance_cubed)*random_fractions + self.near_distance_cubed)

    def generate_temperatures(self, n_stars):
        return np.full(n_stars, 6000)

    def generate_luminosities(self, n_stars):
        return np.full(n_stars, physics_utils.solar_luminosity)

    def generate_star_field(self):

        total_number_of_grid_cells = self.grid.get_total_window_size()
        extent_x, extent_y = self.grid.get_window_extents()

        field_of_view_x = math_utils.polar_angle_from_direction_vector_extent(extent_x)
        field_of_view_y = math_utils.polar_angle_from_direction_vector_extent(extent_y)

        average_number_of_visible_stars = self.compute_average_number_of_visible_stars(field_of_view_x, field_of_view_y)

        # Draw number of stars to generate from a Poisson distribution
        n_stars = self.random_generator.poisson(lam=average_number_of_visible_stars)

        # Generate 1D index into the image array for each star, using a uniform distribution
        star_indices = self.random_generator.randint(low=0, high=total_number_of_grid_cells, size=n_stars)

        distances = self.generate_distances(n_stars)
        temperatures = self.generate_temperatures(n_stars)
        luminosities = self.generate_luminosities(n_stars)

        # Compute flux spectrum recieved from each star
        spectral_fluxes = physics_utils.BlackbodyStars(self.wavelengths, distances, temperatures, luminosities).compute_recieved_spectral_fluxes()

        if self.combine_overlapping_stars:
            # Handle multiple stars in the same grid cells by summing the spectral fluxes with the same indices
            star_indices, spectral_fluxes = sum_overlapping_values(star_indices, spectral_fluxes)

        star_field = np.zeros((self.n_wavelengths, total_number_of_grid_cells), dtype=self.dtype)
        star_field[:, star_indices] = spectral_fluxes
        star_field = star_field.reshape((self.n_wavelengths, *self.grid.window.shape))

        return star_field

    def process(self, field):
        '''
        Implements the FieldProcessor method for adding the star field source field
        to the given field.
        '''
        field.add_within_window(self.generate_star_field())

    def get_stellar_density(self):
        return self.stellar_density

    def get_near_distance(self):
        return self.near_distance

    def get_far_distance(self):
        return self.far_distance

    def get_distance_range(self):
        return self.near_distance, self.far_distance

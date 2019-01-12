# -*- coding: utf-8 -*-
# This file is part of the APSimulator API.
# Author: Lars Frogner
import numpy as np
import math_utils


planck_constant = 6.626070150e-34 # [J s]
speed_of_light = 2.99792458e8 # [m/s]
boltzmann_constant = 1.38064852e-23 # [J/K]
stefan_boltzmann_constant = 5.670367e-8 # [W/m^2/K^4]
light_year = 9.4607304725808e15 # [m]

solar_mass = 1.98847e30 # [kg]
solar_radius = 6.957e8 # [m]
solar_luminosity = 3.828e26 # [W]
solar_surface_temperature = 5778.0 # [K]

V_band_reference_flux = 3e-9 # [W/m^2]


def compute_photons_per_energy_unit(wavelengths):
    photons_per_energy_unit = wavelengths/(planck_constant*speed_of_light)
    return photons_per_energy_unit


def magnitude_from_flux(flux, reference_flux):
    return -2.5*np.log10(flux/reference_flux)


def flux_from_magnitude(magnitude, reference_flux):
    return reference_flux*10**(-0.4*magnitude)


def V_band_magnitude_from_flux(flux):
    return magnitude_from_flux(flux, V_band_reference_flux)


def V_band_flux_from_magnitude(magnitude):
    return flux_from_magnitude(magnitude, V_band_reference_flux)


def V_band_magnitude_from_bortle_class(bortle_class, field_of_view_solid_angle):
     bortle_classes = [0, 1, 2, 3, 4, 5, 7]
     bortle_magnitudes_per_square_arcsec = np.array([22.0, 21.7, 21.5, 21.3, 20.4, 19.1, 18.0])
     bortle_magnitudes = bortle_magnitudes_per_square_arcsec - 2.5*np.log10(math_utils.square_arcsec_from_steradians(field_of_view_solid_angle))
     return math_utils.interp_extrap(bortle_class, bortle_classes, bortle_magnitudes)


def compute_blackbody_spectral_fluxes(wavelengths, temperature):
    return (2*np.pi*planck_constant*speed_of_light**2)/((np.exp((planck_constant*speed_of_light/boltzmann_constant)/(wavelengths*temperature)) - 1)
                                                         *wavelengths**5)


class StarPopulation:

    def __init__(self, red_giant_fraction=0, red_supergiant_fraction=0,
                       temperature_variance_scale=0, luminosity_variance_scale=0,
                       seed=None):
        self.set_star_type_fractions(red_giant_fraction, red_supergiant_fraction)
        self.set_variance_scales(temperature_variance_scale, luminosity_variance_scale)
        self.set_seed(seed)
        self.initialize_mass_ranges()
        self.initialize_coefficients()

    def set_star_type_fractions(self, red_giant_fraction=0, red_supergiant_fraction=0):
        self.red_giant_fraction = float(red_giant_fraction)
        self.red_supergiant_fraction = float(red_supergiant_fraction)
        self.main_sequence_fraction = 1 - (self.red_giant_fraction + self.red_supergiant_fraction)
        assert self.main_sequence_fraction >= 0

    def set_variance_scales(self, temperature_variance_scale=0, luminosity_variance_scale=0):
        self.temperature_variance_scale = float(temperature_variance_scale)
        self.luminosity_variance_scale = float(luminosity_variance_scale)

    def set_seed(self, seed):
        self.seed = None if seed is None else int(seed)
        self.random_generator = np.random.RandomState(seed=self.seed)

    def initialize_mass_ranges(self):
        self.mass_ranges = {'main sequence':  (0.1, 50),
                            'red giant':      (3,   8),
                            'red supergiant': (16,  35)} # [solar masses]

    def initialize_coefficients(self):
        '''
        From Zaninetti (2008).
        '''
        self.temperature_coefs = {'main sequence':  (1/10**(-7.76),   1/2.06),
                                  'red giant':      (1/10**(5.8958), -1/1.4563),
                                  'red supergiant': (1/10**(3.73),  -1/0.64)}

        self.luminosity_coefs = {'main sequence':  (10**0.062, 3.43),
                                 'red giant':      (10**0.32,  2.79),
                                 'red supergiant': (10**1.29,  2.43)}

    def generate_star_masses(self, number_of_stars, star_type):
        '''
        Generates masses using the initial mass function (IMF) of Maschberger (2012).
        The IMF describes the probability distribution of masses for a population of
        stars as they enter the main sequence. The returned masses are in units of the
        solar mass.
        '''
        # Parameters for the Maschberger IMF
        mu = 0.2 # [solar masses]
        alpha = 2.3 # [solar masses]
        beta = 1.4 # [solar masses]

        # Helper function
        def G(m):
            return (1 + (m/mu)**(1 - alpha))**(1 - beta)

        mass_range = self.mass_ranges[star_type]

        random_fractions = self.random_generator.random_sample(size=number_of_stars)
        return mu*((random_fractions*(G(mass_range[1]) - G(mass_range[0])) + G(mass_range[0]))**(1/(1 - beta)) - 1)**(1/(1 - alpha))

    def compute_temperatures(self, masses, star_type):
        return (self.temperature_coefs[star_type][0]*masses)**self.temperature_coefs[star_type][1]

    def compute_luminosities(self, masses, star_type):
        return self.luminosity_coefs[star_type][0]*(masses**self.luminosity_coefs[star_type][1])

    def generate_randomized_temperatures(self, masses, star_type):
        temperatures = self.compute_temperatures(masses, star_type)
        if self.temperature_variance_scale > 0:
            temperatures = self.random_generator.normal(loc=temperatures, scale=temperatures*self.temperature_variance_scale)
        return temperatures

    def generate_randomized_luminosities(self, masses, star_type):
        luminosities = self.compute_luminosities(masses, star_type)
        if self.luminosity_variance_scale > 0:
            luminosities = self.random_generator.normal(loc=luminosities, scale=luminosities*self.luminosity_variance_scale)
        return luminosities

    def generate_temperatures_and_luminosities(self, number_of_stars):

        n_red_giant_stars = int(self.red_giant_fraction*number_of_stars)
        n_red_supergiant_stars = int(self.red_supergiant_fraction*number_of_stars)
        total_n_giant_stars = n_red_giant_stars + n_red_supergiant_stars
        n_main_sequence_stars = number_of_stars - total_n_giant_stars

        masses = np.empty(number_of_stars)
        luminosities = np.empty(number_of_stars)
        temperatures = np.empty(number_of_stars)

        masses[:n_red_giant_stars] = self.generate_star_masses(n_red_giant_stars, 'red giant')
        masses[n_red_giant_stars:total_n_giant_stars] = self.generate_star_masses(n_red_supergiant_stars, 'red supergiant')
        masses[total_n_giant_stars:] = self.generate_star_masses(n_main_sequence_stars, 'main sequence')

        temperatures[:n_red_giant_stars] = self.generate_randomized_temperatures(masses[:n_red_giant_stars], 'red giant')
        temperatures[n_red_giant_stars:total_n_giant_stars] = self.generate_randomized_temperatures(masses[n_red_giant_stars:total_n_giant_stars], 'red supergiant')
        temperatures[total_n_giant_stars:] = self.generate_randomized_temperatures(masses[total_n_giant_stars:], 'main sequence')

        luminosities[:n_red_giant_stars] = self.generate_randomized_luminosities(masses[:n_red_giant_stars], 'red giant')
        luminosities[n_red_giant_stars:total_n_giant_stars] = self.generate_randomized_luminosities(masses[n_red_giant_stars:total_n_giant_stars], 'red supergiant')
        luminosities[total_n_giant_stars:] = self.generate_randomized_luminosities(masses[total_n_giant_stars:], 'main sequence')

        return temperatures, luminosities*solar_luminosity

    def get_red_giant_fraction(self):
        return self.red_giant_fraction

    def get_red_supergiant_fraction(self):
        return self.red_supergiant_fraction

    def get_main_sequence_fraction(self):
        return self.main_sequence_fraction

    def get_temperature_variance_scale(self):
        return self.temperature_variance_scale

    def get_luminosity_variance_scale(self):
        return self.luminosity_variance_scale


class BlackbodyStars:

    def __init__(self, wavelengths, distances, temperatures, luminosities):
        self.wavelengths = np.asfarray(wavelengths) # Wavelengths to compute spectra for [m]
        self.distances = np.asfarray(distances) # Distances from the stars to the observer [m]
        self.temperatures = np.asfarray(temperatures) # Surface temperatures of the stars [K]
        self.luminosities = np.asfarray(luminosities) # Bolometric luminosities of the stars [m]

        assert self.wavelengths.ndim == 1
        assert self.distances.ndim == 1
        assert self.temperatures.ndim == 1
        assert self.luminosities.ndim == 1
        assert self.temperatures.size == self.distances.size
        assert self.luminosities.size == self.distances.size

    def compute_emitted_spectral_fluxes(self):
        one_over_lambda_T = np.multiply.outer(1/self.wavelengths, 1/self.temperatures)
        return (2*np.pi*planck_constant*speed_of_light**2)/((np.exp((planck_constant*speed_of_light/boltzmann_constant)*one_over_lambda_T) - 1)
                                                             *self.wavelengths[:, np.newaxis]**5)

    def compute_emitted_total_fluxes(self):
        return stefan_boltzmann_constant*self.temperatures**4

    def compute_surface_areas(self):
        return self.luminosities/self.compute_emitted_total_fluxes()

    def compute_spectral_luminosities(self):
        return self.compute_emitted_spectral_fluxes()*self.compute_surface_areas()[np.newaxis, :]

    def compute_recieved_spectral_fluxes_at_distance(self, distance):
        return self.compute_spectral_luminosities()/(4*np.pi*distance**2)

    def compute_recieved_spectral_fluxes(self):
        return self.compute_recieved_spectral_fluxes_at_distance(self.distances)




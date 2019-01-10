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

solar_radius = 6.957e8 # [m]
solar_luminosity = 3.828e26 # [W]

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

    def compute_recieved_spectral_fluxes(self):
        return self.compute_spectral_luminosities()/(4*np.pi*self.distances**2)

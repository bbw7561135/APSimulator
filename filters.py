# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
from builtins import zip
import numpy as np


def construct_filtered_flux_array(filtered_fluxes, filter_names, filter_indices, color_axis=0):

    n_included_filters = len(filter_names)

    assert len(filter_indices) == n_included_filters
    assert set(filter_indices) == set(range(n_included_filters))
    assert set(filter_names).issubset(filtered_fluxes.keys())

    filtered_flux_array = np.zeros((len(filter_names), *filtered_fluxes[filter_names[0]].shape), dtype=filtered_fluxes[filter_names[0]].dtype)

    for idx in range(n_included_filters):
        filtered_flux_array[filter_indices[idx], :] = filtered_fluxes[filter_names[idx]]

    return filtered_flux_array.swapaxes(0, color_axis)


class FilterSet:
    '''
    Class for holding a set of filters and computing fluxes integrated through
    each filter.
    '''
    def __init__(self, **filters):
        self.filters = filters
        self.n_filters = len(self.filters)

    def compute_filtered_fluxes(self, wavelengths, spectral_fluxes, wavelength_axis=0):

        filtered_fluxes = {}

        for filter_name in self.filters:
            filtered_fluxes[filter_name] = self.filters[filter_name].compute_integrated_flux(wavelengths, spectral_fluxes, wavelength_axis)

        return filtered_fluxes

    def has(self, filter_names):
        return set(filter_names).issubset(self.filters.keys())


class Filter:
    '''
    Class representing a wavelength filter.
    '''
    def __init__(self, minimum_wavelength, maximum_wavelength):
        self.minimum_wavelength = minimum_wavelength
        self.maximum_wavelength = maximum_wavelength

    def compute_weighted_spectrum(self, wavelengths, spectrum):
        '''
        No weighting is used by default, but this can be overridden by subclasses.
        '''
        return spectrum

    def compute_integrated_flux(self, wavelengths, spectral_fluxes, wavelength_axis):
        '''
        Integrates the given spectral fluxes weighted with the filter transmittance
        to determine the flux passing throught the filter.
        '''
        assert len(wavelengths) == spectral_fluxes.shape[wavelength_axis]

        # Find index range covering the filter wavelengths
        idx_range = np.searchsorted(wavelengths, (self.minimum_wavelength, self.maximum_wavelength))

        # Extract the elements within the wavelength range
        filtered_spectral_fluxes= np.take(spectral_fluxes, np.arange(idx_range[0], idx_range[1]), axis=wavelength_axis)
        filtered_wavelengths = wavelengths[idx_range[0]:idx_range[1]]

        weighted_spectral_fluxes = self.compute_weighted_spectrum(filtered_wavelengths, filtered_spectral_fluxes)

        # Compute intgrated fluxes
        return np.trapz(weighted_spectral_fluxes, x=filtered_wavelengths, axis=wavelength_axis)

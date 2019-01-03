# -*- coding: utf-8 -*-
# This file is part of the APSimulator API.
# Author: Lars Frogner
import numpy as np
import collections
import fields


class FilterSet:
    '''
    Class for holding a set of filters and computing fluxes integrated through
    each filter.
    '''
    def __init__(self, *filter_list):
        self.filters = collections.OrderedDict([(f.label, f) for f in filter_list])
        self.n_filters = len(self.filters)

    def compute_filtered_image_field(self, image_field):

        filtered_image_field_values = np.empty((self.n_filters, *image_field.grid.shape), dtype=image_field.dtype)
        central_wavelengths = np.empty(self.n_filters)

        for idx, filter_ in enumerate(self.filters.values()):
            filtered_image_field_values[idx, :, :] = filter_.compute_integrated_flux(image_field)
            central_wavelengths[idx] = filter_.compute_central_wavelength()

        filter_labels = list(self.filters.keys())

        return fields.FilteredSpectralField(image_field.grid, central_wavelengths, filter_labels,
                                            initial_value=filtered_image_field_values,
                                            use_memmap=False,
                                            copy_initial_array=False)

    def has(self, filter_label):
        return filter_label in self.filters

    def has_all(self, filter_labels):
        return set(filter_labels).issubset(self.filters.keys())


class Filter:
    '''
    Class representing a wavelength filter.
    '''
    def __init__(self, label, minimum_wavelength, maximum_wavelength):
        self.label = str(label)
        self.minimum_wavelength = float(minimum_wavelength)
        self.maximum_wavelength = float(maximum_wavelength)

    def compute_central_wavelength(self):
        return (self.maximum_wavelength + self.minimum_wavelength)/2

    def compute_weighted_spectrum(self, wavelengths, spectrum):
        '''
        No weighting is used by default, but this can be overridden by subclasses.
        '''
        return spectrum

    def compute_integrated_flux(self, image_field):
        '''
        Integrates the given spectral fluxes weighted with the filter transmittance
        to determine the flux passing throught the filter.
        '''

        # Find index range covering the filter wavelengths
        idx_range = np.searchsorted(image_field.wavelengths, (self.minimum_wavelength, self.maximum_wavelength))

        # Extract the elements within the wavelength range
        filtered_spectral_fluxes = np.take(image_field.values, np.arange(idx_range[0], idx_range[1]), axis=0)
        filtered_wavelengths = image_field.wavelengths[idx_range[0]:idx_range[1]]

        weighted_spectral_fluxes = self.compute_weighted_spectrum(filtered_wavelengths, filtered_spectral_fluxes)

        # Compute intgrated fluxes
        integrated_fluxes = np.trapz(weighted_spectral_fluxes, x=filtered_wavelengths, axis=0)

        return integrated_fluxes

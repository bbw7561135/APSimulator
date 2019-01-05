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
        assert self.contains_only_filters(filter_list)
        self.filters = collections.OrderedDict([(filter_.label, filter_) for filter_ in filter_list])
        self.n_filters = len(self.filters)

    def add_filter(self, new_filter):
        assert isinstance(new_filter, Filter)
        self.filters[new_filter.label] = new_filter
        self.n_filters += 1

    def add_filter_on_top(self, new_filter):
        if self.is_empty():
            self.add_filter(new_filter)
        else:
            self.filters = collections.OrderedDict([(filter_.label, filter_.create_merged_filter(new_filter, filter_.label))
                                                    for filter_ in self.filters.values()])

    def compute_filtered_image_field(self, image_field):

        if self.is_empty():
            clear_filter = Filter('clear', 0, np.inf, transmittance=1)
            filtered_image_field_values = clear_filter.compute_integrated_fluxes(image_field)[np.newaxis, :, :]
            central_wavelengths = [clear_filter.get_central_wavelength()]
            filter_labels = [clear_filter.label]
        else:
            filtered_image_field_values = np.empty((self.n_filters, *image_field.grid.shape), dtype=image_field.dtype)
            central_wavelengths = np.empty(self.n_filters)

            for idx, filter_ in enumerate(self.filters.values()):
                filtered_image_field_values[idx, :, :] = filter_.compute_integrated_fluxes(image_field)
                central_wavelengths[idx] = filter_.get_central_wavelength()

            filter_labels = list(self.filters.keys())

        return fields.FilteredSpectralField(image_field.grid, central_wavelengths, filter_labels,
                                            initial_value=filtered_image_field_values,
                                            use_memmap=False,
                                            copy_initial_array=False)

    def copy(self):
        return FilterSet(*map(lambda filter_: filter_.copy(), self.filters.values()))

    def copy_with_filter_on_top(self, new_filter):
        if self.is_empty():
            return FilterSet(new_filter)
        else:
            return FilterSet(*(filter_.create_merged_filter(new_filter, filter_.label) for filter_ in self.filters.values()))

    def contains_only_filters(self, filter_list):
        return len(list(filter(lambda filter_: not isinstance(filter_, Filter), filter_list))) == 0

    def has(self, filter_label):
        return filter_label in self.filters

    def has_all(self, filter_labels):
        return set(filter_labels).issubset(self.filters.keys())

    def is_empty(self):
        return self.n_filters == 0

    def get_full_wavelength_range(self):
        if self.is_empty():
            minimum_wavelength = 0
            maximum_wavelength = np.inf
        else:
            minimum_wavelength = min([filter_.minimum_wavelength for filter_ in self.filters.values()])
            maximum_wavelength = max([filter_.maximum_wavelength for filter_ in self.filters.values()])
        return minimum_wavelength, maximum_wavelength

    def get_filter(self, filter_label):
        assert self.has(filter_label)
        return self.filters[filter_label]


class Filter:
    '''
    Class representing a wavelength filter.
    '''
    def __init__(self, label, minimum_wavelength=0, maximum_wavelength=np.inf, transmittance=1):
        self.set_label(label)
        self.set_wavelength_range(minimum_wavelength, maximum_wavelength)
        self.set_transmittance(transmittance)

    def set_label(self, label):
        self.label = str(label)

    def set_wavelength_range(self, minimum_wavelength, maximum_wavelength):
        self.minimum_wavelength = float(minimum_wavelength)
        self.maximum_wavelength = float(maximum_wavelength)

    def set_transmittance(self, transmittance):
        self.transmittance = np.asfarray(transmittance)
        if self.transmittance.ndim == 0: # Was transmittance just a number?
            self.transmittance = np.array([float(transmittance)])

    def compute_transmittances_for_wavelengths(self, wavelengths):
        transmittance_wavelengths = np.linspace(self.minimum_wavelength, self.maximum_wavelength, self.transmittance.size)
        return np.interp(wavelengths, transmittance_wavelengths, self.transmittance)

    def apply_transmittance_to_image_field(self, image_field):
        self.apply_transmittance_to_spectral_fluxes(image_field.values, image_field.wavelengths)

    def apply_transmittance_to_spectral_fluxes(self, spectral_fluxes, wavelengths):
        assert wavelengths.size == spectral_fluxes.shape[0]
        transmittance_weights = self.compute_transmittances_for_wavelengths(wavelengths)
        spectral_fluxes *= transmittance_weights[:, np.newaxis, np.newaxis]

    def compute_integrated_fluxes(self, image_field):
        '''
        Integrates the given spectral fluxes weighted with the filter transmittance
        to determine the flux passing throught the filter.
        '''

        # Find index range covering the filter wavelengths
        idx_range = np.searchsorted(image_field.wavelengths, (self.minimum_wavelength, self.maximum_wavelength))

        # Extract the elements within the wavelength range
        filtered_spectral_fluxes = np.take(image_field.values, np.arange(idx_range[0], idx_range[1]), axis=0)
        filtered_wavelengths = image_field.wavelengths[idx_range[0]:idx_range[1]]

        self.apply_transmittance_to_spectral_fluxes(filtered_spectral_fluxes, filtered_wavelengths)

        # Compute intgrated fluxes
        integrated_fluxes = np.trapz(filtered_spectral_fluxes, x=filtered_wavelengths, axis=0)

        return integrated_fluxes

    def create_merged_filter(self, other_filter, new_label):
        new_minimum_wavelength = max(self.minimum_wavelength, other_filter.minimum_wavelength)
        new_maximum_wavelength = min(self.maximum_wavelength, other_filter.maximum_wavelength)
        new_number_of_transmittances = max(self.transmittance.size, other_filter.transmittance.size)
        wavelengths = np.linspace(new_minimum_wavelength, new_maximum_wavelength, new_number_of_transmittances)
        new_transmittances = self.compute_transmittances_for_wavelengths(wavelengths)*other_filter.compute_transmittances_for_wavelengths(wavelengths)
        return Filter(new_label, new_minimum_wavelength, new_maximum_wavelength, transmittance=new_transmittances)

    def copy(self):
        return Filter(self.label, self.minimum_wavelength, self.maximum_wavelength, transmittance=self.transmittance.copy())

    def get_central_wavelength(self):
        return (self.maximum_wavelength + self.minimum_wavelength)/2

    def get_wavelength_range(self):
        return self.minimum_wavelength, self.maximum_wavelength

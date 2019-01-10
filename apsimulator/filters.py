# -*- coding: utf-8 -*-
# This file is part of the APSimulator API.
# Author: Lars Frogner
import numpy as np
import os
import collections
import fields
import math_utils
import physics_utils
import io_utils


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

    def compute_filtered_image_field(self, image_field, convert_to_photon_rates=False, use_memmap=False):
        '''
        Integrates the spectral fluxes of the image field weighted with the filter transmittance
        to determine the flux passing throught each filter. If convert_to_photon_rates=True,
        the spectral fluxes will be converted to photon rates before integrating, so that the
        integrated values correspond to the total rate of photons per area transmitted through
        the filter.
        '''

        filters = {'clear': Filter('clear', 0, np.inf, transmittance=1)} if self.is_empty() else self.filters

        central_wavelengths = [filter_.get_central_wavelength() for filter_ in filters.values()]
        filter_labels = list(filters.keys())

        filtered_image_field = fields.FilteredSpectralField(image_field.grid, central_wavelengths, filter_labels,
                                                            initial_value=None,
                                                            use_memmap=use_memmap)

        if convert_to_photon_rates:
            photons_per_energy_unit = physics_utils.compute_photons_per_energy_unit(image_field.wavelengths)[:, np.newaxis, np.newaxis]

        for idx, filter_ in enumerate(filters.values()):
            filtered_image_field.values[idx, :, :] = filter_.compute_integrated_flux(image_field.wavelengths,
                                                                                     image_field.values*photons_per_energy_unit if convert_to_photon_rates else image_field.values)

        return filtered_image_field

    def copy(self):
        return FilterSet(*map(lambda filter_: filter_.copy(), self.filters.values()))

    def copy_with_added_filter_on_top(self, new_filter):
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
    def __init__(self, label, minimum_wavelength=0, maximum_wavelength=np.inf, transmittances=1, wavelengths=None, skip_sorted_check=False):
        self.set_label(label)
        self.set_wavelength_range(minimum_wavelength, maximum_wavelength)
        self.set_transmittances(transmittances)
        self.set_wavelengths(wavelengths, skip_sorted_check=skip_sorted_check)

    def set_label(self, label):
        self.label = str(label)

    def set_wavelength_range(self, minimum_wavelength, maximum_wavelength):
        self.minimum_wavelength = float(minimum_wavelength)
        self.maximum_wavelength = float(maximum_wavelength)

    def set_transmittances(self, transmittances):
        self.transmittances = np.asfarray(transmittances)
        self.has_constant_transmittance = False
        if self.transmittances.ndim == 0: # Was transmittances just a number?
            self.transmittances = np.array([float(transmittances)]*2)
            self.has_constant_transmittance = True
        self.n_transmittances = self.transmittances.size
        assert np.max(self.transmittances) <= 1

    def set_wavelengths(self, wavelengths, skip_sorted_check=False):
        if wavelengths is None:
            if self.has_constant_transmittance:
                self.wavelengths = np.array([self.minimum_wavelength, self.maximum_wavelength])
            else:
                self.wavelengths = np.linspace(self.minimum_wavelength, self.maximum_wavelength, self.n_transmittances)
        else:
            self.wavelengths = np.asfarray(wavelengths)
            assert self.wavelengths.ndim == 1
            assert self.wavelengths.size == self.transmittances.size
            assert skip_sorted_check or math_utils.is_sorted(self.wavelengths)
            self.set_wavelength_range(self.wavelengths[0], self.wavelengths[-1])

    def compute_transmittances_for_wavelengths(self, wavelengths):
        return np.interp(wavelengths, self.wavelengths, self.transmittances, left=0, right=0)

    def apply_transmittance_to_image_field(self, image_field):
        self.apply_transmittance_to_spectral_fluxes(image_field.values, image_field.wavelengths)

    def apply_transmittance_to_spectral_fluxes(self, spectral_fluxes, wavelengths):
        assert spectral_fluxes.ndim in (1, 2, 3)
        assert wavelengths.size == spectral_fluxes.shape[0]
        transmittances = self.compute_transmittances_for_wavelengths(wavelengths)
        if spectral_fluxes.ndim == 1:
            spectral_fluxes *= transmittances
        elif spectral_fluxes.ndim == 2:
            spectral_fluxes *= transmittances[:, np.newaxis]
        else:
            spectral_fluxes *= transmittances[:, np.newaxis, np.newaxis]

    def compute_integrated_flux(self, wavelengths, spectral_fluxes):

        # Find index range covering the filter wavelengths
        idx_range = np.searchsorted(wavelengths, (self.minimum_wavelength, self.maximum_wavelength))

        # Extract the elements within the wavelength range
        filtered_spectral_fluxes = np.take(spectral_fluxes, np.arange(idx_range[0], idx_range[1]), axis=0)
        filtered_wavelengths = wavelengths[idx_range[0]:idx_range[1]]

        self.apply_transmittance_to_spectral_fluxes(filtered_spectral_fluxes, filtered_wavelengths)

        # Compute intgrated flux
        integrated_flux = np.trapz(filtered_spectral_fluxes, x=filtered_wavelengths, axis=0)

        return integrated_flux

    def create_merged_filter(self, other_filter, new_label):
        # Find new wavelenegth range
        new_minimum_wavelength = max(self.minimum_wavelength, other_filter.minimum_wavelength)
        new_maximum_wavelength = min(self.maximum_wavelength, other_filter.maximum_wavelength)

        # Find new wavelengths by merging the individual wavelength arrays
        new_wavelengths = np.unique(np.concatenate((self.wavelengths, other_filter.wavelengths)))

        # Restrict the wavelengths to the new range
        idx_range = np.searchsorted(new_wavelengths, (new_minimum_wavelength, new_maximum_wavelength))
        new_wavelengths = new_wavelengths[idx_range[0]:idx_range[1]+1]

        # Compute new transmittances by multiplying the individual transmittances
        new_transmittances = self.compute_transmittances_for_wavelengths(new_wavelengths)*other_filter.compute_transmittances_for_wavelengths(new_wavelengths)

        return Filter(new_label, transmittances=new_transmittances, wavelengths=new_wavelengths, skip_sorted_check=True)

    def copy(self):
        return Filter(self.label, transmittances=self.transmittances.copy(), wavelengths=self.wavelengths.copy(), skip_sorted_check=True)

    def get_central_wavelength(self):
        return (self.maximum_wavelength + self.minimum_wavelength)/2

    def get_wavelength_range(self):
        return self.minimum_wavelength, self.maximum_wavelength


def create_filter_from_file(input_path, filter_label=None, wavelength_scale=1e-9):

    assert os.path.isfile(input_path)

    if filter_label is None:
        # Use the filname without extension if filter label is not specified
        filter_label = io_utils.get_filename_base(input_path)

    wavelengths = []
    transmittances = []

    # Read wavelengths and transmittances from file
    with open(input_path, 'r') as f:
        for line in f:
            words = line.split()
            if len(words) != 0 and words[0][0] != '#':
                wavelengths.append(float(words[0])*wavelength_scale)
                transmittances.append(float(words[1]))

    return Filter(filter_label, transmittances=transmittances, wavelengths=wavelengths)

V_band_filter = None

def get_V_band_filter(filter_label='V'):
    global V_band_filter
    if V_band_filter is None:
        V_band_filter = create_filter_from_file(io_utils.get_path_relative_to_root('data', 'Ideal_Johnson_V.txt'), filter_label=filter_label, wavelength_scale=1e-10)
    else:
        V_band_filter.set_label(filter_label)
    return V_band_filter

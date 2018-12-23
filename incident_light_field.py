# -*- coding: utf-8 -*-
from __future__ import division
from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import math_utils
import plot_utils


def spatial_from_normalized_coordinates(normalized_x_coordinate, normalized_y_coordinate, wavelength):
    '''
    Computes spatial x- and y-coordinates from wavelength-normalized coordinates.
    '''
    return normalized_x_coordinate*wavelength, normalized_y_coordinate*wavelength


def normalized_from_spatial_coordinates(x_coordinate, y_coordinate, wavelength):
    '''
    Computes wavelength-normalized x- and y-coordinates from spatial coordinates.
    '''
    inverse_wavelength = 1/wavelength
    return x_coordinate*inverse_wavelength, y_coordinate*inverse_wavelength


#@jit(nopython=True, parallel=True)
def add_plane_waves(field_values,
                    wave_amplitudes,
                    normalized_x_coordinate_mesh, normalized_y_coordinate_mesh,
                    direction_vectors_x, direction_vectors_y):
    '''
    Takes a set of plane waves coming from point sources in various directions, and computes the
    combined light field incident on a wavelength-normalized spatial grid. The plane waves are
    assumed to have no initial phase difference, and the (extremely fast) temporal component due
    to the oscillation of the electromagnetic field is neglected.

    Arguments: Array of light field values to update     (complex[n_wavelengths, n_y_coordinates, n_x_coordinates])
               Amplitudes of the plane waves             (float[n_wavelengths, n_sources])
               Meshes of normalized x- and y-coordinates (float[n_y_coordinates, n_x_coordinates])
               x- and y-components of direction vectors  (float[n_sources])
    '''
    phase_shifts_x = np.multiply.outer(normalized_x_coordinate_mesh, direction_vectors_x) # (n_y_coordinates, n_x_coordinates, n_sources)
    phase_shifts_y = np.multiply.outer(normalized_y_coordinate_mesh, direction_vectors_y)
    total_phase_shifts = 2*np.pi*(phase_shifts_x + phase_shifts_y)
    modulation = np.cos(total_phase_shifts) - 1j*np.sin(total_phase_shifts)
    field_values[:, :, :] += np.tensordot(wave_amplitudes, modulation, axes=(1, 2))


class IncidentLightField:
    '''
    This class encodes the total electromagnetic field incident on a spatial grid as a sum of plane waves
    from different directions. A source with wave vector k is assumed to contributes with a the complex
    field value of A*exp(i*(r.k)) to total field at position r, where A is the amplitude of the plane wave.
    A^2 is taken to be the energy flux per wavelength interval [W/m^2/m] from the source. The directions are
    defined relative to direction perpendicular to the spatial grid.

    The spatial grid is normalized with wavelength, so the actual spatial position of a grid cell is
    found by multiplying its normalized coordinate with the wavelength. Thus the spatial extent and
    resolution of the grid depends on the relevant wavelength. This has the advantage of making the
    field phases independent of wavelength, so they don't have to be recomputed for each wavelength.
    If r' is the normalized position, the plane wave becomes A*exp(-2*pi*i*(r'.d)), where d is the
    unit direction vector to the source.
    '''
    def __init__(self, normalized_x_coordinates, normalized_y_coordinates, wavelengths):
        self.normalized_x_coordinates = normalized_x_coordinates # Array of x-coordinates for the incident field grid [wavelengths]
        self.normalized_y_coordinates = normalized_y_coordinates # Array of y-coordinates for the incident field grid [wavelengths]
        self.wavelengths = wavelengths # Array of wavelengths [m]

        self.n_wavelengths = len(wavelengths)
        self.n_grid_cells_x = len(normalized_x_coordinates)
        self.n_grid_cells_y = len(normalized_y_coordinates)

        # Note that 'xy'-indexing means that x corresponds to the outer dimension of the array
        self.normalized_x_coordinate_mesh, self.normalized_y_coordinate_mesh = np.meshgrid(normalized_x_coordinates,
                                                                                           normalized_y_coordinates, indexing='xy')

        # Create data cube for spectral and spatial distribution of complex incident field values [sqrt(W/m^2/m)]
        self.field_values = np.zeros((self.n_wavelengths, self.n_grid_cells_y, self.n_grid_cells_x), dtype='complex128')

        # Store different versions of the field, so that modified versions of the field can be held together with unmodified one
        self.fields = {'pure': self.field_values, 'masked': self.field_values, 'modulated': self.field_values}

    def add_point_sources(self, polar_angles, azimuth_angles, incident_spectral_fluxes, field_of_view_x, field_of_view_y):
        '''
        Arguments: Polar angles of sources [rad]                               (float[n_sources])
                   Azimuth angles of sources [rad]                             (float[n_sources])
                   Incident spectral fluxes from sources [W/m^2/m]             (float[n_wavelengths, n_sources])
                   Angular x- and y-extents covering the visible sources [rad] (float)
        '''
        n_sources = len(polar_angles)
        assert incident_spectral_fluxes.shape == (self.n_wavelengths, n_sources)

        # Create mask to select sources inside the field of view
        angular_x_coordinates, angular_y_coordinates = math_utils.angular_coordinates_from_spherical_angles(polar_angles, azimuth_angles)

        is_inside_FOV = np.logical_and(np.abs(angular_x_coordinates) <= field_of_view_x/2,
                                       np.abs(angular_y_coordinates) <= field_of_view_y/2)

        # Compute x- and y-components of the wave vectors of the plane waves
        direction_vectors_x, direction_vectors_y = math_utils.direction_vector_from_spherical_angles(polar_angles[is_inside_FOV],
                                                                                                     azimuth_angles[is_inside_FOV])

        wave_amplitudes = np.sqrt(incident_spectral_fluxes[:, is_inside_FOV]) # [sqrt(W/m^2/m)]

        add_plane_waves(self.field_values,
                        wave_amplitudes,
                        self.normalized_x_coordinate_mesh, self.normalized_y_coordinate_mesh,
                        direction_vectors_x, direction_vectors_y)

    def add_extended_source(self, angular_x_coordinates, angular_y_coordinates, incident_spectral_fluxes, field_of_view_x, field_of_view_y):
        '''
        Arguments: Angular x-coordinates for the source grid [rad]                     (float[n_x_angles])
                   Angular y-coordinates for the source grid [rad]                     (float[n_y_angles])
                   Incident spectral fluxes from source grid cells [W/m^2/m]           (float[n_wavelengths, n_y_angles, n_x_angles])
                   Angular x- and y-extents covering visible parts of the source [rad] (float)
        '''
        n_x_angles = len(angular_x_coordinates)
        n_y_angles = len(angular_y_coordinates)
        assert incident_spectral_fluxes.shape == (self.n_wavelengths, n_y_angles, n_x_angles)

        # Find index ranges to select parts of source inside the field of view
        idx_range_x = np.searchsorted(angular_x_coordinates, (-field_of_view_x/2, field_of_view_x/2)) # Note: this is lower inclusive and upper exclusive
        idx_range_y = np.searchsorted(angular_y_coordinates, (-field_of_view_y/2, field_of_view_y/2))

        # Compute x- and y-components of the wave vectors of the plane waves
        direction_vectors_x, \
            direction_vectors_y = math_utils.direction_vector_from_angular_coordinates(angular_x_coordinate_mesh[idx_range_y[0]:idx_range_y[1],
                                                                                                                 idx_range_x[0]:idx_range_x[1]],
                                                                                       angular_y_coordinate_mesh[idx_range_y[0]:idx_range_y[1],
                                                                                                                 idx_range_x[0]:idx_range_x[1]])

        wave_amplitudes = np.sqrt(incident_spectral_fluxes[:, idx_range_y[0]:idx_range_y[1], idx_range_x[0]:idx_range_x[1]]) # [sqrt(W/m^2/m)]

        add_plane_waves(self.field_values,
                        wave_amplitudes.reshape(self.n_wavelengths, -1),
                        self.normalized_x_coordinate_mesh, self.normalized_y_coordinate_mesh,
                        np.ravel(direction_vectors_x), np.ravel(direction_vectors_y))

    def apply_transmission_mask(self, transmission_mask):
        assert transmission_mask.dtype == 'bool'
        assert transmission_mask.shape == (self.n_wavelengths, self.n_grid_cells_y, self.n_grid_cells_x)
        self.fields['masked'] = self.field_values*transmission_mask
        self.fields['modulated'] = self.fields['masked'] # Reset the modulated version of the field

    def modulate(self, modulation):
        assert modulation.shape == (self.n_wavelengths, self.n_grid_cells_y, self.n_grid_cells_x)
        self.fields['modulated'] = self.fields['masked']*modulation

    def get(self, field_stage):
        assert field_stage in self.fields
        return self.fields[field_stage]

    def get_wavelengths(self):
        return self.wavelengths

    def set(self, field_values):
        assert field_values.dtype == self.field_values.dtype
        assert field_values.shape == self.field_values.shape
        self.field_values = field_values # This does not copy the data, it only modifies the reference
        self.fields['pure'] = self.field_values # Update reference to the pure version of the field
        self.fields['masked'] = self.field_values # Reset the masked version of the field
        self.fields['modulated'] = self.field_values # Reset the modulated version of the field

    def clear(self):
        self.field_values[:, :, :] = 0 + 0j
        self.fields['masked'] = self.field_values
        self.fields['modulated'] = self.field_values

    def save(self, output_path, compressed=False):
        if compressed:
            np.savez_compressed(output_path,
                                normalized_x_coordinates=self.normalized_x_coordinates,
                                normalized_y_coordinates=self.normalized_y_coordinates,
                                wavelengths=self.wavelengths,
                                field_values=self.field_values)
        else:
            np.savez(output_path,
                     normalized_x_coordinates=self.normalized_x_coordinates,
                     normalized_y_coordinates=self.normalized_y_coordinates,
                     wavelengths=self.wavelengths,
                     field_values=self.field_values)

    def visualize(self, approximate_wavelength, field_stage='modulated', output_path=None):
        '''
        Plots the amplitudes and phases of the incident light field on the spatial grid.
        '''

        # Find index of closest wavelength
        wavelength_idx = np.argmin(np.abs(self.wavelengths - approximate_wavelength))
        wavelength = self.wavelengths[wavelength_idx]

        amplitudes = np.abs(self.get(field_stage)[wavelength_idx, :, :])
        phases = np.angle(self.get(field_stage)[wavelength_idx, :, :])
        phases[amplitudes == 0] = np.nan

        first_x_coordinate, first_y_coordinate = spatial_from_normalized_coordinates(self.normalized_x_coordinates[0],
                                                                                     self.normalized_y_coordinates[0],
                                                                                     wavelength)
        last_x_coordinate, last_y_coordinate   = spatial_from_normalized_coordinates(self.normalized_x_coordinates[-1],
                                                                                     self.normalized_y_coordinates[-1],
                                                                                     wavelength)
        spatial_extent = [first_x_coordinate, last_x_coordinate, first_y_coordinate, last_y_coordinate]

        fig = plt.figure()
        left_ax = fig.add_subplot(121)
        right_ax = fig.add_subplot(122)

        amplitude_image = left_ax.imshow(amplitudes,
                                         extent=spatial_extent,
                                         origin='lower',
                                         interpolation='none',
                                         cmap=plt.get_cmap('gray'))
        left_ax.set_xlabel(r'$x$ [m]')
        left_ax.set_ylabel(r'$y$ [m]')
        left_ax.set_title(r'Incident light field amplitudes ($\lambda = {:.1f}$ nm)'.format(wavelength*1e9))

        angle_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('angle_cmap', ['black', 'white', 'black'])

        phase_image = right_ax.imshow(phases,
                                      extent=spatial_extent,
                                      origin='lower',
                                      vmin=-np.pi, vmax=np.pi,
                                      interpolation='none',
                                      cmap=angle_cmap)
        right_ax.set_xlabel(r'$x$ [m]')
        right_ax.set_ylabel(r'$y$ [m]')
        right_ax.set_title(r'Incident light field phases ($\lambda = {:.1f}$ nm)'.format(wavelength*1e9))

        # Add colorbars
        plot_utils.add_colorbar(fig, left_ax, amplitude_image, label='Amplitude [sqrt(W/m^2/m)]')
        plot_utils.add_colorbar(fig, right_ax, phase_image, label='Phase [rad]')

        plt.tight_layout()

        if output_path is None:
            plt.show()
        else:
            plt.savefig(output_path)


def load_incident_light_field(input_path):

    # Add extension if missing
    final_input_path = input_path + ('.npz' if len(input_path.split('.')) == 1 else '')

    arrays = np.load(final_input_path)
    incident_light_field = IncidentLightField(arrays['normalized_x_coordinates'],
                                              arrays['normalized_y_coordinates'],
                                              arrays['wavelengths'])
    incident_light_field.set(arrays['field_values'])
    return incident_light_field

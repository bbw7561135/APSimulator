# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math_utils
import plot_utils
import filters


def angular_from_spatial_coordinates(x_coordinates, y_coordinates, distance):
    angular_x_coordinates = np.arctan2(x_coordinates, distance)
    angular_y_coordinates = np.arctan2(y_coordinates, distance)
    return angular_x_coordinates, angular_y_coordinates


def spatial_from_angular_coordinates(angular_x_coordinates, angular_y_coordinates, distance):
    x_coordinates = distance*np.tan(angular_x_coordinates)
    y_coordinates = distance*np.tan(angular_y_coordinates)
    return x_coordinates, y_coordinates


def construct_RGB_image_array(filtered_fluxes, clipping_factor):
    filtered_flux_array = filters.construct_filtered_flux_array(filtered_fluxes, ['red', 'green', 'blue'], [0, 1, 2], color_axis=2)
    vmin = np.min(filtered_flux_array)
    vmax = np.max(filtered_flux_array)*clipping_factor
    return (filtered_flux_array - vmin)/(vmax - vmin)


class ImageLightField:

    def __init__(self, x_coordinates, y_coordinates, focal_length, wavelengths, spectral_fluxes=None):
        self.x_coordinates = x_coordinates # Array of x-coordinates for the image plane grid [m]
        self.y_coordinates = y_coordinates # Array of y-coordinates for the image plane grid [m]
        self.focal_length = focal_length # Distance of the image plane from the aperture [m]
        self.wavelengths = wavelengths # Array of wavelengths [m]

        self.n_wavelengths = len(wavelengths)
        self.n_grid_cells_x = len(x_coordinates)
        self.n_grid_cells_y = len(y_coordinates)

        self.angular_x_coordinates, self.angular_y_coordinates = angular_from_spatial_coordinates(x_coordinates,
                                                                                                  y_coordinates,
                                                                                                  focal_length)

        # Use provided spectral fluxes if present, otherwise initialize with zero
        if spectral_fluxes is None:
            # Create data cube for spectral and spatial distribution of image fluxes [W/m^2/m]
            self.spectral_fluxes = np.zeros((self.n_wavelengths, self.n_grid_cells_y, self.n_grid_cells_x), dtype='float64')
        else:
            assert spectral_fluxes.shape == (self.n_wavelengths, self.n_grid_cells_y, self.n_grid_cells_y)
            self.spectral_fluxes = spectral_fluxes

        # Store different versions of the field, so that modified versions of the field can be held together with unmodified one
        self.fields = {'pure': self.spectral_fluxes, 'convolved': self.spectral_fluxes, 'filtered': None}

    def scale_to_new_focal_length(self, new_focal_length):
        '''
        Rescales the fluxes and spatial coordinates to a different focal length.
        '''
        flux_scale = (self.focal_length/new_focal_length)**2 # Flux decreases with square of focal length

        self.spectral_fluxes *= flux_scale

        # Scale the convolved version as well unless it is just a reference to the already scaled pure version
        if self.fields['convolved'].base is not self.spectral_fluxes:
            self.fields['convolved'] *= flux_scale

        # Scale the filtered fields if present
        if self.has_filtered():
            for filter_name in self.fields['filtered']:
                self.fields['filtered'][filter_name] *= flux_scale

        # Compute new spatial coordinates
        self.x_coordinates, self.y_coordinates = spatial_from_angular_coordinates(self.angular_x_coordinates,
                                                                                  self.angular_y_coordinates,
                                                                                  new_focal_length)

        self.focal_length = new_focal_length

    def convolve(self, point_spread_function):
        assert point_spread_function.shape[0] == self.n_wavelengths
        self.fields['convolved'] = math_utils.fftconvolve(self.fields['pure'], point_spread_function, mode='same', axes=(1, 2))
        self.fields['filtered'] = None # Reset the filtered versions of the field

    def filter(self, filter_set):
        self.fields['filtered'] = filter_set.compute_filtered_fluxes(self.wavelengths, self.fields['convolved'], wavelength_axis=0)

    def has_filtered(self):
        return self.fields['filtered'] is not None

    def get(self, field_stage):
        assert field_stage in self.fields
        return self.fields[field_stage]

    def get_wavelengths(self):
        return self.wavelengths

    def get_wavelength(self, wavelength_idx):
        return self.wavelengths[wavelength_idx]

    def find_index_of_closest_wavelength(self, approximate_wavelength):
        return np.argmin(np.abs(self.wavelengths - approximate_wavelength))

    def get_spatial_coordinates(self):
        return self.x_coordinates, self.y_coordinates

    def get_angular_coordinates(self):
        return self.angular_x_coordinates, self.angular_y_coordinates

    def compute_image_plane_area(self):
        return (self.x_coordinates[-1] - self.x_coordinates[0])*(self.y_coordinates[-1] - self.y_coordinates[0])

    def compute_total_flux(self, field_stage):
        assert field_stage in self.fields
        return np.sum(self.fields[field_stage])*self.compute_image_plane_area()

    def compute_total_monochromatic_flux(self, field_stage, approximate_wavelength):
        assert field_stage in self.fields
        assert field_stage != 'filtered'
        return np.sum(self.fields[field_stage][self.find_index_of_closest_wavelength(approximate_wavelength), :, :])*self.compute_image_plane_area()

    def set(self, spectral_fluxes):
        assert spectral_fluxes.dtype == self.spectral_fluxes.dtype
        assert spectral_fluxes.shape == self.spectral_fluxes.shape
        self.spectral_fluxes = spectral_fluxes # This does not copy the data, it only modifies the reference
        self.fields['pure'] = self.spectral_fluxes # Update reference to the pure version of the field
        self.fields['convolved'] = self.spectral_fluxes # Reset the convolved version of the field
        self.fields['filtered'] = None # Reset the filtered versions of the field

    def set_stage(self, field_stage, spectral_fluxes):
        assert field_stage in self.fields
        if field_stage != 'filtered':
            assert spectral_fluxes.dtype == self.spectral_fluxes.dtype
            assert spectral_fluxes.shape == self.spectral_fluxes.shape
        self.fields[field_stage] = spectral_fluxes

    def clear(self):
        self.spectral_fluxes[:, :, :] = 0 + 0j
        self.fields['convolved'] = self.spectral_fluxes
        self.fields['filtered'] = None

    def save(self, output_path, compressed=False):
        if compressed:
            np.savez_compressed(output_path,
                                x_coordinates=self.x_coordinates,
                                y_coordinates=self.y_coordinates,
                                focal_length=self.focal_length,
                                wavelengths=self.wavelengths,
                                spectral_fluxes=self.spectral_fluxes,
                                convolved=self.fields['convolved'])
        else:
            np.savez(output_path,
                     x_coordinates=self.x_coordinates,
                     y_coordinates=self.y_coordinates,
                     focal_length=self.focal_length,
                     wavelengths=self.wavelengths,
                     spectral_fluxes=self.spectral_fluxes,
                     convolved=self.fields['convolved'])


def load_image_light_field(input_path):

    # Add extension if missing
    final_input_path = input_path + ('.npz' if len(input_path.split('.')) == 1 else '')

    arrays = np.load(final_input_path)
    image_light_field = ImageLightField(arrays['x_coordinates'],
                                        arrays['y_coordinates'],
                                        arrays['focal_length'],
                                        arrays['wavelengths'],
                                        spectral_fluxes=arrays['spectral_fluxes'])
    image_light_field.set_stage('convolved', arrays['convolved'])
    return image_light_field


def visualize_image(image_light_field, field_stage, approximate_wavelength=None, use_spatial_extent=False, clipping_factor=1, output_path=None):
    '''
    Plots the given version of the image light field.
    '''
    if approximate_wavelength is None:
        wavelength_idx = 0
        wavelength = image_light_field.get_wavelength(wavelength_idx)
    else:
        wavelength_idx = image_light_field.find_index_of_closest_wavelength(approximate_wavelength)
        wavelength = image_light_field.get_wavelength(wavelength_idx)

    if field_stage == 'filtered':
        assert image_light_field.has_filtered()
        image_values = construct_RGB_image_array(image_light_field.get(field_stage), clipping_factor)
        cmap = None
        title = 'Filtered image'
    else:
        image_values = image_light_field.get(field_stage)[wavelength_idx, :, :]
        cmap = plt.get_cmap('gray')
        title = r'Image flux ($\lambda = {:.1f}$ nm)'.format(wavelength*1e9)

    if use_spatial_extent:
        x_coordinates, y_coordinates = image_light_field.get_spatial_coordinates()
        extent = [x_coordinates[0], x_coordinates[-1],
                  y_coordinates[0], y_coordinates[-1]]
        xlabel = r'$x$ [m]'
        ylabel = r'$y$ [m]'
    else:
        angular_x_coordinates, angular_y_coordinates = image_light_field.get_angular_coordinates()
        extent = math_utils.arcsec_from_radian(np.array([angular_x_coordinates[0], angular_x_coordinates[-1],
                                                         angular_y_coordinates[0], angular_y_coordinates[-1]]))
        xlabel = r'$\alpha_x$ [arcsec]'
        ylabel = r'$\alpha_y$ [arcsec]'

    fig, ax = plt.subplots()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    image = ax.imshow(image_values,
                      extent=extent,
                      origin='lower',
                      interpolation='none',
                      cmap=cmap)

    if field_stage != 'filtered':
        plot_utils.add_colorbar(fig, ax, image, label='Flux [W/m^2/m]')

    plt.tight_layout()

    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)


def animate_image(image_light_field, field_stage, times, update_function, approximate_wavelength=None, use_spatial_extent=False, clipping_factor=1, accumulate=False, output_path=False):
    '''
    Animates the given version of the image light field.
    '''
    update_function()

    if approximate_wavelength is None:
        wavelength_idx = 0
        wavelength = image_light_field.get_wavelength(wavelength_idx)
    else:
        wavelength_idx = image_light_field.find_index_of_closest_wavelength(approximate_wavelength)
        wavelength = image_light_field.get_wavelength(wavelength_idx)

    if field_stage == 'filtered':
        assert image_light_field.has_filtered()
        image_values = construct_RGB_image_array(image_light_field.get(field_stage), clipping_factor)
        vmin = None
        vmax = None
        cmap = None
        title = 'Filtered image'
    else:
        image_values = image_light_field.get(field_stage)[wavelength_idx, :, :]
        vmin = np.min(image_values)
        vmax = np.max(image_values)
        cmap = plt.get_cmap('gray')
        title = r'Image flux ($\lambda = {:.1f}$ nm)'.format(wavelength*1e9)

    if use_spatial_extent:
        x_coordinates, y_coordinates = image_light_field.get_spatial_coordinates()
        extent = [x_coordinates[0], x_coordinates[-1],
                  y_coordinates[0], y_coordinates[-1]]
        xlabel = r'$x$ [m]'
        ylabel = r'$y$ [m]'
    else:
        angular_x_coordinates, angular_y_coordinates = image_light_field.get_angular_coordinates()
        extent = math_utils.arcsec_from_radian(np.array([angular_x_coordinates[0], angular_x_coordinates[-1],
                                                         angular_y_coordinates[0], angular_y_coordinates[-1]]))
        xlabel = r'$\alpha_x$ [arcsec]'
        ylabel = r'$\alpha_y$ [arcsec]'

    image_light_field.integrated_image_values = np.zeros(image_values.shape)

    figheight = 5
    aspect = 1.3

    fig = plt.figure(figsize=(aspect*figheight, figheight))
    ax = fig.add_subplot(111)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    image = ax.imshow(image_values,
                      extent=extent,
                      origin='lower',
                      vmin=vmin, vmax=vmax,
                      interpolation='none',
                      animated=True,
                      cmap=cmap)

    time_text = ax.text(0.01, 0.99, 'Time: {:g} s'.format(times[0]), color='white', ha='left', va='top', transform=ax.transAxes)

    plt.tight_layout(pad=1)

    n_time_steps = len(times)

    def init():
        return image, time_text,

    def update(time_idx):
        print('Frame {:d}/{:d}'.format(time_idx, n_time_steps-1))
        update_function()
        image_values = construct_RGB_image_array(image_light_field.get(field_stage), clipping_factor) \
                           if field_stage == 'filtered' else \
                       image_light_field.get(field_stage)[wavelength_idx, :, :]
        image_light_field.integrated_image_values += image_values
        image.set_array(image_light_field.integrated_image_values/(time_idx+1) if accumulate else image_values)
        time_text.set_text('Time: {:g} s'.format(times[time_idx]))

        return image, time_text,

    anim = animation.FuncAnimation(fig, update, init_func=init, blit=True, frames=np.arange(1, n_time_steps))

    if output_path:
        anim.save(output_path, writer=animation.FFMpegWriter(fps=30,
                                                             bitrate=3200,
                                                             extra_args=['-vcodec', 'libx264']))
    else:
        plt.show()

    image_light_field.integrated_image_values = None

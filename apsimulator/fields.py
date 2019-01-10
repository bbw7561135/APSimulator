# -*- coding: utf-8 -*-
# This file is part of the APSimulator API.
# Author: Lars Frogner
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import grids
import plot_utils
import image_utils
import parallel_utils


class Regular2DField:
    '''
    Represents a 2D field, specified by a regular 2D grid and an array of field values on the grid.
    '''
    def __init__(self, grid, initial_value=0, dtype='float64', use_memmap=False, copy_initial_array=True):
        self.grid = grid # Grid2D object representing the shape and physical dimensions of the field
        self.dtype = str(dtype) # Numpy data type to use for the field values
        self.use_memmap = bool(use_memmap) # Whether to store the field values in a memory mapped file
        self.copy_initial_array = bool(copy_initial_array) # If initial_value is an array (or memmap), this specifies
                                                           # whether to copy the values or use a reference

        self.initialize_shape()
        self.initialize_values(initial_value)

    def create_constructor_argument_list(self, **new_args):
        return self.grid if not 'grid' in new_args else new_args['grid'],

    def initialize_shape(self):
        self.shape = self.grid.shape
        self.window_shape = self.grid.window.shape

    def initialize_values(self, initial_value):

        has_initial_value = initial_value is not None
        inital_value_is_array = isinstance(initial_value, np.ndarray)

        if inital_value_is_array:
            assert initial_value.shape == self.shape
            self.dtype = initial_value.dtype # Change the data type to that of the initial array
        elif has_initial_value:
            initial_value = float(initial_value)

        if self.use_memmap and not (inital_value_is_array and not self.copy_initial_array):
            # Store the field values in a memory mapped file.
            # It will automatically be deleted when the memmap object goes out of scope.
            with tempfile.NamedTemporaryFile() as temporary_file:
                self.values = np.memmap(temporary_file, shape=self.shape, dtype=self.dtype, mode='w+')
            if has_initial_value:
                self.values[:] = initial_value
        else:
            if inital_value_is_array:
                self.values = initial_value.copy() if self.copy_initial_array else initial_value
            elif has_initial_value:
                # Initialize value array with constant number
                self.values = np.full(self.shape, initial_value, dtype=self.dtype)
            else:
                self.values = np.empty(self.shape, dtype=self.dtype)

    def set_constant_value(self, constant_value):
        self.values[:] = float(constant_value)

    def set_values(self, values, copy=True):
        assert isinstance(values, np.ndarray)
        assert values.shape == self.shape
        if copy:
            assert values.dtype == self.dtype
            self.values[:] = values
        else:
            self.values = values
            self.dtype = values.dtype

    def set_values_inside_window(self, values):
        '''
        Assigns the given array of values to the field values within the view window defined for
        the grid. The shape of the input array can either correspond to the full grid shape or
        just to the shape of the window (with the size of the wavelength axis being the same).
        '''
        assert isinstance(values, np.ndarray)
        assert values.shape == self.shape or values.shape == self.window_shape
        assert values.dtype == self.dtype

        window_values = self.get_values_inside_window()

        if values.shape == self.shape:
            window_values[:] = self.get_window_view_of_array(values)
        else:
            window_values[:] = values

    def __iadd__(self, values):
        '''
        Implements the += operator.
        '''
        self.values += values
        self.dtype = self.values.dtype
        return self

    def __imul__(self, values):
        '''
        Implements the *= operator.
        '''
        self.values *= values
        self.dtype = self.values.dtype
        return self

    def apply_function(self, function):
        self.values[:] = function(self.values)
        assert isinstance(self.values, np.ndarray) and \
               self.values.shape == self.shape and \
               self.values.dtype == self.dtype

    def multiply_within_window(self, factors):
        '''
        Multiplies the field values within the view window defined for the grid with the given factors.
        The shape of the input array can either correspond to the full grid shape or just to the shape
        of the window (with the size of the wavelength axis being the same)
        '''
        assert isinstance(factors, np.ndarray)
        assert factors.shape == self.shape or factors.shape == self.window_shape

        window_values = self.get_values_inside_window()

        if factors.shape == self.shape:
            window_values[:] *= self.get_window_view_of_array(factors)
        else:
            window_values[:] *= factors

    def add_within_window(self, offsets):
        '''
        Assigns the given array of offsets to the field values within the view window defined for
        the grid. The shape of the input array can either correspond to the full grid shape or
        just to the shape of the window (with the size of the wavelength axis being the same).
        '''
        assert isinstance(offsets, np.ndarray)
        assert offsets.shape == self.shape or offsets.shape == self.window_shape

        window_values = self.get_values_inside_window()

        if offsets.shape == self.shape:
            window_values[:] += self.get_window_view_of_array(offsets)
        else:
            window_values[:] += offsets

    def apply_function_within_window(self, function):
        window_values = self.get_values_inside_window()
        new_window_values = function(window_values)
        assert isinstance(new_window_values, np.ndarray) and \
               new_window_values.shape == self.window_shape and \
               new_window_values.dtype == self.dtype
        window_values[:] = new_window_values

    def copy(self, use_memmap=None):
        return self.__class__(*self.create_constructor_argument_list(),
                              initial_value=self.values,
                              use_memmap=(self.use_memmap if use_memmap is None else use_memmap),
                              copy_initial_array=True)

    def multiplied(self, factors, use_memmap=None):
        assert isinstance(factors, (int, float, complex)) or (isinstance(factors, np.ndarray) and factors.shape == self.shape)
        return self.__class__(*self.create_constructor_argument_list(),
                              initial_value=self.values*factors,
                              use_memmap=(self.use_memmap if use_memmap is None else use_memmap),
                              copy_initial_array=False)

    def added(self, offsets, use_memmap=None):
        assert isinstance(offsets, (int, float, complex)) or (isinstance(offsets, np.ndarray) and offsets.shape == self.shape)
        return self.__class__(*self.create_constructor_argument_list(),
                              initial_value=(self.values + offsets),
                              use_memmap=(self.use_memmap if use_memmap is None else use_memmap),
                              copy_initial_array=False)

    def multiplied_within_window(self, factors, use_memmap=None):
        multiplied_field = self.copy(use_memmap=use_memmap)
        multiplied_field.multiply_within_window(factors)
        return multiplied_field

    def added_within_window(self, offsets, use_memmap=None):
        offset_field = self.copy(use_memmap=use_memmap)
        offset_field.add_within_window(offsets)
        return offset_field

    def with_function_applied_within_window(self, function, use_memmap=None):
        new_field = self.copy(use_memmap=use_memmap)
        new_field.apply_function_within_window(function)
        return new_field

    def with_function_applied(self, function, use_memmap=None):
        return self.__class__(*self.create_constructor_argument_list(),
                              initial_value=function(self.values),
                              use_memmap=(self.use_memmap if use_memmap is None else use_memmap),
                              copy_initial_array=False)

    def compute_fourier_transformed_values(self, inverse=False):
        '''
        Computes the 2D Fourier transform of the field values and returns the result as an array.
        If inverse=True, the inverse inverse Fourier transform is computed instead.
        '''
        assert isinstance(self.grid, grids.FFTGrid)

    def compute_fourier_transformed_values(self, inverse=False):
        '''
        Computes the 2D Fourier transform of the field values and returns the result as an array.
        If inverse=True, the inverse inverse Fourier transform is computed instead.
        '''
        assert isinstance(self.grid, grids.FFTGrid)

        # Make sure the values to transform are not centered
        uncentered_values = np.fft.ifftshift(self.values, axes=(-2, -1)) if self.grid.is_centered else self.values

        # Perform FFT
        fourier_coefficients = np.fft.ifft2(uncentered_values, axes=(-2, -1)) if inverse else np.fft.fft2(uncentered_values, axes=(-2, -1))

        # Recenter the Fourier coefficients if necessary
        recentered_fourier_coefficients = np.fft.fftshift(fourier_coefficients, axes=(-2, -1)) if self.grid.is_centered else fourier_coefficients

        return recentered_fourier_coefficients

    def compute_fourier_transformed_window_values(self, inverse=False):
        return self.get_window_view_of_array(self.compute_fourier_transformed_values(inverse=inverse))

    def to_fourier_space(self, inverse=False, transform_grid=True, use_memmap=None):
        '''
        Constructs a new field corresponding to the Fourier transform of the current field.
        If transform_grid=True, the grid of the new field is also transformed to Fourier space.
        '''
        assert isinstance(self.grid, grids.FFTGrid)

        fourier_coefficients = self.compute_fourier_transformed_values(inverse=inverse)

        # Construct grid for the Fourier transformed field
        grid = self.grid.to_spatial_frequency_grid() if transform_grid else self.grid

        return self.__class__(*self.create_constructor_argument_list(grid=grid),
                              initial_value=fourier_coefficients,
                              use_memmap=(self.use_memmap if use_memmap is None else use_memmap),
                              copy_initial_array=False)

    def create_window_field(self, grid_type=None, use_memmap=None, copy_values=True):
        '''
        Returns a new field corresponding to the part of the current field inside
        the grid window.
        '''
        window_grid = self.grid.create_window_grid(grid_type=grid_type)
        return self.__class__(*self.create_constructor_argument_list(grid=window_grid),
                              initial_value=self.get_values_inside_window(),
                              use_memmap=(self.use_memmap if use_memmap is None else use_memmap),
                              copy_initial_array=copy_values)

    def get_window_view_of_array(self, values):
        assert values.shape == self.shape
        window = self.grid.window
        return values[window.y.start:window.y.end, window.x.start:window.x.end]

    def get_values_inside_window(self):
        return self.get_window_view_of_array(self.values)


class SpectralField(Regular2DField):
    '''
    Represents a field on a 2D grid with an added third dimension for wavelengths.
    '''
    def __init__(self, grid, wavelengths, **field_kwargs):
        self.wavelengths = np.asfarray(wavelengths) # Array of wavelengths

        assert(self.wavelengths.ndim == 1)
        self.n_wavelengths = self.wavelengths.size

        super().__init__(grid, **field_kwargs)

    def create_constructor_argument_list(self, **new_args):
        return self.grid if not 'grid' in new_args else new_args['grid'], \
               self.wavelengths if not 'wavelengths' in new_args else new_args['wavelengths']

    def initialize_shape(self):
        '''
        Overrides the corresponding Field2D method and adds an additional dimension for wavelength.
        '''
        self.shape = (self.n_wavelengths, *self.grid.shape)
        self.window_shape = (self.n_wavelengths, *self.grid.window.shape)

    def add_within_window(self, offsets):
        if isinstance(offsets, np.ndarray) and offsets.ndim == 1 and offsets.size == self.n_wavelengths:
            window_values = self.get_values_inside_window()
            window_values[:] += offsets[:, np.newaxis, np.newaxis]
        else:
            super().add_within_window(offsets)

    def compute_fourier_transformed_values(self, inverse=False):
        '''
        Computes the 2D Fourier transform of the field values and returns the result as an array.
        If inverse=True, the inverse inverse Fourier transform is computed instead.
        If n_threads > 1 in parallel_utils, the computations are parallellized over the wavelength axis.
        '''
        assert isinstance(self.grid, grids.FFTGrid)
        return parallel_utils.parallel_fft2(self.values, centered=self.grid.is_centered, inverse=inverse)

    def get_window_view_of_array(self, values):
        assert values.shape == self.shape
        window = self.grid.window
        return values[:, window.y.start:window.y.end, window.x.start:window.x.end]


class FilteredSpectralField(SpectralField):

    def __init__(self, grid, central_wavelengths, filter_labels, **field_kwargs):

        self.central_wavelengths = np.asfarray(central_wavelengths)

        super().__init__(grid, self.central_wavelengths, **field_kwargs)

        self.n_channels = self.n_wavelengths

        self.filter_labels = list(filter_labels)
        assert len(self.filter_labels) == self.central_wavelengths.size

        self.channels = {label: idx for label, idx in zip(self.filter_labels, range(self.n_channels))}

    def create_constructor_argument_list(self, **new_args):
        return self.grid if not 'grid' in new_args else new_args['grid'], \
               self.wavelengths if not 'wavelengths' in new_args else new_args['wavelengths'], \
               self.filter_labels if not 'filter_labels' in new_args else new_args['filter_labels']

    def create_field_for_channel(self, filter_label, use_memmap=None, copy_values=True):
        assert filter_label in self.channels
        return Regular2DField(self.grid,
                              initial_value=self.values[self.channels[filter_label], :, :],
                              use_memmap=(self.use_memmap if use_memmap is None else use_memmap),
                              copy_initial_array=copy_values)


def visualize_field(field, only_window=True, approximate_wavelength=None, filter_label=None, use_autostretch=False, white_point_scale=1, use_log=False, title='', output_path=None):

    use_colors = False
    was_stretched = False

    wavelength = None

    vmin = None
    vmax = None

    if only_window:
        field_values = field.get_values_inside_window()
        extent = field.grid.get_window_bounds()
    else:
        field_values = field.values
        extent = field.grid.get_bounds()

    if isinstance(field, FilteredSpectralField):
        if filter_label is None:
            use_colors = True
            use_log = False
            field_values = np.moveaxis(field_values, 0, 2)
            if not use_autostretch:
                field_values = image_utils.perform_histogram_clip(field_values, white_point_scale=white_point_scale)
                was_stretched = True
        else:
            channel_idx = field.channels[filter_label]
            field_values = field_values[channel_idx, :, :]
            wavelength = field.central_wavelengths[channel_idx]
    elif isinstance(field, SpectralField):
        wavelength_idx = 0 if approximate_wavelength is None else np.argmin(np.abs(field.wavelengths - approximate_wavelength))
        wavelength = field.wavelengths[wavelength_idx]
        field_values = field_values[wavelength_idx, :, :]

    if field.dtype == 'complex128' and not use_colors:
        phases = np.angle(field_values)
        field_values = np.abs(field_values)
        phases[field_values == 0] = np.nan
    else:
        field_values = field_values.astype('float64')

    if use_log:
        field_values = np.log10(field_values)
        log_label = 'log10 '
    else:
        log_label = ''

    if not was_stretched:
        if use_autostretch:
            field_values = image_utils.perform_autostretch(field_values)
            vmin = 0
            vmax = 1
        else:
            vmin = np.min(field_values)
            vmax = np.max(field_values)*white_point_scale

    if wavelength is None:
        wavelength = 1
    else:
        wavelength_text = '{:g} nm'.format(wavelength*1e9)
        title = wavelength_text if title == '' else '{} ({})'.format(title, wavelength_text)

    colorbar = not (use_colors or use_autostretch)

    xlabel = r'$x$'
    ylabel = r'$y$'
    phase_clabel = 'Field phase [rad]'

    if field.grid.grid_type == 'source':
        xlabel = r'$d_x$'
        ylabel = r'$d_y$'
        clabel = '{}Field amplitude [sqrt(W/m^2/m)]'.format(log_label)
    elif field.grid.grid_type == 'aperture':
        xlabel = r'$x$ [m]'
        ylabel = r'$y$ [m]'
        clabel = '{}Field amplitude [sqrt(W/m^2/m)]'.format(log_label)
        extent *= wavelength
    elif field.grid.grid_type == 'image':
        xlabel = r'$x$ [focal lengths]'
        ylabel = r'$y$ [focal lengths]'
        clabel = '{}Flux [W/m^2/m]'.format(log_label)

    fig = plt.figure()

    if field.dtype == 'complex128' and not use_colors:
        left_ax = fig.add_subplot(121)
        plot_utils.plot_image(fig, left_ax, field_values, xlabel=xlabel, ylabel=ylabel, title=title, extent=extent, clabel=clabel)
        right_ax = fig.add_subplot(122)
        plot_utils.plot_image(fig, right_ax, phases, xlabel=xlabel, ylabel=ylabel, title=title, extent=extent, clabel=phase_clabel)
    else:
        ax = fig.add_subplot(111)
        plot_utils.plot_image(fig, ax, field_values, vmin=vmin, vmax=vmax, xlabel=xlabel, ylabel=ylabel, title=title, extent=extent, clabel=clabel, colorbar=colorbar)

    plt.tight_layout()

    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)

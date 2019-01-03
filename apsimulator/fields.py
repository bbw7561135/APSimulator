# -*- coding: utf-8 -*-
# This file is part of the APSimulator API.
# Author: Lars Frogner
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import grids
import plot_utils


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

        inital_value_is_array = isinstance(initial_value, np.ndarray)

        if inital_value_is_array:
            assert initial_value.shape == self.shape
            self.dtype = initial_value.dtype # Change the data type to that of the initial array
        else:
            initial_value = float(initial_value)

        if self.use_memmap and not (inital_value_is_array and not self.copy_initial_array):
            # Store the field values in a memory mapped file.
            # It will automatically be deleted when the memmap object goes out of scope.
            with tempfile.NamedTemporaryFile() as temporary_file:
                self.values = np.memmap(temporary_file, shape=self.shape, dtype=self.dtype, mode='w+')
            self.values[:] = initial_value
        else:
            if inital_value_is_array:
                self.values = initial_value.copy() if self.copy_initial_array else initial_value
            else:
                # Initialize value array with constant number
                self.values = np.full(self.shape, initial_value, dtype=self.dtype)

    def window_view(self, values):
        assert values.shape == self.shape
        window = self.grid.window
        return values[window.y.start:window.y.end, window.x.start:window.x.end]

    def get_values_inside_window(self):
        return self.window_view(self.values)

    def set_constant_value(self, constant_value):
        self.values[:] = float(constant_value)

    def set_values(self, values, copy=True):
        assert isinstance(values, np.ndarray)
        assert values.shape == self.shape
        assert values.dtype == self.dtype
        if copy:
            self.values[:] = values
        else:
            self.values = values

    def __iadd__(self, values):
        '''
        Implements the += operator.
        '''
        assert values.shape == self.shape
        self.values += values
        self.dtype = self.values.dtype
        return self

    def __imul__(self, values):
        '''
        Implements the *= operator.
        '''
        assert values.shape == self.shape
        self.values *= values
        self.dtype = self.values.dtype
        return self

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
            window_values[:] = self.window_view(values)
        else:
            window_values[:] = values

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
            window_values[:] *= self.window_view(factors)
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
            window_values[:] += self.window_view(offsets)
        else:
            window_values[:] += offsets

    def copy(self, use_memmap=None):
        return self.__class__(*self.create_constructor_argument_list(),
                              initial_value=self.values,
                              use_memmap=(self.use_memmap if use_memmap is None else use_memmap),
                              copy_initial_array=True)

    def multiplied(self, factors, use_memmap=None):
        assert isinstance(factors, np.ndarray)
        assert factors.shape == self.shape
        return self.__class__(*self.create_constructor_argument_list(),
                              initial_value=self.values*factors,
                              use_memmap=(self.use_memmap if use_memmap is None else use_memmap),
                              copy_initial_array=False)

    def added(self, offsets, use_memmap=None):
        assert isinstance(offsets, np.ndarray)
        assert offsets.shape == self.shape
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

    def fourier_transformed_values(self, inverse=False):
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

    def fourier_transformed_values_inside_window(self, inverse=False):
        return self.window_view(self.fourier_transformed_values(inverse=inverse))

    def to_fourier_space(self, inverse=False, transform_grid=True, use_memmap=None):
        '''
        Constructs a new field corresponding to the Fourier transform of the current field.
        If transform_grid=True, the grid of the new field is also transformed to Fourier space.
        '''
        assert isinstance(self.grid, grids.FFTGrid)

        fourier_coefficients = self.fourier_transformed_values(inverse=inverse)

        # Construct grid for the Fourier transformed field
        grid = self.grid.to_spatial_frequency_grid() if transform_grid else self.grid

        return self.__class__(*self.create_constructor_argument_list(grid=grid),
                              initial_value=fourier_coefficients,
                              use_memmap=(self.use_memmap if use_memmap is None else use_memmap),
                              copy_initial_array=False)

    def construct_window_field(self, grid_type=None, use_memmap=None, copy_values=True):
        '''
        Returns a new field corresponding to the part of the current field inside
        the grid window.
        '''
        window_grid = self.grid.construct_window_grid(grid_type=grid_type)
        return self.__class__(*self.create_constructor_argument_list(grid=window_grid),
                              initial_value=self.get_values_inside_window(),
                              use_memmap=(self.use_memmap if use_memmap is None else use_memmap),
                              copy_initial_array=copy_values)


class SpectralField(Regular2DField):
    '''
    Represents a field on a 2D grid with an added third dimension for wavelengths.
    '''
    def __init__(self, grid, wavelengths, **field_kwargs):
        self.wavelengths = np.asfarray(wavelengths) # Array of wavelengths

        self.n_wavelengths = len(self.wavelengths)
        assert(self.wavelengths.ndim == 1)

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

    def window_view(self, values):
        assert values.shape == self.shape
        window = self.grid.window
        return values[:, window.y.start:window.y.end, window.x.start:window.x.end]


class FilteredSpectralField(SpectralField):

    def __init__(self, grid, central_wavelengths, filter_labels, **field_kwargs):

        self.central_wavelengths = np.asfarray(central_wavelengths)

        super().__init__(grid, self.central_wavelengths, **field_kwargs)

        self.n_channels = self.n_wavelengths

        self.filter_labels = list(filter_labels)
        assert len(self.filter_labels) == len(self.central_wavelengths)

        self.channels = {label: idx for label, idx in zip(self.filter_labels, range(self.n_channels))}

    def create_constructor_argument_list(self, **new_args):
        return self.grid if not 'grid' in new_args else new_args['grid'], \
               self.wavelengths if not 'wavelengths' in new_args else new_args['wavelengths'], \
               self.filter_labels if not 'filter_labels' in new_args else new_args['filter_labels']

    def construct_field_for_channel(self, filter_label, use_memmap=None, copy_values=True):
        assert filter_label in self.channels
        return Regular2DField(self.grid,
                              initial_value=self.values[self.channels[filter_label], :, :],
                              use_memmap=(self.use_memmap if use_memmap is None else use_memmap),
                              copy_initial_array=copy_values)


def visualize_field(field, only_window=True, approximate_wavelength=None, filter_label=None, clipping_factor=1, use_log=False, title='', output_path=None):

    if only_window:
        field_values = field.get_values_inside_window()
        extent = field.grid.get_window_bounds()
    else:
        field_values = field.values
        extent = field.grid.get_bounds()

    use_colors = False
    wavelength = None

    vmin = None
    vmax = None

    if isinstance(field, FilteredSpectralField):
        if filter_label is None:
            use_colors = True
            vmin = np.min(field_values)
            vmax = np.max(field_values)*clipping_factor
            field_values = (field_values - vmin)/(vmax - vmin)
            field_values = np.moveaxis(field_values, 0, 2)
            clipping_factor = None
            use_log = False
        else:
            channel_idx = field.channels[filter_label]
            field_values = field_values[channel_idx, :, :]
            wavelength = field.central_wavelengths[channel_idx]
    elif isinstance(field, SpectralField):
        wavelength_idx = 0 if approximate_wavelength is None else np.argmin(np.abs(field.wavelengths - approximate_wavelength))
        wavelength = field.wavelengths[wavelength_idx]
        field_values = field_values[wavelength_idx, :, :]

    if clipping_factor is not None:
        vmin = np.min(field_values)
        vmax = np.max(field_values)*clipping_factor

    log_label = 'log10 ' if use_log else ''

    if wavelength is None:
        wavelength = 1
    else:
        wavelength_text = '{:g} nm'.format(wavelength*1e9)
        title = wavelength_text if title == '' else '{} ({})'.format(title, wavelength_text)

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
        amplitudes = np.abs(field_values)
        amplitudes = np.log10(amplitudes) if use_log else amplitudes
        plot_utils.plot_image(fig, left_ax, amplitudes, xlabel=xlabel, ylabel=ylabel, title=title, extent=extent, clabel=clabel)

        right_ax = fig.add_subplot(122)
        phases = np.angle(field_values)
        phases[amplitudes == 0] = np.nan
        plot_utils.plot_image(fig, right_ax, phases, xlabel=xlabel, ylabel=ylabel, title=title, extent=extent, clabel=phase_clabel)

    else:
        ax = fig.add_subplot(111)
        field_values = field_values.astype('float64')
        field_values = np.log10(field_values) if use_log else field_values
        plot_utils.plot_image(fig, ax, field_values, vmin=vmin, vmax=vmax, xlabel=xlabel, ylabel=ylabel, title=title, extent=extent, clabel=clabel, colorbar=(not use_colors))

    plt.tight_layout()

    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)

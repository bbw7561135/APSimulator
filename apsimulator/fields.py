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
        self.copy_initial_array = bool(copy_initial_array) # If initial_value is an array, this specifies
                                                           # whether to copy the values or use a reference

        self.initialize_shape()
        self.initialize_values(initial_value)

    def initialize_shape(self):
        self.shape = self.grid.shape

    def initialize_values(self, initial_value):

        inital_value_is_array = isinstance(initial_value, np.ndarray)

        if inital_value_is_array:
            assert initial_value.shape == self.shape
            self.dtype = initial_value.dtype # Change the data type to that of the initial array
        else:
            initial_value = float(initial_value)

        if self.use_memmap:
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


class SpectralField(Regular2DField):
    '''
    Represents a field on a 2D grid with an added third dimension for wavelengths.
    '''
    def __init__(self, grid, wavelengths, initial_value=0, dtype='float64', use_memmap=False, copy_initial_array=True):
        self.wavelengths = np.asfarray(wavelengths) # Array of wavelengths

        self.n_wavelengths = len(self.wavelengths)
        assert(self.wavelengths.ndim == 1)

        super().__init__(grid, initial_value=initial_value, dtype=dtype,
                         use_memmap=use_memmap, copy_initial_array=copy_initial_array)

    def initialize_shape(self):
        '''
        Overrides the corresponding Field2D method and adds an additional dimension for wavelength.
        '''
        self.shape = (self.n_wavelengths, *self.grid.shape)

    def set_values_inside_window(self, values):
        '''
        Assigns the given array of values to the field values within the view window defined for
        the grid. The shape of the input array can either correspond to the full grid shape or
        just to the shape of the window (with the size of the wavelength axis being the same).
        '''
        assert isinstance(values, np.ndarray)
        assert values.shape == self.shape or values.shape == (self.n_wavelengths, *self.grid.window.shape)
        assert values.dtype == self.dtype

        window = self.grid.window

        if values.shape == self.shape:
            self.values[:, window.y.start:window.y.end,
                           window.x.start:window.x.end] = values[:, window.y.start:window.y.end,
                                                                    window.x.start:window.x.end]
        else:
            self.values[:, window.y.start:window.y.end,
                           window.x.start:window.x.end] = values

    def multiply_within_window(self, factors):
        '''
        Multiplies the field values within the view window defined for the grid with the given factors.
        The shape of the input array can either correspond to the full grid shape or just to the shape
        of the window (with the size of the wavelength axis being the same)
        '''
        assert isinstance(factors, np.ndarray)
        assert factors.shape == self.shape or factors.shape == (self.n_wavelengths, *self.grid.window.shape)

        window = self.grid.window

        if factors.shape == self.shape:
            self.values[:, window.y.start:window.y.end,
                           window.x.start:window.x.end] *= factors[:, window.y.start:window.y.end,
                                                                      window.x.start:window.x.end]
        else:
            self.values[:, window.y.start:window.y.end,
                           window.x.start:window.x.end] *= factors

    def add_within_window(self, offsets):
        '''
        Assigns the given array of offsets to the field values within the view window defined for
        the grid. The shape of the input array can either correspond to the full grid shape or
        just to the shape of the window (with the size of the wavelength axis being the same).
        '''
        assert isinstance(offsets, np.ndarray)
        assert offsets.shape == self.shape or offsets.shape == (self.n_wavelengths, *self.grid.window.shape)

        window = self.grid.window

        if offsets.shape == self.shape:
            self.values[:, window.y.start:window.y.end,
                           window.x.start:window.x.end] += offsets[:, window.y.start:window.y.end,
                                                                      window.x.start:window.x.end]
        else:
            self.values[:, window.y.start:window.y.end,
                           window.x.start:window.x.end] += offsets

    def copy(self, use_memmap=None):
        return SpectralField(self.grid, self.wavelengths, initial_value=self.values,
                             use_memmap=(self.use_memmap if use_memmap is None else use_memmap),
                             copy_initial_array=True)

    def multiplied(self, factors, use_memmap=None):
        assert isinstance(factors, np.ndarray)
        assert factors.shape == self.shape
        return SpectralField(self.grid, self.wavelengths, initial_value=self.values*factors,
                             use_memmap=(self.use_memmap if use_memmap is None else use_memmap),
                             copy_initial_array=False)

    def added(self, offsets, use_memmap=None):
        assert isinstance(offsets, np.ndarray)
        assert offsets.shape == self.shape
        return SpectralField(self.grid, self.wavelengths, initial_value=(self.values + offsets),
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
        # Make sure the values to transform are not centered
        uncentered_values = np.fft.ifftshift(self.values, axes=(1, 2)) if self.grid.is_centered else self.values

        # Perform FFT
        fourier_coefficients = np.fft.ifft2(uncentered_values, axes=(1, 2)) if inverse else np.fft.fft2(uncentered_values, axes=(1, 2))

        # Recenter the Fourier coefficients if necessary
        recentered_fourier_coefficients = np.fft.fftshift(fourier_coefficients, axes=(1, 2)) if self.grid.is_centered else fourier_coefficients

        return recentered_fourier_coefficients

    def fourier_transformed_values_inside_window(self, inverse=False):
        window = self.grid.window
        return self.fourier_transformed_values(inverse=inverse)[:, window.y.start:window.y.end, window.x.start:window.x.end]

    def to_fourier_space(self, inverse=False, transform_grid=True, use_memmap=None):
        '''
        Constructs a new field corresponding to the Fourier transform of the current field.
        If transform_grid=True, the grid of the new field is also transformed to Fourier space.
        '''
        assert isinstance(self.grid, grids.FFTGrid)

        fourier_coefficients = self.fourier_transformed_values(inverse=inverse)

        # Construct grid for the Fourier transformed field
        grid = self.grid.to_spatial_frequency_grid() if transform_grid else self.grid

        return SpectralField(grid, self.wavelengths, initial_value=fourier_coefficients,
                             use_memmap=(self.use_memmap if use_memmap is None else use_memmap),
                             copy_initial_array=False)

    def get_values_inside_window(self):
        window = self.grid.window
        return self.values[:, window.y.start:window.y.end, window.x.start:window.x.end]


def visualize_field(field, only_window=True, approximate_wavelength=None, use_log=False, title='', output_path=None):

    if only_window:
        field_values = field.get_values_inside_window()
        extent = field.grid.get_window_bounds()
    else:
        field_values = field.values
        extent = field.grid.get_bounds()

    if isinstance(field, SpectralField):
        wavelength_idx = 0 if approximate_wavelength is None else np.argmin(np.abs(field.wavelengths - approximate_wavelength))
        wavelength = field.wavelengths[wavelength_idx]
        field_values = field_values[wavelength_idx, :, :]
    else:
        wavelength = 1

    log_label = 'log10 ' if use_log else ''

    xlabel = r'$x$'
    ylabel = r'$y$'

    if field.grid.grid_type == 'source':
        xlabel = r'$d_x$'
        ylabel = r'$d_y$'
        clabel = '{}Field amplitude [sqrt(W/m^2/m)]'.format(log_label)
    elif field.grid.grid_type == 'aperture':
        xlabel = r'$x$ [m]'
        ylabel = r'$y$ [m]'
        clabel = '{}Field amplitude [sqrt(W/m^2/m)]'.format(log_label)
        phase_clabel = 'Field phase [rad]'
        extent *= wavelength
    elif field.grid.grid_type == 'image':
        xlabel = r'$x$ [focal lengths]'
        ylabel = r'$y$ [focal lengths]'
        clabel = '{}Flux [W/m^2/m]'.format(log_label)

    fig = plt.figure()

    if field.grid.grid_type == 'aperture' and field.dtype == 'complex128':

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
        plot_utils.plot_image(fig, ax, field_values, xlabel=xlabel, ylabel=ylabel, title=title, extent=extent, clabel=clabel)

    plt.tight_layout()

    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)

# -*- coding: utf-8 -*-
# This file is part of the APSimulator API.
# Author: Lars Frogner
import numpy as np


class IndexRange:
    def __init__(self, start, end):
        self.start = int(start)
        self.end = int(end)


class IndexRange2D:
    def __init__(self, x_start, x_end, y_start, y_end):
        self.x = IndexRange(x_start, x_end)
        self.y = IndexRange(y_start, y_end)
        self.size_x = x_end - x_start
        self.size_y = y_end - y_start
        self.shape = (self.size_y, self.size_x)


class Regular2DGrid:
    '''
    Represents a 2D grid defined by a size (number of grid cells) and extent in each direction.
    '''
    def __init__(self, size_x, size_y, extent_x, extent_y, shift_x=0, shift_y=0, grid_type=None):
        self.size_x = int(size_x) # Number of grid cells in the x-direction
        self.size_y = int(size_y) # Number of grid cells in the y-direction
        self.extent_x = float(extent_x) # Extent of the grid in the x-direction
        self.extent_y = float(extent_y) # Extent of the grid in the y-direction
        self.shift_x = int(shift_x) # Number of cells to shift the origin of the grid in the x-direction
        self.shift_y = int(shift_y) # Number of cells to shift the origin of the grid in the y-direction
        self.grid_type = grid_type # Identifier allowing for distinguishing between different grid types

        self.initialize_cell_numbers()
        self.initialize_cell_extents()
        self.initialize_coordinates()
        self.initialize_window()

    def initialize_cell_numbers(self):
        '''
        Computes integer grid coordinates.
        '''
        self.cell_numbers_x = np.arange(-self.shift_x, self.size_x - self.shift_x)
        self.cell_numbers_y = np.arange(-self.shift_y, self.size_y - self.shift_y)

    def initialize_cell_extents(self):
        self.cell_extent_x = self.extent_x/self.size_x
        self.cell_extent_y = self.extent_y/self.size_y

    def initialize_coordinates(self):
        self.x_coordinates = self.cell_numbers_x*self.cell_extent_x
        self.y_coordinates = self.cell_numbers_y*self.cell_extent_y
        self.x_coordinate_mesh, self.y_coordinate_mesh = np.meshgrid(self.x_coordinates, self.y_coordinates, indexing='xy', copy=False)
        self.shape = (self.size_y, self.size_x)

    def initialize_window(self):
        '''
        The grid window is a range of indices can be used to operate on a limited segment
        of the associated field.
        '''
        self.window = IndexRange2D(0, self.size_x, 0, self.size_y)

    def create_scaled_grid(self, scale, grid_type=None):
        '''
        Returns a new grid with scaled extents. By default the grid type identifier is copied
        to the new grid, but a new identifier can be specified by setting the grid_type argument.
        '''
        return Regular2DGrid(self.size_x, self.size_y, self.extent_x*scale, self.extent_y*scale,
                             shift_x=self.shift_x, shift_y=self.shift_y,
                             grid_type=(self.grid_type if grid_type is None else grid_type))

    def find_index_ranges(self, x_coordinate_range, y_coordinate_range):
        '''
        Finds the index ranges corresponding the given coordinate ranges.
        The indec ranges are lower inclusive and upper exclusive.
        '''
        assert len(x_coordinate_range) == 2
        assert len(y_coordinate_range) == 2
        x_index_range = np.searchsorted(self.x_coordinates, (x_coordinate_range[0], x_coordinate_range[1]))
        y_index_range = np.searchsorted(self.y_coordinates, (y_coordinate_range[0], y_coordinate_range[1]))
        return x_index_range, y_index_range

    def define_window(self, x_coordinate_range, y_coordinate_range):
        '''
        Creates a new grid window corresponding to the given coordinate ranges.
        '''
        x_index_range, y_index_range = self.find_index_ranges(x_coordinate_range, y_coordinate_range)
        self.window = IndexRange2D(x_index_range[0], x_index_range[1],
                                   y_index_range[0], y_index_range[1])

    def create_window_grid(self, grid_type=None):
        '''
        Returns a new grid corresponding to the part of the current grid inside
        the grid window.
        '''
        window_extent_x, window_extent_y = self.get_window_extents()
        return Regular2DGrid(self.window.size_x, self.window.size_y, window_extent_x, window_extent_y,
                             shift_x=self.shift_x - (self.size_x - self.window.size_x)//2,
                             shift_y=self.shift_y - (self.size_y - self.window.size_y)//2,
                             grid_type=(self.grid_type if grid_type is None else grid_type))

    def get_bounds(self, x_index_range=(0, -1), y_index_range=(0, -1)):
        '''
        Finds the coordinate ranges corresponding to the given index ranges.
        This is returned in a 4-element array for convenient use with the
        extent argument in matplotlib's imshow.
        '''
        assert len(x_index_range) == 2
        assert len(y_index_range) == 2
        return np.array([self.x_coordinates[x_index_range[0]], self.x_coordinates[x_index_range[1]-1],
                         self.y_coordinates[y_index_range[0]], self.y_coordinates[y_index_range[1]-1]])

    def get_coordinate_meshes(self):
        return self.x_coordinate_mesh, self.y_coordinate_mesh

    def get_coordinate_meshes_within_window(self):
        return self.x_coordinate_mesh[self.window.y.start:self.window.y.end,
                                      self.window.x.start:self.window.x.end], \
               self.y_coordinate_mesh[self.window.y.start:self.window.y.end,
                                      self.window.x.start:self.window.x.end]

    def get_window_bounds(self):
        return self.get_bounds(x_index_range=(self.window.x.start, self.window.x.end),
                               y_index_range=(self.window.y.start, self.window.y.end))

    def get_window_extents(self):
        return self.window.size_x*self.cell_extent_x, self.window.size_y*self.cell_extent_y

    def get_total_size(self):
        return self.size_x*self.size_y

    def get_total_window_size(self):
        return self.window.size_x*self.window.size_y

    def get_area(self):
        return self.extent_x*self.extent_y

    def get_cell_area(self):
        return self.cell_extent_x*self.cell_extent_y

    def compute_squared_distances(self):
        return self.x_coordinate_mesh**2 + self.y_coordinate_mesh**2

    def compute_squared_distances_within_window(self):
        x_coordinate_mesh_within_window, y_coordinate_mesh_within_window = self.get_coordinate_meshes_within_window()
        return x_coordinate_mesh_within_window**2 + y_coordinate_mesh_within_window**2

    def compute_distances(self):
        return np.sqrt(self.compute_squared_distances())

    def compute_distances_within_window(self):
        return np.sqrt(self.compute_squared_distances_within_window())


class FFTGrid(Regular2DGrid):
    '''
    Version if the 2D regular grid that is designed for use with Fast Fourier Transforms.
    The grid sizes are restricted to powers of two, resulting in the fastet possible FFTs.
    '''
    def __init__(self, size_exponent_x, size_exponent_y, extent_x, extent_y, is_centered=True, grid_type=None):

        self.size_exponent_x = int(size_exponent_x) # How many powers of two for size_x
        self.size_exponent_y = int(size_exponent_y) # How many powers of two for size_y
        self.is_centered = bool(is_centered) # Whether the origin of the grid should be put in the center

        size_x = 2**int(self.size_exponent_x)
        size_y = 2**int(self.size_exponent_y)

        self.half_size_x = size_x//2
        self.half_size_y = size_y//2

        shift_x = self.half_size_x if self.is_centered else 0
        shift_y = self.half_size_y if self.is_centered else 0

        super().__init__(size_x, size_y, extent_x, extent_y,
                         shift_x=shift_x, shift_y=shift_y,
                         grid_type=grid_type)

    def create_scaled_grid(self, scale, grid_type=None):
        return self.__class__(self.size_exponent_x, self.size_exponent_y,
                              self.extent_x*scale, self.extent_y*scale,
                              is_centered=self.is_centered,
                              grid_type=(self.grid_type if grid_type is None else grid_type))

    def create_centered_grid(self):
        return self.__class__(self.size_exponent_x, self.size_exponent_y,
                              self.extent_x, self.extent_y,
                              is_centered=True)

    def create_uncentered_grid(self):
        return self.__class__(self.size_exponent_x, self.size_exponent_y,
                              self.extent_x, self.extent_y,
                              is_centered=False)

    def to_spatial_frequency_grid(self, grid_type=None):
        '''
        Returns a new grid where the coordinates correspond to the spatial frequencies
        of the current coordinates.
        '''
        frequency_extent_x = 1/self.cell_extent_x
        frequency_extent_y = 1/self.cell_extent_y
        return FFTGrid(self.size_exponent_x, self.size_exponent_y, frequency_extent_x, frequency_extent_y,
                       is_centered=self.is_centered,
                       grid_type=(self.grid_type if grid_type is None else grid_type))

# -*- coding: utf-8 -*-
import numpy as np


class IndexRange:
    def __init__(self, start, end):
        self.start = int(start)
        self.end = int(end)


class IndexRange2D:
    def __init__(self, x_start, x_end, y_start, y_end):
        self.x = IndexRange(x_start, x_end)
        self.y = IndexRange(y_start, y_end)
        self.shape = (y_end - y_start, x_end - x_start)


class RegularGrid:

    def __init__(self, size_x, size_y, extent_x, extent_y, center_index_x=0, center_index_y=0, grid_type=None):
        self.size_x = int(size_x)
        self.size_y = int(size_y)
        self.extent_x = float(extent_x)
        self.extent_y = float(extent_y)
        self.center_index_x = int(center_index_x)
        self.center_index_y = int(center_index_y)
        self.grid_type = grid_type

        self.initialize_cell_numbers()
        self.initialize_cell_extents()
        self.initialize_coordinates()
        self.initialize_window()

    def initialize_cell_numbers(self):
        self.cell_numbers_x = np.arange(-self.center_index_x, self.size_x - self.center_index_x)
        self.cell_numbers_y = np.arange(-self.center_index_y, self.size_y - self.center_index_y)

    def initialize_cell_extents(self):
        self.cell_extent_x = self.extent_x/self.size_x
        self.cell_extent_y = self.extent_y/self.size_y

    def initialize_coordinates(self):
        self.x_coordinates = self.cell_numbers_x*self.cell_extent_x
        self.y_coordinates = self.cell_numbers_y*self.cell_extent_y
        self.x_coordinate_mesh, self.y_coordinate_mesh = np.meshgrid(self.x_coordinates, self.y_coordinates, indexing='xy', copy=False)
        self.shape = (self.size_y, self.size_x)

    def initialize_window(self):
        self.window = IndexRange2D(0, self.size_x, 0, self.size_y)

    def scaled(self, scale, grid_type=None):
        return RegularGrid(self.size_x, self.size_y, self.extent_x*scale, self.extent_y*scale,
                           center_index_x=self.center_index_x, center_index_y=self.center_index_y,
                           grid_type=(self.grid_type if grid_type is None else grid_type))

    def get_index_ranges(self, x_coordinate_range, y_coordinate_range):
        assert len(x_coordinate_range) == 2
        assert len(y_coordinate_range) == 2
        x_index_range = np.searchsorted(self.x_coordinates, (x_coordinate_range[0], x_coordinate_range[1]))
        y_index_range = np.searchsorted(self.y_coordinates, (y_coordinate_range[0], y_coordinate_range[1]))
        return x_index_range, y_index_range

    def define_window(self, x_coordinate_range, y_coordinate_range):
        x_index_range, y_index_range = self.get_index_ranges(x_coordinate_range, y_coordinate_range)
        self.window = IndexRange2D(x_index_range[0], x_index_range[1],
                                   y_index_range[0], y_index_range[1])

    def get_bounds(self, x_index_range=(0, -1), y_index_range=(0, -1)):
        assert len(x_index_range) == 2
        assert len(y_index_range) == 2
        return np.array([self.x_coordinates[x_index_range[0]], self.x_coordinates[x_index_range[1]],
                         self.y_coordinates[y_index_range[0]], self.y_coordinates[y_index_range[1]]])

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

    def compute_squared_distances(self):
        return self.x_coordinate_mesh**2 + self.y_coordinate_mesh**2

    def compute_squared_distances_within_window(self):
        x_coordinate_mesh_within_window, y_coordinate_mesh_within_window = self.get_coordinate_meshes_within_window()
        return x_coordinate_mesh_within_window**2 + y_coordinate_mesh_within_window**2

    def compute_distances(self):
        return np.sqrt(self.compute_squared_distance_mesh())

    def compute_distances_within_window(self):
        return np.sqrt(self.compute_squared_distances_within_window())

    def compute_total_size(self):
        return self.size_x*self.size_y

    def compute_total_window_size(self):
        return self.window.shape[0]*self.window.shape[1]

    def compute_area(self):
        return self.extent_x*self.extent_y

    def compute_cell_area(self):
        return self.cell_extent_x*self.cell_extent_y


class PowerOfTwoGrid(RegularGrid):

    def __init__(self, size_exponent_x, size_exponent_y, extent_x, extent_y, is_centered=True, grid_type=None):

        self.size_exponent_x = int(size_exponent_x)
        self.size_exponent_y = int(size_exponent_y)

        size_x = 2**int(self.size_exponent_x)
        size_y = 2**int(self.size_exponent_y)

        self.half_size_x = size_x//2
        self.half_size_y = size_y//2

        self.is_centered = bool(is_centered)

        center_index_x = self.half_size_x if self.is_centered else 0
        center_index_y = self.half_size_y if self.is_centered else 0

        super().__init__(size_x, size_y, extent_x, extent_y,
                         center_index_x=center_index_x, center_index_y=center_index_y,
                         grid_type=grid_type)

        def scaled(self, scale, grid_type=None):
            return self.__class__(self.size_exponent_x, self.size_exponent_y,
                                  self.extent_x*scale, self.extent_y*scale,
                                  is_centered=self.is_centered,
                                  grid_type=(self.grid_type if grid_type is None else grid_type))

        def centered(self):
            return self.__class__(self.size_exponent_x, self.size_exponent_y,
                                  self.extent_x, self.extent_y,
                                  is_centered=True)

        def uncentered(self):
            return self.__class__(self.size_exponent_x, self.size_exponent_y,
                                  self.extent_x, self.extent_y,
                                  is_centered=False)


class FFTGrid(PowerOfTwoGrid):

    def __init__(self, *grid_args, **grid_kwargs):
        super().__init__(*grid_args, **grid_kwargs)

    def to_spatial_frequency_grid(self, grid_type=None):
        frequency_extent_x = 1/self.cell_extent_x
        frequency_extent_y = 1/self.cell_extent_y
        return FFTGrid(self.size_exponent_x, self.size_exponent_y, frequency_extent_x, frequency_extent_y,
                       is_centered=self.is_centered,
                       grid_type=(self.grid_type if grid_type is None else grid_type))

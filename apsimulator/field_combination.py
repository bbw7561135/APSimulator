# -*- coding: utf-8 -*-
import numpy as np
import fields


class FieldGenerator:

    def initialize_spectral_grid(self, grid, wavelengths):
        self.grid = grid
        self.wavelengths = wavelengths
        self.n_wavelengths = len(self.wavelengths)

    def apply(self, combined_field):
        # combined_field.val += ...
        raise NotImplementedError

    def generate_field(self, use_memmap=False):
        # self.generated_field = ...
        raise NotImplementedError

    def apply_generated_field(self, combined_field):
        # combined_field.[multiply/add]_within_window(self.generated_field.values)
        raise NotImplementedError

    def visualize_generated_field(self, **plot_kwargs):
        fields.visualize_field(self.generated_field, **plot_kwargs)


class FieldCombinerItem:

    def __init__(self, field_generator, keep_field):
        self.field_generator = field_generator
        self.keep_field = bool(keep_field)

        self.has_initialized_grid = False
        self.has_generated_field = False


class FieldCombiner:

    def __init__(self, initial_field_value=0, dtype='float64'):
        self.initial_field_value = float(initial_field_value)
        self.dtype = str(dtype)

        self.items = {}
        self.has_combined_field = False

    def initialize_combined_field(self, grid, wavelengths, use_memmap=False):
        self.combined_field = fields.SpectralField(grid, wavelengths,
                                                   initial_value=self.initial_field_value,
                                                   dtype=self.dtype,
                                                   use_memmap=use_memmap)
        self.has_combined_field = True

    def add(self, name, field_generator, keep_field=False):
        assert not self.has(name)
        assert isinstance(field_generator, FieldGenerator)
        name = str(name)
        self.items[name] = FieldCombinerItem(field_generator, keep_field)

    def initialize_spectral_grid(self, item):
        item.field_generator.initialize_spectral_grid(self.combined_field.grid, self.combined_field.wavelengths)
        item.has_initialized_grid = True
        item.has_generated_field = False

    def generate_field(self, item):
        item.field_generator.generate_field()
        item.has_generated_field = True

    def compute_combined_field(self):
        '''
        Computes the combined field for all the field generators.
        '''
        self.combined_field.set_constant_value(self.initial_field_value)

        for item in self.items.values():

            # Initialize the spectral grid for the generator if necessary
            if not item.has_initialized_grid:
                self.initialize_spectral_grid(item)

            if item.keep_field:
                # Generate a field if it does not already exist, and apply it to the combined field
                if not item.has_generated_field:
                    self.generate_field(item)
                item.field_generator.apply_generated_field(self.combined_field)
            else:
                # Apply the field directly to the combined field without storing it
                item.field_generator.apply(self.combined_field)

    def has(self, name):
        return str(name) in self.items

    def get(self, name):
        assert self.has(name)
        return self.items[str(name)].field_generator

    def get_combined_field(self):
        assert self.has_combined_field
        return self.combined_field

    def visualize(self, name, **plot_kwargs):
        assert self.has(name)
        item = self.items[name]
        assert item.has_generated_field
        item.field_generator.visualize_generated_field(**plot_kwargs)

# -*- coding: utf-8 -*-
# This file is part of the APSimulator API.
# Author: Lars Frogner
import numpy as np
import collections
import fields


class FieldProcessor:
    '''
    This class defines a common interface for any process that can be applied to a SpectralField.
    This interface is assumed by the FieldProcessingPipeline class. Inheriting from this class and
    implementing the method for applying the process to a field allows instances of the class to
    be added to the FieldProcessingPipeline.
    '''
    def initialize_processing(self, unprocessed_field):
        '''
        Stores properties of the unprocessed field.
        '''
        self.grid = unprocessed_field.grid
        self.wavelengths = unprocessed_field.wavelengths
        self.dtype = unprocessed_field.dtype

        self.n_wavelengths = len(self.wavelengths)
        assert(self.wavelengths.ndim == 1)

        # Run possible additional precomputations
        self.precompute_processing_quantities()

    def precompute_processing_quantities(self):
        '''
        This method can be overridden to precompute quantities that require knowledge about
        the properties of the unprocessed field.
        '''
        return None

    def process(self, field):
        '''
        Should apply the physical process to the given SpectralField object, updating its values.
        '''
        raise NotImplementedError

    def can_create_process_field(self):
        '''
        Whether the process is independent of the values of the input field, so that a field
        representing the output of the process can be created by applying the process to an
        identiy field. This will be true for linear processes.
        '''
        return False


class LinearFieldProcessor(FieldProcessor):
    '''
    Often the application of the process to the field will be linear, typically either additive or
    multiplicative. An example of additive processing is adding fluxes to a source field, while an
    example of multiplicative processing is modulating an optical aberration field with an additional
    aberration.

    A linear process can be represented without loss of generality by applying the process to an
    identity field, which can be stored. This "process field" can then be applied to an arbitrary
    field, with the result being the same as if the process was applied directly to the field.
    '''
    def create_identity_field_with_value(self, identity_value, use_memmap):
        '''
        Creates a SpectralField object with values corresponding to the identity value of
        the linear process. This can be used to generate a process field.
        '''
        return fields.SpectralField(self.grid, self.wavelengths,
                                    dtype=self.dtype,
                                    initial_value=identity_value,
                                    use_memmap=use_memmap)

    def create_process_field(self, use_memmap=True):
        '''
        Applies the process to an identiy field to produce a process field. Memory mapping is
        used by default for storing the field, since potentially many process fields will be
        created.
        '''
        process_field = self.create_identity_field(use_memmap)
        self.process(process_field)
        return process_field

    def apply_process_field(self, field, process_field):
        '''
        Should apply the given process field to the given field.
        '''
        return NotImplementedError

    def can_create_process_field(self):
        return True


class MultiplicativeFieldProcessor(LinearFieldProcessor):

    def create_identity_field(self, use_memmap):
        return self.create_identity_field_with_value(1, use_memmap)

    def apply_process_field(self, field, process_field):
        field.multiply_within_window(process_field.values)


class AdditiveFieldProcessor(LinearFieldProcessor):

    def create_identity_field(self, use_memmap):
        return self.create_identity_field_with_value(0, use_memmap)

    def apply_process_field(self, field, process_field):
        field.add_within_window(process_field.values)


class FieldProcessingPipelineStage:
    '''
    Wrapper around FieldProcessor objects exposing a convenient interface to the
    FieldProcessingPipeline. The main role is to encapsulate the management of any
    process field.
    '''
    def __init__(self, field_processor, unprocessed_field, store_process_field_if_possible):

        assert isinstance(field_processor, FieldProcessor)

        self.field_processor = field_processor
        self.store_process_field = bool(store_process_field_if_possible) and self.field_processor.can_create_process_field()

        self.field_processor.initialize_processing(unprocessed_field)

        self.process_field = None
        self.has_stored_process_field = False

    def process(self, field):
        if self.store_process_field:
            if not self.has_stored_process_field:
                self.process_field = self.field_processor.create_process_field()
                self.has_stored_process_field = True
            self.field_processor.apply_process_field(field, self.process_field)
        else:
            self.field_processor.process(field)

    def visualize_process_field(self, **plot_kwargs):
        assert self.has_stored_process_field
        fields.visualize_field(self.process_field, **plot_kwargs)


class FieldProcessingPipeline:
    '''
    Holds a set of FieldProcessor objects and allows for applying them to an initial
    field one by one in order to produce a processed field.
    '''
    def __init__(self, original_field):
        assert isinstance(original_field, fields.SpectralField)
        self.original_field = original_field
        self.processed_field = self.original_field
        self.stages = collections.OrderedDict()

    def add_field_processor(self, name, field_processor, store_process_field_if_possible=False):
        '''
        Adds a new FieldProcessor. If store_process_field_if_possible=True, a field
        representing the process in isolation will be stored as an attribute of the
        FieldProcessingPipelineStage (provided that the FieldProcessor supports this).
        '''
        assert not self.has_processor(name)
        self.stages[name] = FieldProcessingPipelineStage(field_processor, self.original_field, store_process_field_if_possible)

    def compute_processed_field(self):
        '''
        Computes the processed output field from all the field processors.
        '''
        self.processed_field = self.original_field.copy()

        for stage in self.stages.values():
            stage.process(self.processed_field)

    def has_processor(self, name):
        return name in self.stages

    def get_processor(self, name):
        assert self.has_processor(name)
        return self.stages[name].field_processor

    def get_original_field(self):
        return self.original_field

    def get_processed_field(self):
        return self.processed_field

    def visualize_process_field(self, name, **plot_kwargs):
        assert self.has_processor(name)
        self.stages[name].visualize_process_field(**plot_kwargs)

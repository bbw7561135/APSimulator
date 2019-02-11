# -*- coding: utf-8 -*-
# This file is part of the APSimulator API.
# Author: Lars Frogner
import numpy as np
import collections
import noise_utils
import math_utils
import plot_utils
import image_utils


class Galaxy:

    def __init__(self, resolution=256, spectrum_size=1, scale=1, morphology=None, orientation=None, disk_component_list=[], bulge_component_list=[]):
        self.set_resolution(resolution)
        self.set_scale(scale)
        self.set_morphology(morphology)
        self.set_orientation(orientation)
        self.set_disk_components(disk_component_list)
        self.set_bulge_components(bulge_component_list)
        self.set_spectrum_size(spectrum_size)

    def set_resolution(self, resolution):
        self.resolution = int(resolution)
        self.size_x = self.resolution
        self.size_y = self.resolution
        self.size_z = self.resolution
        self.shape = (self.size_x, self.size_y, self.size_z)

    def set_spectrum_size(self, spectrum_size):
        self.spectrum_size = int(spectrum_size)
        assert self.spectrum_size > 0
        self.sync_disk_component_spectrum_sizes()
        self.sync_bulge_component_spectrum_sizes()

    def set_scale(self, scale):
        self.scale = float(scale)

    def set_morphology(self, morphology):
        self.morphology = GalaxyMorphology() if morphology is None else morphology
        assert isinstance(self.morphology, GalaxyMorphology)

    def set_orientation(self, orientation):
        self.orientation = GalaxyOrientation() if orientation is None else orientation
        assert isinstance(self.orientation, GalaxyOrientation)

    def set_disk_components(self, disk_component_list):
        disk_component_list = list(disk_component_list)
        assert self.areinstances(disk_component_list, GalaxyDiskComponent)
        labels = [component.get_label() for component in disk_component_list]
        assert self.no_equal_elements(labels)
        self.disk_components = {label: component for label, component in zip(labels, disk_component_list)}
        self.number_of_disk_components = len(disk_component_list)
        self.sync_disk_component_spectrum_sizes()

    def set_bulge_components(self, bulge_component_list):
        bulge_component_list = list(bulge_component_list)
        assert self.areinstances(bulge_component_list, GalaxyBulgeComponent)
        labels = [component.get_label() for component in bulge_component_list]
        assert self.no_equal_elements(labels)
        self.bulge_components = {label: component for label, component in zip(labels, bulge_component_list)}
        self.number_of_bulge_components = len(bulge_component_list)
        self.sync_bulge_component_spectrum_sizes()

    def set_disk_component(self, component_label, new_disk_component):
        assert self.has_disk_component(component_label)
        assert isinstance(disk_component, GalaxyDiskComponent)
        new_disk_component.set_label(component_label)
        self.disk_components[component_label] = new_disk_component
        self.sync_component_spectrum_size(new_disk_component)

    def set_bulge_component(self, component_label, new_bulge_component):
        assert self.has_bulge_component(component_label)
        assert isinstance(bulge_component, GalaxyBulgeComponent)
        new_bulge_component.set_label(component_label)
        self.bulge_components[component_label] = new_bulge_component
        self.sync_component_spectrum_size(new_bulge_component)

    def set_disk_component_spectral_weights(self, component_label, spectral_weights):
        assert self.has_disk_component(component_label)
        assert len(spectral_weights) == self.spectrum_size
        self.disk_components[component_label].set_spectral_weights(spectral_weights)

    def set_bulge_component_spectral_weights(self, component_label, spectral_weights):
        assert self.has_bulge_component(component_label)
        assert len(spectral_weights) == self.spectrum_size
        self.bulge_components[component_label].set_spectral_weights(spectral_weights)

    def add_disk_component(self, disk_component):
        assert isinstance(disk_component, GalaxyDiskComponent)
        label = disk_component.get_label()
        assert not self.has_disk_component(label)
        self.disk_components[label] = disk_component
        self.number_of_disk_components += 1
        self.sync_component_spectrum_size(disk_component)

    def add_bulge_component(self, bulge_component):
        assert isinstance(bulge_component, GalaxyBulgeComponent)
        label = bulge_component.get_label()
        assert not self.has_bulge_component(label)
        self.bulge_components[label] = bulge_component
        self.number_of_bulge_components += 1
        self.sync_component_spectrum_size(bulge_component)

    def remove_disk_component(self, component_label):
        removed_component = self.disk_components.pop(component_label, None)
        if removed_component is not None:
            self.number_of_disk_components -= 1

    def remove_bulge_component(self, component_label):
        removed_component = self.bulge_components.pop(component_label, None)
        if removed_component is not None:
            self.number_of_bulge_components -= 1

    def set_disk_component_label(self, old_component_label, new_component_label):
        assert self.has_disk_component(old_component_label)
        assert not self.has_disk_component(new_component_label)
        disk_component = self.disk_components.pop(old_component_label)
        disk_component.set_label(new_component_label)
        self.disk_components[new_component_label] = disk_component

    def set_bulge_component_label(self, old_component_label, new_component_label):
        assert self.has_bulge_component(old_component_label)
        assert not self.has_bulge_component(new_component_label)
        bulge_component = self.bulge_components.pop(old_component_label)
        bulge_component.set_label(new_component_label)
        self.bulge_components[new_component_label] = bulge_component

    def sync_component_spectrum_size(self, component):
        component.set_spectrum_size(self.spectrum_size)

    def sync_disk_component_spectrum_sizes(self):
        for disk_component in self.disk_components.values():
            self.sync_component_spectrum_size(disk_component)

    def sync_bulge_component_spectrum_sizes(self):
        for bulge_component in self.bulge_components.values():
            self.sync_component_spectrum_size(bulge_component)

    def areinstances(self, instances, class_name):
        return len(list(filter(lambda instance: not isinstance(instance, class_name), instances))) == 0

    def no_equal_elements(self, elements):
        return len(elements) == len(set(elements))

    def has_disk_component(self, component_label):
        return component_label in self.disk_components

    def has_bulge_component(self, component_label):
        return component_label in self.bulge_components

    def compute_coordinates(self):
        coordinate_range = (-0.5*self.scale, 0.5*self.scale)
        self.x_coordinates = np.linspace(*coordinate_range, self.size_x, dtype='float32')
        self.y_coordinates = np.linspace(*coordinate_range, self.size_y, dtype='float32')
        self.z_coordinates = np.linspace(*coordinate_range, self.size_z, dtype='float32')
        self.dx = self.x_coordinates[1] - self.x_coordinates[0]
        self.dy = self.y_coordinates[1] - self.y_coordinates[0]
        self.dz = self.z_coordinates[1] - self.z_coordinates[0]

    def compute_intensity(self):
        return self.compute_intensity_from_emission_and_attenuation(*self.compute_emission_and_attenuation())

    def compute_intensity_from_emission_and_attenuation(self, emission, attenuation):
        transparency = np.exp(-np.cumsum(attenuation[:, :, :, ::-1], axis=3)*self.dz)
        intensity = np.sum(emission[:, :, :, ::-1]*transparency, axis=3)*self.dz
        return intensity

    def compute_emission_and_attenuation(self):

        self.compute_coordinates()

        full_shape = (self.spectrum_size, *self.shape)
        total_emission = np.zeros(full_shape, dtype='float32')
        total_attenuation = np.zeros(full_shape, dtype='float32')

        self.add_disk_emission_and_attenuation(total_emission, total_attenuation)
        self.add_bulge_emission_and_attenuation(total_emission, total_attenuation)

        return total_emission, total_attenuation

    def add_disk_emission_and_attenuation(self, total_emission, total_attenuation, extent_bound_scale=6, thickness_bound_scale=5):

        if self.get_number_of_active_disk_components() == 0:
            return

        max_extent = max([component.get_disk_extent() for component in self.disk_components.values()])
        max_thickness = max([component.get_disk_thickness() for component in self.disk_components.values()])

        lower_bounds, upper_bounds = self.orientation.compute_bounds(max_extent*extent_bound_scale, max_thickness*thickness_bound_scale)
        x_idx_range = np.searchsorted(self.x_coordinates, (lower_bounds[0], upper_bounds[0]))
        y_idx_range = np.searchsorted(self.y_coordinates, (lower_bounds[1], upper_bounds[1]))
        z_idx_range = np.searchsorted(self.z_coordinates, (lower_bounds[2], upper_bounds[2]))

        x_coordinates_inside = self.x_coordinates[x_idx_range[0]:x_idx_range[1]]
        y_coordinates_inside = self.y_coordinates[y_idx_range[0]:y_idx_range[1]]
        z_coordinates_inside = self.z_coordinates[z_idx_range[0]:z_idx_range[1]]

        nonzero_emission = total_emission[:,
                                          x_idx_range[0]:x_idx_range[1],
                                          y_idx_range[0]:y_idx_range[1],
                                          z_idx_range[0]:z_idx_range[1]]

        nonzero_attenuation = total_attenuation[:,
                                                x_idx_range[0]:x_idx_range[1],
                                                y_idx_range[0]:y_idx_range[1],
                                                z_idx_range[0]:z_idx_range[1]]

        azimuth_angles, radii, heights = self.orientation.compute_cylindrical_galaxy_frame_coordinates(x_coordinates_inside,
                                                                                                       y_coordinates_inside,
                                                                                                       z_coordinates_inside)

        unoriented_arm_center_angles = self.morphology.compute_unoriented_arm_center_angles(radii)

        for disk_component in self.disk_components.values():

            if disk_component.is_active():

                strength = disk_component.compute_strength(self.morphology, self.orientation,
                                                           x_coordinates_inside, y_coordinates_inside, z_coordinates_inside,
                                                           azimuth_angles, radii, heights,
                                                           unoriented_arm_center_angles)

                spectral_weights = disk_component.get_spectral_weights()

                if disk_component.is_emissive():
                    for i in range(self.spectrum_size):
                        nonzero_emission[i, :, :, :] += strength*spectral_weights[i]
                else:
                    for i in range(self.spectrum_size):
                        nonzero_attenuation[i, :, :, :] += strength*spectral_weights[i]

    def add_bulge_emission_and_attenuation(self, total_emission, total_attenuation):

        if self.get_number_of_active_bulge_components() == 0:
            return

        radii = self.orientation.compute_spherical_radii(self.x_coordinates, self.y_coordinates, self.z_coordinates)

        for bulge_component in self.bulge_components.values():

            if bulge_component.is_active():

                strength = bulge_component.compute_strength(radii)
                spectral_weights = bulge_component.get_spectral_weights()

                if bulge_component.is_emissive():
                    for i in range(self.spectrum_size):
                        total_emission[i, :, :, :] += strength*spectral_weights[i]
                else:
                    for i in range(self.spectrum_size):
                        total_attenuation[i, :, :, :] += strength*spectral_weights[i]

    def get_resolution(self):
        return self.resolution

    def get_size_x(self):
        return self.size_x

    def get_size_y(self):
        return self.size_y

    def get_size_z(self):
        return self.size_z

    def get_scale(self):
        return self.scale

    def get_morphology(self):
        return self.morphology

    def get_orientation(self):
        return self.orientation

    def get_number_of_disk_components(self):
        return self.number_of_disk_components

    def get_number_of_bulge_components(self):
        return self.number_of_bulge_components

    def get_number_of_active_disk_components(self):
        return len(list(filter(lambda component: component.is_active(), self.disk_components.values())))

    def get_number_of_active_bulge_components(self):
        return len(list(filter(lambda component: component.is_active(), self.bulge_components.values())))

    def get_disk_component_labels(self):
        return list(self.disk_components.keys())

    def get_bulge_component_labels(self):
        return list(self.bulge_components.keys())

    def get_disk_component(self, component_label):
        assert self.has_disk_component(component_label)
        return self.disk_components[component_label]

    def get_bulge_component(self, component_label):
        assert self.has_bulge_component(component_label)
        return self.bulge_components[component_label]

    GUI_param_ranges = {'resolution': [16, 32, 64, 96, 128, 192, 256, 384, 512, 768],
                        'scale': (0.1, 10, 0.1)}


class GalaxyMorphology:

    def __init__(self, **kwargs):
        self.initialize_setter_getter_dicts()
        self.params = {}
        self.reset_params_to_default()
        self.set_params(kwargs)

    def initialize_setter_getter_dicts(self):

        self.setters = {'winding_number': self.set_winding_number,
                        'bulge_to_arm_ratio': self.set_bulge_to_arm_ratio,
                        'arm_scale': self.set_arm_scale,
                        'arm_orientations': self.set_arm_orientations,
                        'arm_orientation': self.set_arm_orientation}

        self.getters = {'winding_number': self.get_winding_number,
                        'bulge_to_arm_ratio': self.get_bulge_to_arm_ratio,
                        'arm_scale': self.get_arm_scale,
                        'arm_orientations': self.get_arm_orientations,
                        'arm_orientation_angle': self.get_arm_orientation_angle,
                        'number_of_arms': self.get_number_of_arms}

    def reset_params_to_default(self):
        self.set_params(self.__class__.get_default_params())

    def set_params(self, params):
        for parameter_name in params:
            self.set(parameter_name, params[parameter_name])

    def set(self, parameter_name, *args, **kwargs):
        assert parameter_name in self.setters
        self.setters[parameter_name](*args, **kwargs)

    def get(self, quantity_name, *args, **kwargs):
        assert quantity_name in self.getters
        return self.getters[quantity_name](*args, **kwargs)

    def set_winding_number(self, winding_number):
        self.params['winding_number'] = float(winding_number)

    def set_bulge_to_arm_ratio(self, bulge_to_arm_ratio):
        self.params['bulge_to_arm_ratio'] = float(bulge_to_arm_ratio)

    def set_arm_scale(self, arm_scale):
        self.params['arm_scale'] = float(arm_scale)

    def set_arm_orientations(self, arm_orientations):
        arm_orientations_array = np.asfarray(arm_orientations, dtype='float32')
        assert arm_orientations_array.ndim == 1
        self.params['arm_orientations'] = arm_orientations_array

    def set_arm_orientation(self, arm_index, orientation_angle):
        assert arm_index < self.get_number_of_arms()
        self.get_arm_orientations()[arm_index] = orientation_angle

    def add_arm(self, orientation_angle):
        self.set_arm_orientations(list(self.get_arm_orientations()) + [orientation_angle])

    def remove_arm(self, arm_index):
        assert arm_index < self.get_number_of_arms()
        arm_orientations = self.get_arm_orientations()
        self.set_arm_orientations(list(arm_orientations[:arm_index]) + list(arm_orientations[arm_index+1:]))

    def compute_unoriented_arm_center_angles(self, radii):
        return self.get_arm_scale()*2*self.get_winding_number()*np.arctan(np.exp(-1/(2*(radii + 5e-2)))/self.get_bulge_to_arm_ratio())

    def compute_center_angles_for_all_arms(self, unoriented_arm_center_angles):
        return np.add.outer(self.get_arm_orientations(), unoriented_arm_center_angles) # Axis 0 goes over the separate arms

    def compute_arm_modulation(self, azimuth_angles, unoriented_arm_center_angles, arm_narrowness):
        arm_center_angles = self.compute_center_angles_for_all_arms(unoriented_arm_center_angles)
        angular_distances = math_utils.angular_distances(arm_center_angles, azimuth_angles[np.newaxis, :])
        angular_distances_to_closest_arm = np.min(angular_distances, axis=0)
        return (1 - angular_distances_to_closest_arm/np.pi)**arm_narrowness

    def compute_disk_modulation(self, radii, heights, disk_extent, disk_thickness):
        return np.exp(-radii/disk_extent)*(1/np.cosh(heights/disk_thickness))**2

    def get_params(self):
        return dict(self.params)

    def get_winding_number(self):
        return self.params['winding_number']

    def get_bulge_to_arm_ratio(self):
        return self.params['bulge_to_arm_ratio']

    def get_arm_scale(self):
        return self.params['arm_scale']

    def get_arm_orientations(self):
        return self.params['arm_orientations']

    def get_arm_orientation_angle(self, arm_index):
        assert arm_index < self.get_number_of_arms()
        return self.get_arm_orientations()[arm_index]

    def get_number_of_arms(self):
        return self.get_arm_orientations().size

    @staticmethod
    def get_default_params():
        return {'winding_number':     5,
                'bulge_to_arm_ratio': 0.4,
                'arm_scale':          1,
                'arm_orientations':   np.array([0, np.pi])}

    GUI_param_ranges = {'winding_number': (1, 10, 0.1),
                        'bulge_to_arm_ratio': (0.05, 1, 0.01),
                        'arm_scale': (0.1, 5, 0.05)}


class GalaxyOrientation:

    def __init__(self, normal_axis=(0, 0, 1),
                       bar_axis=(0, 1, 0)):

        self.set_normal_axis(normal_axis, reinitialize=False)
        self.set_bar_axis(bar_axis, reinitialize=False)
        self.initialize_basis_vectors()

    def set_normal_axis(self, normal_axis, reinitialize=True):
        self.normal_axis = np.asfarray(normal_axis, dtype='float32')
        assert self.normal_axis.ndim == 1 and self.normal_axis.size == 3
        normal_axis_length = np.linalg.norm(self.normal_axis)
        assert normal_axis_length > 0
        self.normal_axis /= normal_axis_length
        if reinitialize:
            self.initialize_basis_vectors()

    def set_bar_axis(self, bar_axis, reinitialize=True):
        self.bar_axis = np.asfarray(bar_axis, dtype='float32')
        assert self.bar_axis.ndim == 1 and self.bar_axis.size == 3
        bar_axis_length = np.linalg.norm(self.bar_axis)
        assert bar_axis_length > 0
        self.bar_axis /= bar_axis_length
        if reinitialize:
            self.initialize_basis_vectors()

    def set_bar_perp_axis(self, bar_perp_axis):
        self.bar_perp_axis = np.asfarray(bar_perp_axis, dtype='float32')
        assert self.bar_perp_axis.ndim == 1 and self.bar_perp_axis.size == 3
        bar_perp_axis_length = np.linalg.norm(self.bar_perp_axis)
        assert bar_perp_axis_length > 0
        self.bar_perp_axis /= bar_perp_axis_length

    def set_polar_angle(self, polar_angle):

        old_polar_angle = self.get_polar_angle()
        polar_angle_diff = polar_angle - old_polar_angle

        cos_polar_angle_diff = np.cos(polar_angle_diff)
        sin_polar_angle_diff = np.sin(polar_angle_diff)

        new_normal_axis_x = sin_polar_angle_diff*self.bar_axis[0] + cos_polar_angle_diff*self.normal_axis[0]
        new_normal_axis_y = sin_polar_angle_diff*self.bar_axis[1] + cos_polar_angle_diff*self.normal_axis[1]
        new_normal_axis_z = sin_polar_angle_diff*self.bar_axis[2] + cos_polar_angle_diff*self.normal_axis[2]

        new_bar_axis_x = cos_polar_angle_diff*self.bar_axis[0] - sin_polar_angle_diff*self.normal_axis[0]
        new_bar_axis_y = cos_polar_angle_diff*self.bar_axis[1] - sin_polar_angle_diff*self.normal_axis[1]
        new_bar_axis_z = cos_polar_angle_diff*self.bar_axis[2] - sin_polar_angle_diff*self.normal_axis[2]

        self.set_normal_axis((new_normal_axis_x, new_normal_axis_y, new_normal_axis_z), reinitialize=False)
        self.set_bar_axis((new_bar_axis_x, new_bar_axis_y, new_bar_axis_z), reinitialize=False)

    def set_azimuth_angle(self, azimuth_angle):

        cos_polar_angle = self.normal_axis[2]
        sin_polar_angle = np.sqrt(max(0, 1 - cos_polar_angle**2))
        cos_azimuth_angle = np.cos(azimuth_angle)
        sin_azimuth_angle = np.sin(azimuth_angle)

        new_normal_axis_x = sin_polar_angle*cos_azimuth_angle
        new_normal_axis_y = sin_polar_angle*sin_azimuth_angle
        new_normal_axis_z = cos_polar_angle

        new_bar_axis_x = cos_polar_angle*cos_azimuth_angle
        new_bar_axis_y = cos_polar_angle*sin_azimuth_angle
        new_bar_axis_z = -sin_polar_angle

        self.set_normal_axis((new_normal_axis_x, new_normal_axis_y, new_normal_axis_z), reinitialize=False)
        self.set_bar_axis((new_bar_axis_x, new_bar_axis_y, new_bar_axis_z), reinitialize=False)
        self.set_bar_perp_axis(self.compute_bar_perp_axis())

    def project_bar_axis_onto_disk_plane(self):
        self.bar_axis -= np.dot(self.bar_axis, self.normal_axis)*self.normal_axis;
        bar_axis_length = np.linalg.norm(self.bar_axis)
        assert bar_axis_length > 0
        self.bar_axis /= bar_axis_length

    def compute_bar_perp_axis(self):
        return np.cross(self.normal_axis, self.bar_axis)

    def initialize_basis_vectors(self):
        self.project_bar_axis_onto_disk_plane()
        self.set_bar_perp_axis(self.compute_bar_perp_axis())

    def compute_bounds(self, max_extent, max_thickness):

        galaxy_halfwidth = max_extent*self.bar_axis
        galaxy_halfdepth = max_extent*self.bar_perp_axis
        galaxy_halfheight = max_thickness*self.normal_axis

        lower_bounds = np.array([np.inf]*3)
        upper_bounds = np.array([-np.inf]*3)

        for x_sign in (-1, 1):
            for y_sign in (-1, 1):
                for z_sign in (-1, 1):
                    corner = x_sign*galaxy_halfwidth + y_sign*galaxy_halfdepth + z_sign*galaxy_halfheight
                    lower_bounds = np.minimum(lower_bounds, corner)
                    upper_bounds = np.maximum(upper_bounds, corner)

        return lower_bounds, upper_bounds

    def compute_galaxy_frame_coordinates(self, x_coordinates, y_coordinates, z_coordinates):

        x, y, z = np.meshgrid(x_coordinates, y_coordinates, z_coordinates, indexing='ij')

        galaxy_x = x*self.bar_axis[0] + y*self.bar_axis[1] + z*self.bar_axis[2]
        galaxy_y = x*self.bar_perp_axis[0] + y*self.bar_perp_axis[1] + z*self.bar_perp_axis[2]
        galaxy_z = x*self.normal_axis[0] + y*self.normal_axis[1] + z*self.normal_axis[2]

        return galaxy_x, galaxy_y, galaxy_z

    def compute_cylindrical_galaxy_frame_coordinates(self, x_coordinates, y_coordinates, z_coordinates):

        galaxy_x, galaxy_y, galaxy_z = self.compute_galaxy_frame_coordinates(x_coordinates, y_coordinates, z_coordinates)

        azimuth_angles = np.arctan2(galaxy_y, galaxy_x)
        radii = np.sqrt(galaxy_x**2 + galaxy_y**2)
        heights = galaxy_z

        return azimuth_angles, radii, heights

    def compute_spherical_radii(self, x_coordinates, y_coordinates, z_coordinates):
        x, y, z = np.meshgrid(x_coordinates, y_coordinates, z_coordinates, indexing='ij')
        return np.sqrt(x**2 + y**2 + z**2)

    def compute_disk_rotated_coordinates(self, azimuth_angles, radii, heights, rotation_angles):

        new_azimuth_angles = azimuth_angles + rotation_angles
        galaxy_x = radii*np.cos(new_azimuth_angles)
        galaxy_y = radii*np.sin(new_azimuth_angles)
        galaxy_z = heights

        x = galaxy_x*self.bar_axis[0] + galaxy_y*self.bar_perp_axis[0] + galaxy_z*self.normal_axis[0]
        y = galaxy_x*self.bar_axis[1] + galaxy_y*self.bar_perp_axis[1] + galaxy_z*self.normal_axis[1]
        z = galaxy_x*self.bar_axis[2] + galaxy_y*self.bar_perp_axis[2] + galaxy_z*self.normal_axis[2]

        return x, y, z

    def get_normal_axis(self):
        return self.normal_axis

    def get_bar_axis(self):
        return self.bar_axis

    def get_bar_perp_axis(self):
        return self.bar_perp_axis

    def get_polar_angle(self):
        return np.arccos(self.normal_axis[2])

    def get_azimuth_angle(self):
        return np.arctan2(self.normal_axis[1], self.normal_axis[0]) if self.normal_axis[2] < 1 else np.arctan2(self.bar_axis[1], self.bar_axis[0])


class GalaxyBulgeComponent:

    def __init__(self, label, **kwargs):

        self.initialize_setter_getter_dicts()
        self.initialize_spectral_weights()

        self.set_label(label)

        self.params = {}
        self.reset_params_to_default()
        self.set_params(kwargs)

    def initialize_setter_getter_dicts(self):

        self.setters = {'label': self.set_label,
                        'active': self.set_active,
                        'emissive': self.set_emission,
                        'strength_scale': self.set_strength_scale,
                        'bulge_size': self.set_bulge_size}

        self.getters = {'label': self.get_label,
                        'active': self.is_active,
                        'emissive': self.is_emissive,
                        'strength_scale': self.get_strength_scale,
                        'bulge_size': self.get_bulge_size,
                        'spectrum_size': self.get_spectrum_size,
                        'spectral_weights': self.get_spectral_weights}

    def initialize_spectral_weights(self):
        self.spectrum_size = 1
        self.spectral_weights = np.array([1], dtype='float32')

    def reset_params_to_default(self):
        self.set_params(self.__class__.get_default_params())

    def set_params(self, params):
        for parameter_name in params:
            self.set(parameter_name, params[parameter_name])

    def set(self, parameter_name, *args, **kwargs):
        assert parameter_name in self.setters
        self.setters[parameter_name](*args, **kwargs)

    def get(self, quantity_name, *args, **kwargs):
        assert quantity_name in self.getters
        return self.getters[quantity_name](*args, **kwargs)

    def set_label(self, label):
        self.label = str(label)

    def set_active(self, active):
        self.params['active'] = bool(active)

    def set_emission(self, emissive):
        self.params['emissive'] = bool(emissive)

    def set_strength_scale(self, strength_scale):
        self.params['strength_scale'] = float(strength_scale)

    def set_bulge_size(self, bulge_size):
        self.params['bulge_size'] = float(bulge_size)

    def set_spectrum_size(self, spectrum_size):
        old_spectrum_size = self.spectrum_size
        self.spectrum_size = int(spectrum_size)
        assert self.spectrum_size > 0

        if self.spectrum_size > old_spectrum_size:
            new_spectral_weights = np.ones(self.spectrum_size, dtype='float32')
            new_spectral_weights[:old_spectrum_size] = self.spectral_weights
            self.spectral_weights = new_spectral_weights
        elif self.spectrum_size < old_spectrum_size:
            self.spectral_weights = self.spectral_weights[:self.spectrum_size]

    def set_spectral_weights(self, spectral_weights):
        spectral_weights_array = np.asfarray(spectral_weights)
        assert spectral_weights_array.ndim == 1
        max_idx = min(self.spectrum_size, spectral_weights_array.size)
        self.spectral_weights[:max_idx] = spectral_weights_array[:max_idx]

    def compute_strength(self, radii):
        r = (radii + 1e-2)/self.get_bulge_size()
        return self.get_strength_scale()/(r**0.855*np.exp(r**0.25))

    def get_params(self):
        return dict(self.params)

    def get_label(self):
        return self.label

    def is_active(self):
        return self.params['active']

    def is_emissive(self):
        return self.params['emissive']

    def get_strength_scale(self):
        return self.params['strength_scale']

    def get_bulge_size(self):
        return self.params['bulge_size']

    def get_spectrum_size(self):
        return self.spectrum_size

    def get_spectral_weights(self):
        return self.spectral_weights

    @staticmethod
    def get_default_params():
        return {'active':           True,
                'emissive':         True,
                'strength_scale':   2,
                'bulge_size':       0.03}

    GUI_param_ranges = {'strength_scale': (0.01, 100, 1),
                        'bulge_size': (0.001, 0.1, 0.001)}


class GalaxyDiskComponent:

    def __init__(self, label, **kwargs):

        self.initialize_setter_getter_dicts()
        self.initialize_spectral_weights()

        self.set_label(label)

        self.fractal_noise_pattern = noise_utils.FractalNoisePattern()

        self.params = {}
        self.reset_params_to_default()
        self.set_params(kwargs)

    def initialize_setter_getter_dicts(self):
        self.setters = {'label': self.set_label,
                        'active': self.set_active,
                        'emissive': self.set_emission,
                        'strength_scale': self.set_strength_scale,
                        'disk_extent': self.set_disk_extent,
                        'disk_thickness': self.set_disk_thickness,
                        'arm_narrowness': self.set_arm_narrowness,
                        'twirl_factor': self.set_twirl_factor,
                        'number_of_octaves': self.set_number_of_octaves,
                        'initial_frequency': self.set_initial_frequency,
                        'lacunarity': self.set_lacunarity,
                        'persistence': self.set_persistence,
                        'seed': self.set_seed,
                        'noise_threshold': self.set_noise_threshold,
                        'noise_cutoff': self.set_noise_cutoff,
                        'noise_exponent': self.set_noise_exponent,
                        'noise_offset': self.set_noise_offset}

        self.getters = {'label': self.get_label,
                        'active': self.is_active,
                        'emissive': self.is_emissive,
                        'strength_scale': self.get_strength_scale,
                        'disk_extent': self.get_disk_extent,
                        'disk_thickness': self.get_disk_thickness,
                        'arm_narrowness': self.get_arm_narrowness,
                        'twirl_factor': self.get_twirl_factor,
                        'number_of_octaves': self.get_number_of_octaves,
                        'initial_frequency': self.get_initial_frequency,
                        'lacunarity': self.get_lacunarity,
                        'persistence': self.get_persistence,
                        'seed': self.get_seed,
                        'noise_threshold': self.get_noise_threshold,
                        'noise_cutoff': self.get_noise_cutoff,
                        'noise_exponent': self.get_noise_exponent,
                        'noise_offset': self.get_noise_offset,
                        'spectrum_size': self.get_spectrum_size,
                        'spectral_weights': self.get_spectral_weights}

    def initialize_spectral_weights(self):
        self.spectrum_size = 1
        self.spectral_weights = np.array([1], dtype='float32')

    def reset_params_to_default(self):
        self.set_params(self.__class__.get_default_params())

    def set_params(self, params):
        for parameter_name in params:
            self.set(parameter_name, params[parameter_name])

    def set(self, parameter_name, *args, **kwargs):
        assert parameter_name in self.setters
        self.setters[parameter_name](*args, **kwargs)

    def get(self, quantity_name, *args, **kwargs):
        assert quantity_name in self.getters
        return self.getters[quantity_name](*args, **kwargs)

    def set_label(self, label):
        self.label = str(label)

    def set_active(self, active):
        self.params['active'] = bool(active)

    def set_emission(self, emissive):
        self.params['emissive'] = bool(emissive)

    def set_strength_scale(self, strength_scale):
        self.params['strength_scale'] = float(strength_scale)

    def set_disk_extent(self, disk_extent):
        self.params['disk_extent'] = float(disk_extent)

    def set_disk_thickness(self, disk_thickness):
        self.params['disk_thickness'] = float(disk_thickness)

    def set_arm_narrowness(self, arm_narrowness):
        self.params['arm_narrowness'] = float(arm_narrowness)

    def set_twirl_factor(self, twirl_factor):
        self.params['twirl_factor'] = float(twirl_factor)

    def set_number_of_octaves(self, number_of_octaves):
        self.fractal_noise_pattern.set_number_of_octaves(number_of_octaves)
        self.params['number_of_octaves'] = self.fractal_noise_pattern.get_number_of_octaves()

    def set_initial_frequency(self, initial_frequency):
        self.fractal_noise_pattern.set_initial_frequency(initial_frequency)
        self.params['initial_frequency'] = self.fractal_noise_pattern.get_initial_frequency()

    def set_lacunarity(self, lacunarity):
        self.fractal_noise_pattern.set_lacunarity(lacunarity)
        self.params['lacunarity'] = self.fractal_noise_pattern.get_lacunarity()

    def set_persistence(self, persistence):
        self.fractal_noise_pattern.set_persistence(persistence)
        self.params['persistence'] = self.fractal_noise_pattern.get_persistence()

    def set_seed(self, seed):
        self.fractal_noise_pattern.set_seed(seed)
        self.params['seed'] = self.fractal_noise_pattern.get_seed()

    def set_noise_threshold(self, noise_threshold):
        self.params['noise_threshold'] = float(noise_threshold)

    def set_noise_cutoff(self, noise_cutoff):
        self.params['noise_cutoff'] = float(noise_cutoff)

    def set_noise_exponent(self, noise_exponent):
        self.params['noise_exponent'] = float(noise_exponent)

    def set_noise_offset(self, noise_offset):
        self.params['noise_offset'] = float(noise_offset)

    def set_spectrum_size(self, spectrum_size):
        old_spectrum_size = self.spectrum_size
        self.spectrum_size = int(spectrum_size)
        assert self.spectrum_size > 0

        if self.spectrum_size > old_spectrum_size:
            new_spectral_weights = np.ones(self.spectrum_size, dtype='float32')
            new_spectral_weights[:old_spectrum_size] = self.spectral_weights
            self.spectral_weights = new_spectral_weights
        elif self.spectrum_size < old_spectrum_size:
            self.spectral_weights = self.spectral_weights[:self.spectrum_size]

    def set_spectral_weights(self, spectral_weights):
        spectral_weights_array = np.asfarray(spectral_weights)
        assert spectral_weights_array.ndim == 1
        max_idx = min(self.spectrum_size, spectral_weights_array.size)
        self.spectral_weights[:max_idx] = spectral_weights_array[:max_idx]

    def compute_strength(self, morphology, orientation,
                               x_coordinates, y_coordinates, z_coordinates,
                               azimuth_angles, radii, heights,
                               unoriented_arm_center_angles,
                               cutoff_limit=1e-6):

        largest_scale = max((x_coordinates[-1] - x_coordinates[0],
                             y_coordinates[-1] - y_coordinates[0],
                             z_coordinates[-1] - z_coordinates[0]))

        smallest_scale = min((x_coordinates[1] - x_coordinates[0],
                              y_coordinates[1] - y_coordinates[0],
                              z_coordinates[1] - z_coordinates[0]))

        if self.get_twirl_factor() != 0:
            twirl_rotations = unoriented_arm_center_angles*(-self.get_twirl_factor())
            x_coordinates, y_coordinates, z_coordinates = orientation.compute_disk_rotated_coordinates(azimuth_angles, radii, heights, twirl_rotations)

        modulation = morphology.compute_disk_modulation(radii, heights, self.get_disk_extent(), self.get_disk_thickness())*\
                     morphology.compute_arm_modulation(azimuth_angles, unoriented_arm_center_angles, self.get_arm_narrowness())

        mask = modulation > cutoff_limit
        noise_pattern = self.generate_noise_pattern(x_coordinates, y_coordinates, z_coordinates, smallest_scale, largest_scale, mask)

        strength_scale = self.get_strength_scale()
        if not self.is_emissive():
            strength_scale *= 30 # Keep required input strength scale values within a practical range

        return noise_pattern*modulation*strength_scale

    def generate_noise_pattern(self, x_coordinates, y_coordinates, z_coordinates, smallest_scale, largest_scale, mask):

        noise = self.fractal_noise_pattern.compute(x_coordinates, y_coordinates, z_coordinates, smallest_scale, largest_scale, mask)

        noise_threshold = self.get_noise_threshold()
        if noise_threshold != 0:
            noise += noise_threshold

        np.abs(noise, out=noise)

        if noise_threshold != 0:
            noise /= 1 + noise_threshold

        noise_cutoff = self.get_noise_cutoff()
        if noise_cutoff > 0:
            noise[noise < noise_cutoff] = 0

        noise_exponent = self.get_noise_exponent()
        if noise_exponent != 1:
            noise **= noise_exponent

        noise_offset = self.get_noise_offset()
        if noise_offset != 0:
            noise += noise_offset

        noise /= np.mean(noise)

        return noise

    def get_label(self):
        return self.label

    def get_params(self):
        return dict(self.params)

    def is_active(self):
        return self.params['active']

    def is_emissive(self):
        return self.params['emissive']

    def get_strength_scale(self):
        return self.params['strength_scale']

    def get_disk_extent(self):
        return self.params['disk_extent']

    def get_disk_thickness(self):
        return self.params['disk_thickness']

    def get_arm_narrowness(self):
        return self.params['arm_narrowness']

    def get_twirl_factor(self):
        return self.params['twirl_factor']

    def get_number_of_octaves(self):
        return self.params['number_of_octaves']

    def get_initial_frequency(self):
        return self.params['initial_frequency']

    def get_lacunarity(self):
        return self.params['lacunarity']

    def get_persistence(self):
        return self.params['persistence']

    def get_seed(self):
        return self.params['seed']

    def get_noise_threshold(self):
        return self.params['noise_threshold']

    def get_noise_cutoff(self):
        return self.params['noise_cutoff']

    def get_noise_exponent(self):
        return self.params['noise_exponent']

    def get_noise_offset(self):
        return self.params['noise_offset']

    def get_spectrum_size(self):
        return self.spectrum_size

    def get_spectral_weights(self):
        return self.spectral_weights

    @staticmethod
    def get_max_seed():
        return noise_utils.get_max_seed()

    @staticmethod
    def get_default_params():
        return {'active':            True,
                'emissive':          True,
                'strength_scale':    1,
                'disk_extent':       0.2,
                'disk_thickness':    0.015,
                'arm_narrowness':    5,
                'twirl_factor':      0.2,
                'number_of_octaves': 10,
                'initial_frequency': 8,
                'lacunarity':        2,
                'persistence':       1,
                'noise_threshold':   0,
                'noise_cutoff':      0,
                'noise_exponent':    1,
                'noise_offset':      0,
                'seed':              noise_utils.get_random_seed()}

    @staticmethod
    def get_component_types():
        return collections.OrderedDict([('Filaments', GalaxyFilaments),
                                        ('Dust', GalaxyDust),
                                        ('Stars', GalaxyStars)])

    GUI_param_ranges = {'strength_scale': (0.01, 100, 1),
                        'disk_extent': (0.01, 0.5, 0.005),
                        'disk_thickness': (0.001, 0.05, 0.0005),
                        'arm_narrowness': (0, 10, 0.1),
                        'twirl_factor': (0, 0.5, 0.005),
                        'number_of_octaves': (1, 12),
                        'initial_frequency': (0, 20, 0.2),
                        'lacunarity': (1.5, 3.5, 0.01),
                        'persistence': (0.5, 1.5, 0.01),
                        'noise_threshold': (0, 1, 0.01),
                        'noise_cutoff': (0, 1, 0.01),
                        'noise_exponent': (0.1, 10, 0.1),
                        'noise_offset': (0, 1, 0.01)}


class GalaxyFilaments(GalaxyDiskComponent):

    def __init__(self, label, **kwargs):
        super().__init__(label, **kwargs)

    @staticmethod
    def get_default_params():
        return {'active':            True,
                'emissive':          True,
                'strength_scale':    1,
                'disk_extent':       0.2,
                'disk_thickness':    0.015,
                'arm_narrowness':    4,
                'twirl_factor':      0.2,
                'number_of_octaves': 8,
                'initial_frequency': 6,
                'lacunarity':        2,
                'persistence':       0.8,
                'noise_threshold':   0,
                'noise_cutoff':      0,
                'noise_exponent':    1,
                'noise_offset':      0,
                'seed':              noise_utils.get_random_seed()}


class GalaxyDust(GalaxyDiskComponent):

    def __init__(self, label, **kwargs):
        super().__init__(label, **kwargs)

    @staticmethod
    def get_default_params():
        return {'active':            True,
                'emissive':          False,
                'strength_scale':    1,
                'disk_extent':       0.2,
                'disk_thickness':    0.01,
                'arm_narrowness':    3,
                'twirl_factor':      0.2,
                'number_of_octaves': 5,
                'initial_frequency': 8,
                'lacunarity':        2,
                'persistence':       0.95,
                'noise_threshold':   0,
                'noise_cutoff':      0,
                'noise_exponent':    1,
                'noise_offset':      0,
                'seed':              noise_utils.get_random_seed()}


class GalaxyStars(GalaxyDiskComponent):

    def __init__(self, label, **kwargs):
        super().__init__(label, **kwargs)

    @staticmethod
    def get_default_params():
        return {'active':            True,
                'emissive':          True,
                'strength_scale':    0.3,
                'disk_extent':       0.2,
                'disk_thickness':    0.01,
                'arm_narrowness':    3,
                'twirl_factor':      0.1,
                'number_of_octaves': 10,
                'initial_frequency': 8,
                'lacunarity':        2,
                'persistence':       1.2,
                'noise_threshold':   0,
                'noise_cutoff':      0,
                'noise_exponent':    6,
                'noise_offset':      0,
                'seed':              noise_utils.get_random_seed()}

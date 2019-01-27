# -*- coding: utf-8 -*-
# This file is part of the APSimulator API.
# Author: Lars Frogner
import numpy as np
import noise_utils
import math_utils
import plot_utils
import image_utils


class Galaxy:

    def __init__(self, morphology, orientation, disk_components, bulge_components):
        self.morphology = morphology
        self.orientation = orientation
        self.disk_components = disk_components
        self.bulge_components = bulge_components

    def compute_emission_and_attenuation(self, x_coordinates, y_coordinates, z_coordinates):

        emission = np.zeros((x_coordinates.size, y_coordinates.size, z_coordinates.size), dtype='float32')
        attenuation = np.zeros((x_coordinates.size, y_coordinates.size, z_coordinates.size), dtype='float32')

        self.compute_disk_emission_and_attenuation(x_coordinates, y_coordinates, z_coordinates, emission, attenuation)
        self.compute_bulge_emission_and_attenuation(x_coordinates, y_coordinates, z_coordinates, emission, attenuation)

        dz = z_coordinates[1] - z_coordinates[0]
        transparency = np.exp(-np.cumsum(attenuation[:, :, ::-1], axis=2)*dz)
        intensity = np.sum(emission[:, :, ::-1]*transparency, axis=2)*dz

        return intensity

    def compute_disk_emission_and_attenuation(self, x_coordinates, y_coordinates, z_coordinates, emission, attenuation):

        max_extent = max([component.disk_extent for component in self.disk_components])
        max_thickness = max([component.disk_thickness for component in self.disk_components])

        lower_bounds, upper_bounds = self.orientation.compute_bounds(max_extent*5, max_thickness*4)
        x_idx_range = np.searchsorted(x_coordinates, (lower_bounds[0], upper_bounds[0]))
        y_idx_range = np.searchsorted(y_coordinates, (lower_bounds[1], upper_bounds[1]))
        z_idx_range = np.searchsorted(z_coordinates, (lower_bounds[2], upper_bounds[2]))

        x_coordinates_inside = x_coordinates[x_idx_range[0]:x_idx_range[1]]
        y_coordinates_inside = y_coordinates[y_idx_range[0]:y_idx_range[1]]
        z_coordinates_inside = z_coordinates[z_idx_range[0]:z_idx_range[1]]

        nonzero_emission = emission[x_idx_range[0]:x_idx_range[1],
                                    y_idx_range[0]:y_idx_range[1],
                                    z_idx_range[0]:z_idx_range[1]]

        nonzero_attenuation = attenuation[x_idx_range[0]:x_idx_range[1],
                                          y_idx_range[0]:y_idx_range[1],
                                          z_idx_range[0]:z_idx_range[1]]

        azimuth_angles, radii, heights = self.orientation.compute_cylindrical_galaxy_frame_coordinates(x_coordinates_inside,
                                                                                                       y_coordinates_inside,
                                                                                                       z_coordinates_inside)

        unoriented_arm_center_angles = self.morphology.compute_unoriented_arm_center_angles(radii)

        for disk_component in self.disk_components:
            strength = disk_component.compute_strength(self.morphology, self.orientation,
                                                       x_coordinates_inside, y_coordinates_inside, z_coordinates_inside,
                                                       azimuth_angles, radii, heights,
                                                       unoriented_arm_center_angles)
            if disk_component.is_emissive:
                nonzero_emission[:] += strength
            else:
                nonzero_attenuation[:] += strength

    def compute_bulge_emission_and_attenuation(self, x_coordinates, y_coordinates, z_coordinates, emission, attenuation):

        radii = self.orientation.compute_spherical_radii(x_coordinates, y_coordinates, z_coordinates)

        for bulge_component in self.bulge_components:
            strength = bulge_component.compute_strength(radii)
            if bulge_component.is_emissive:
                emission[:] += strength
            else:
                attenuation[:] += strength


class GalaxyMorphology:

    def __init__(self, winding_number=5, bulge_to_arm_ratio=0.4, arm_scale=1, arm_orientations=[0, np.pi]):
        self.winding_number = float(winding_number)
        self.bulge_to_arm_ratio = float(bulge_to_arm_ratio)
        self.arm_scale = float(arm_scale)
        self.arm_orientations = np.asfarray(arm_orientations, dtype='float32')
        self.number_of_arms = self.arm_orientations.size
        assert self.arm_orientations.ndim == 1

    def compute_unoriented_arm_center_angles(self, radii):
        return self.arm_scale*2*self.winding_number*np.arctan(np.exp(-1/(2*(radii + 5e-2)))/self.bulge_to_arm_ratio)

    def compute_center_angles_for_all_arms(self, unoriented_arm_center_angles):
        return np.add.outer(self.arm_orientations, unoriented_arm_center_angles) # Axis 0 goes over the separate arms

    def compute_arm_modulation(self, azimuth_angles, unoriented_arm_center_angles, arm_narrowness):
        arm_center_angles = self.compute_center_angles_for_all_arms(unoriented_arm_center_angles)
        angular_distances = math_utils.angular_distances(arm_center_angles, azimuth_angles[np.newaxis, :])
        angular_distances_to_closest_arm = np.min(angular_distances, axis=0)
        return (1 - angular_distances_to_closest_arm/np.pi)**arm_narrowness

    def compute_disk_modulation(self, radii, heights, disk_extent, disk_thickness):
        return np.exp(-radii/disk_extent)*(1/np.cosh(heights/disk_thickness))**2


class GalaxyOrientation:

    def __init__(self, normal_axis=(0, 0, 1),
                       bar_axis=(1, 0, 0)):

        self.set_normal_axis(normal_axis)
        self.set_bar_axis(bar_axis)
        self.initialize_basis_vectors()

    def set_normal_axis(self, normal_axis):
        self.normal_axis = np.asfarray(normal_axis, dtype='float32')
        assert self.normal_axis.ndim == 1 and self.normal_axis.size == 3
        normal_axis_length = np.linalg.norm(self.normal_axis)
        assert normal_axis_length > 0
        self.normal_axis /= normal_axis_length

    def set_bar_axis(self, bar_axis):
        self.bar_axis = np.asfarray(bar_axis, dtype='float32')
        assert self.bar_axis.ndim == 1 and self.bar_axis.size == 3
        bar_axis_length = np.linalg.norm(self.bar_axis)
        assert bar_axis_length > 0
        self.bar_axis /= bar_axis_length

    def project_bar_axis_onto_disk_plane(self):
        self.bar_axis -= np.dot(self.bar_axis, self.normal_axis)*self.normal_axis;
        bar_axis_length = np.linalg.norm(self.bar_axis)
        assert bar_axis_length > 0
        self.bar_axis /= bar_axis_length

    def initialize_basis_vectors(self):
        self.project_bar_axis_onto_disk_plane()
        self.bar_perp_axis = np.cross(self.normal_axis, self.bar_axis)
        self.bar_perp_axis /= np.linalg.norm(self.bar_perp_axis)

    def compute_bounds(self, max_extent, max_thickness):# extent_cutoff=5, thickness_cutoff=4):

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


class GalaxyBulgeComponent:

    def __init__(self, is_emissive=True, strength_scale=2, bulge_size=0.03):
        self.is_emissive = bool(is_emissive)
        self.strength_scale = float(strength_scale)
        self.bulge_size = float(bulge_size)

    def compute_strength(self, radii):
        r = (radii + 1e-2)/self.bulge_size
        return self.strength_scale/(r**0.855*np.exp(r**0.25))


class GalaxyDiskComponent:

    def __init__(self, is_emissive=True,
                       strength_scale=1,
                       disk_extent=0.2,
                       disk_thickness=0.015,
                       arm_narrowness=5,
                       twirl_factor=0.2,
                       number_of_octaves=10,
                       initial_frequency=8,
                       persistence=1,
                       noise_exponent=1,
                       noise_offset=0,
                       seed=None):

        self.is_emissive = bool(is_emissive)
        self.strength_scale = float(strength_scale)
        self.disk_extent = float(disk_extent)
        self.disk_thickness = float(disk_thickness)
        self.arm_narrowness = float(arm_narrowness)
        self.twirl_factor = float(twirl_factor)

        self.fractal_noise_pattern = noise_utils.FractalNoisePattern(number_of_octaves=number_of_octaves,
                                                                     initial_frequency=initial_frequency,
                                                                     persistence=persistence,
                                                                     seed=seed)

        self.noise_exponent = float(noise_exponent)
        self.noise_offset = float(noise_offset)

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

        if self.twirl_factor != 0:
            twirl_rotations = unoriented_arm_center_angles*(-self.twirl_factor)
            x_coordinates, y_coordinates, z_coordinates = orientation.compute_disk_rotated_coordinates(azimuth_angles, radii, heights, twirl_rotations)

        modulation = morphology.compute_disk_modulation(radii, heights, self.disk_extent, self.disk_thickness)*\
                     morphology.compute_arm_modulation(azimuth_angles, unoriented_arm_center_angles, self.arm_narrowness)

        mask = modulation > cutoff_limit
        noise_pattern = self.generate_noise_pattern(x_coordinates, y_coordinates, z_coordinates, smallest_scale, largest_scale, mask)

        return noise_pattern*modulation*self.strength_scale

    def generate_noise_pattern(self, x_coordinates, y_coordinates, z_coordinates, smallest_scale, largest_scale, mask):

        noise = np.abs(self.fractal_noise_pattern.compute(x_coordinates, y_coordinates, z_coordinates, smallest_scale, largest_scale, mask))

        if self.noise_exponent != 1:
            noise **= self.noise_exponent

        if self.noise_offset != 0:
            noise += self.noise_offset

        noise /= np.mean(noise)

        return noise


class GalaxyDisk(GalaxyDiskComponent):

    def __init__(self, is_emissive=True,
                       strength_scale=1,
                       disk_extent=0.2,
                       disk_thickness=0.015,
                       arm_narrowness=4,
                       twirl_factor=0.2,
                       number_of_octaves=8,
                       initial_frequency=6,
                       persistence=0.8,
                       noise_exponent=1,
                       noise_offset=0,
                       seed=None):

        super().__init__(is_emissive=is_emissive,
                         strength_scale=strength_scale,
                         disk_extent=disk_extent,
                         disk_thickness=disk_thickness,
                         arm_narrowness=arm_narrowness,
                         twirl_factor=twirl_factor,
                         number_of_octaves=number_of_octaves,
                         initial_frequency=initial_frequency,
                         persistence=persistence,
                         noise_exponent=noise_exponent,
                         noise_offset=noise_offset,
                         seed=seed)


class GalaxyDust(GalaxyDiskComponent):

    def __init__(self, is_emissive=False,
                       strength_scale=30,
                       disk_extent=0.2,
                       disk_thickness=0.01,
                       arm_narrowness=3,
                       twirl_factor=0.2,
                       number_of_octaves=5,
                       initial_frequency=8,
                       persistence=0.95,
                       noise_exponent=1,
                       noise_offset=0,
                       seed=None):

        super().__init__(is_emissive=is_emissive,
                         strength_scale=strength_scale,
                         disk_extent=disk_extent,
                         disk_thickness=disk_thickness,
                         arm_narrowness=arm_narrowness,
                         twirl_factor=twirl_factor,
                         number_of_octaves=number_of_octaves,
                         initial_frequency=initial_frequency,
                         persistence=persistence,
                         noise_exponent=noise_exponent,
                         noise_offset=noise_offset,
                         seed=seed)


class GalaxyStars(GalaxyDiskComponent):

    def __init__(self, is_emissive=True,
                       strength_scale=0.5,
                       disk_extent=0.2,
                       disk_thickness=0.01,
                       arm_narrowness=3,
                       twirl_factor=0.1,
                       number_of_octaves=10,
                       initial_frequency=8,
                       persistence=1.2,
                       noise_exponent=6,
                       noise_offset=0,
                       seed=None):

        super().__init__(is_emissive=is_emissive,
                         strength_scale=strength_scale,
                         disk_extent=disk_extent,
                         disk_thickness=disk_thickness,
                         arm_narrowness=arm_narrowness,
                         twirl_factor=twirl_factor,
                         number_of_octaves=number_of_octaves,
                         initial_frequency=initial_frequency,
                         persistence=persistence,
                         noise_exponent=noise_exponent,
                         noise_offset=noise_offset,
                         seed=seed)


if __name__ == '__main__':

    N = 512
    x_coordinates = np.linspace(-0.5, 0.5, N, dtype='float32')
    y_coordinates = np.linspace(-0.5, 0.5, N, dtype='float32')
    z_coordinates = np.linspace(-0.5, 0.5, N, dtype='float32')

    galaxy = Galaxy(GalaxyMorphology(), GalaxyOrientation(normal_axis=(0, 0, 1)), [GalaxyDust(), GalaxyStars(), GalaxyDisk()], [GalaxyBulgeComponent()])
    intensity = galaxy.compute_emission_and_attenuation(x_coordinates, y_coordinates, z_coordinates)

    fig, ax = plot_utils.subplots()
    plot_utils.plot_image(fig, ax, intensity.T, vmax=0.1)#image_utils.perform_autostretch(np.sum(emission, axis=2).T))
    plot_utils.render()

    '''
    noise = GalaxyDisk().generate_noise_pattern(x_coordinates, y_coordinates, z_coordinates)

    fig, ax = plot_utils.subplots()
    plot_utils.plot_image(fig, ax, np.sum(noise, axis=2), vmin=None, vmax=None)
    plot_utils.render()
    '''

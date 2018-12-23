# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import incident_light_fields
import image_light_fields
import math_utils
import plot_utils


class FraunhoferDiffractionOptics:
    '''
    This class can compute the Fraunhofer diffracted field in the imaging plane
    from the light field incident on an aperture. The incident field is represented
    as a sum of plane waves from different directions, and the resulting Fraunhofer
    diffracted field, which is found by Fourier transforming the incident field on
    passing through the aperture, gives the spatial distribution of spectral flux
    in the imagingnplane. Note that Fraunhofer diffraction only gives the perfectly
    focused image.

    The aperture grid coordinates are normalized with wavelength, so the actual spatial
    position of a grid cell is found by multiplying its normalized coordinate with the
    wavelength. Thus the spatial extent and resolution of the grid depends on the relevant
    wavelength. Performing the Fourier transform with such a grid has the advantage of
    producing results with the same angular resolution for each wavelength, meaning that
    the diffracted field can be directly integrated over wavelength without spatial
    interpolation.
    '''
    def __init__(self, aperture_diameter, focal_length, field_of_view_x, field_of_view_y, max_angular_coarseness, wavelengths):

        self.aperture_diameter = aperture_diameter # Diameter of the aperture [m]
        self.focal_length = focal_length # Focal length of the optical system [m]
        self.wavelengths = wavelengths # Array of wavelengths [m]
        self.field_of_view_x = field_of_view_x # Field of view covered by the diffracted field in the x-direction [rad]
        self.field_of_view_y = field_of_view_y # Field of view covered by the diffracted field in the y-direction [ra
        self.max_angular_coarseness = max_angular_coarseness # Maximum angle that an image grid cell is allowed to subtend [rad]

        self.n_wavelengths = len(wavelengths)
        self.max_field_of_view = max(field_of_view_x, field_of_view_y) # The resolution of the aperture grid is decided by the largest FOV

        self.setup_aperture_plane()
        self.setup_image_plane()

    def setup_aperture_plane(self):

        # Compute the required extent of a cell in the aperture grid to achieve the desired field of view
        self.normalized_grid_cell_extent = 1/(2*np.tan(self.max_field_of_view/2))

        # Compute the required number of cells in the aperture grid to achieve the desired angular resolution
        self.n_grid_cells = math_utils.nearest_higher_power_of_2(1/(self.max_angular_coarseness*self.normalized_grid_cell_extent))

        # The origin of the aperture coordinates is at the center of the grid
        self.normalized_coordinates = np.arange(-self.n_grid_cells//2, self.n_grid_cells//2)*self.normalized_grid_cell_extent

        # Compute number of grid cells required to cover the largest aperture
        max_normalized_aperture_radius = 0.5*self.aperture_diameter/np.min(self.wavelengths)
        self.n_grid_cells_across_aperture = 2*int(np.ceil(max_normalized_aperture_radius/self.normalized_grid_cell_extent))

        print(self.n_grid_cells, self.n_grid_cells_across_aperture)

        # The largest aperture must not exceed the extent of the grid
        assert self.n_grid_cells_across_aperture <= self.n_grid_cells

        # Compute ranges of indices that just cover the aperture
        self.n_pad_grid_cells = (self.n_grid_cells - self.n_grid_cells_across_aperture)//2
        self.aperture_idx_range = (self.n_pad_grid_cells, self.n_pad_grid_cells + self.n_grid_cells_across_aperture)

        # Extract coordinates covering the aperture
        self.cropped_normalized_coordinates = self.normalized_coordinates[self.aperture_idx_range[0]:self.aperture_idx_range[1]]

        # Note that 'xy'-indexing means that x corresponds to the outer dimension of the array
        self.cropped_normalized_x_coordinate_mesh, \
            self.cropped_normalized_y_coordinate_mesh = np.meshgrid(self.cropped_normalized_coordinates,
                                                                    self.cropped_normalized_coordinates, indexing='xy')

        # Compute values for scaling squared Fourier coeefficients in order to get fluxes
        self.flux_scales = (self.wavelengths*(self.normalized_grid_cell_extent**2/self.focal_length))**2

        # Create instance of incident light field class
        self.incident_light_field = incident_light_fields.IncidentLightField(self.cropped_normalized_coordinates,
                                                                             self.cropped_normalized_coordinates,
                                                                             self.wavelengths)

    def setup_image_plane(self):

        # Compute the spatial extent of a cell in the image plane grid where the diffracted field lives
        self.image_grid_cell_extent = self.focal_length/(self.n_grid_cells*self.normalized_grid_cell_extent)

        # The image plane coordinates are proportional to the spatial frequencies of the aperture grid
        full_image_coordinates = np.arange(-self.n_grid_cells//2, self.n_grid_cells//2)*self.image_grid_cell_extent

        # Compute corresponding angles with respect to the optical axis
        self.full_image_angular_coordinates = np.arctan2(full_image_coordinates, self.focal_length)

        # Compute index ranges for the part of the image spanning the field of view.
        # Outside these ranges, sources will have been filtered out from the incident field.
        # They must still be included in the Fourier transform since we want to have square
        # cells in both the aperture and image grids.
        self.image_idx_range_x = np.searchsorted(self.full_image_angular_coordinates, (-self.field_of_view_x/2, self.field_of_view_x/2))
        self.image_idx_range_y = np.searchsorted(self.full_image_angular_coordinates, (-self.field_of_view_y/2, self.field_of_view_y/2))

        image_x_coordinates = full_image_coordinates[self.image_idx_range_x[0]:self.image_idx_range_x[1]]
        image_y_coordinates = full_image_coordinates[self.image_idx_range_x[0]:self.image_idx_range_x[1]]

        self.image_light_field = image_light_fields.ImageLightField(image_x_coordinates,
                                                                    image_y_coordinates,
                                                                    self.focal_length,
                                                                    self.wavelengths)

    def get_incident_light_field(self):
        return self.incident_light_field

    def get_image_light_field(self):
        return self.image_light_field

    def set_incident_field_values(self, field_values):
        self.incident_light_field.set(field_values)

    def set_image_spectral_fluxes(self, spectral_fluxes):
        self.image_light_field.set(spectral_fluxes)

    def modulate_incident_light_field(self, modulation):
        self.incident_light_field.modulate(modulation)

    def get_aperture_diameter(self):
        return self.aperture_diameter

    def get_focal_length(self):
        return self.focal_length

    def get_n_aperture_grid_cells(self):
        return self.n_grid_cells_across_aperture

    def get_n_pad_grid_cells(self):
        return self.n_pad_grid_cells

    def get_normalized_aperture_grid_cell_extent(self):
        return self.normalized_grid_cell_extent

    def get_full_image_angular_coordinates(self):
        return self.full_image_angular_coordinates

    def add_point_sources(self, polar_angles, azimuth_angles, incident_spectral_fluxes):
        self.incident_light_field.add_point_sources(polar_angles, azimuth_angles,
                                                    incident_spectral_fluxes,
                                                    self.field_of_view_x, self.field_of_view_y)

    def add_extended_source(self, angular_x_coordinates, angular_y_coordinates, incident_spectral_fluxes):
        self.incident_light_field.add_extended_source(angular_x_coordinates, angular_y_coordinates,
                                                      incident_spectral_fluxes,
                                                      self.field_of_view_x, self.field_of_view_y)

    def apply_transmission_mask(self, transmission_mask_function):
        '''
        Masks the incident light field with a boolean function of the spatial aperture coordinates.
        This allows for the aperture to take on arbitrary shapes within its diameter. There will
        be one mask per wavelength, since the number of grid cells covered by the aperture in
        normalized coordinates depends on the wavelength.

        Arguments: Function taking two 3D arrays corresponding to spatial x- and y-meshes for each
                   wavelength. The function must return a boolean array of the same shape with False
                   for every grid cell where the incident field should be set to zero.
        '''
        assert callable(transmission_mask_function)

        # Compute the spatial coordinates of the aperture grid for each wavelength
        x_coordinate_meshes = np.multiply.outer(self.wavelengths, self.cropped_normalized_x_coordinate_mesh)
        y_coordinate_meshes = np.multiply.outer(self.wavelengths, self.cropped_normalized_y_coordinate_mesh)

        self.incident_light_field.apply_transmission_mask(transmission_mask_function(x_coordinate_meshes, y_coordinate_meshes))

    def update_image_light_field(self, incident_field_stage='modulated'):
        '''
        Performs the Fourier transformation of the incident field and subsequent scaling that produces
        the flux distribution focused in the image plane.
        '''

        # Since the incident field only just covers the aperture diameter, it must be padded with zeros
        # at the edges to produce the desired angular resolution in the diffracted image
        field_values = np.pad(self.incident_light_field.get(incident_field_stage),
                              ((0, 0),
                               (self.n_pad_grid_cells, self.n_pad_grid_cells),
                               (self.n_pad_grid_cells, self.n_pad_grid_cells)),
                              'constant')

        # Perform the 2D spatial Fourier transform for each wavelength and shift the result to put the
        # origin of the image plane coordinates in the center
        fourier_coefficients = np.fft.fftshift(np.fft.fft2(field_values, axes=(1, 2)), axes=(1, 2))

        # Trim the image to only include the specified field of view
        trimmed_fourier_coefficients = fourier_coefficients[:, self.image_idx_range_y[0]:self.image_idx_range_y[1],
                                                               self.image_idx_range_x[0]:self.image_idx_range_x[1]]

        # Square the Fourier coefficients and multiply with a wavelength-dependent scale that ensures energy conservation
        spectral_fluxes = (trimmed_fourier_coefficients*np.conj(trimmed_fourier_coefficients)).real*self.flux_scales[:, np.newaxis, np.newaxis]

        self.set_image_spectral_fluxes(spectral_fluxes)

    def save(self, output_path, compressed=False):

        self.incident_light_field.save(output_path, compressed=compressed)

        if compressed:
            np.savez_compressed(output_path,
                                aperture_diameter=self.aperture_diameter,
                                focal_length=self.focal_length,
                                field_of_view_x=self.field_of_view_x,
                                field_of_view_y=self.field_of_view_y,
                                max_angular_coarseness=self.max_angular_coarseness)
        else:
            np.savez(output_path,
                     aperture_diameter=self.aperture_diameter,
                     focal_length=self.focal_length,
                     field_of_view_x=self.field_of_view_x,
                     field_of_view_y=self.field_of_view_y,
                     max_angular_coarseness=self.max_angular_coarseness)


def load_fraunhofer_diffraction_optics(input_path):

    incident_light_field = incident_light_fields.load_incident_light_field(input_path)

    # Add extension if missing
    final_input_path = input_path + ('.npz' if len(input_path.split('.')) == 1 else '')

    arrays = np.load(final_input_path)
    fraunhofer_diffraction_optics = FraunhoferDiffractionOptics(arrays['aperture_diameter'],
                                                                arrays['focal_length'],
                                                                arrays['field_of_view_x'],
                                                                arrays['field_of_view_y'],
                                                                arrays['max_angular_coarseness'],
                                                                incident_light_field.get_wavelengths())

    fraunhofer_diffraction_optics.set_incident_field_values(incident_light_field.get('pure'))

    return fraunhofer_diffraction_optics


def animate_seeing(optical_system, phase_screen, time_step_scale, duration, filter_set=None, incident_field_stage='modulated', image_field_stage='auto', **animation_kwargs):

    if image_field_stage == 'auto':
        if filter_set is not None:
            image_field_stage = 'filtered'
        else:
            image_field_stage = 'convolved'
    elif image_field_stage == 'filtered':
        assert filter_set is not None

    phase_screen.initialize(optical_system)

    time_step = time_step_scale*phase_screen.get_coherence_time()
    n_time_steps = int(np.ceil(duration/time_step)) + 1
    times = np.arange(n_time_steps)*time_step

    def update():
        phase_screen.move_phase_screen(time_step)
        optical_system.modulate_incident_light_field(phase_screen.compute_perturbation_field())
        optical_system.update_image_light_field(incident_field_stage=incident_field_stage)
        if image_field_stage == 'filtered':
            optical_system.get_image_light_field().filter(filter_set)

    image_light_field.animate_image(optical_system.get_image_light_field(), image_field_stage, times, update, **animation_kwargs)


def test():

    from APSimulator import Constants, StarCluster
    from turbulence import TurbulencePhaseScreen, MovingTurbulencePhaseScreen
    from filters import Filter, FilterSet

    n_stars = 1#0000
    FOV_x = math_utils.radian_from_arcsec(15)
    FOV_y = math_utils.radian_from_arcsec(15)
    aperture_diameter = 0.15
    secondary_diameter = 0.2*aperture_diameter
    spider_wane_width = 0.01*aperture_diameter
    focal_length = 0.75
    max_angular_coarseness = math_utils.radian_from_arcsec(0.1)
    fried_parameter_500 = 0.04
    n_subharmonic_levels = 5
    wind_speed = 10.0
    zenith_angle = 0.0
    view_wavelength = 400e-9

    def transmission_mask_function(x, y):
        r = np.sqrt(x**2 + y**2)
        return np.logical_and.reduce((r <= aperture_diameter/2,
                                      r >= secondary_diameter/2,
                                      np.abs(x) >= spider_wane_width,
                                      np.abs(y) >= spider_wane_width))

    target = StarCluster(n_stars, 4, (3000, 30000), (1e-2, 2.0e4), 2e4)
    target_info = target.get_target_info()
    polar_angles, azimuth_angles = target.get_spherical_direction_angles()
    incident_spectral_fluxes = np.swapaxes(target_info.get_incident_field_amplitudes()**2, 0, 1)

    optics = FraunhoferDiffractionOptics(aperture_diameter, focal_length, FOV_x, FOV_y, max_angular_coarseness, Constants.wavelengths)
    optics.add_point_sources(polar_angles, azimuth_angles, incident_spectral_fluxes)
    optics.apply_transmission_mask(transmission_mask_function)

    #turbulence = TurbulencePhaseScreen(fried_parameter_500, 500e-9, zenith_angle, n_subharmonic_levels, Constants.wavelengths)
    turbulence = MovingTurbulencePhaseScreen(fried_parameter_500, 500e-9, zenith_angle, wind_speed, n_subharmonic_levels, Constants.wavelengths)
    #turbulence.initialize(optics)

    filter_set = FilterSet(red=Filter(595e-9, 680e-9), green=Filter(500e-9, 575e-9), blue=Filter(420e-9, 505e-9))
    #filter_set = FilterSet(blue=Filter(400e-9, 405e-9), green=Filter(550e-9, 555e-9), red=Filter(690e-9, 695e-9))

    animate_seeing(optics, turbulence, 2, 0.5, filter_set=filter_set, accumulate=0, output_path='movie.mp4')

if __name__ == '__main__':
    test()

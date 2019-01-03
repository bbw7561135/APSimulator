import numpy as np
import math_utils
import physics_utils
import imaging_system
import sources
import imagers
import apertures
import turbulence
import filters

def test():
    field_of_view_x = math_utils.radian_from_arcsec(120)
    field_of_view_y = math_utils.radian_from_arcsec(120)
    angular_coarseness = math_utils.radian_from_arcsec(0.2)
    wavelengths = np.linspace(400, 700, 50)*1e-9
    aperture_diameter = 0.15
    secondary_diameter = 0.03
    spider_wane_width = 0.006
    focal_length = 0.75

    system = imaging_system.ImagingSystem(field_of_view_x, field_of_view_y, angular_coarseness, wavelengths)
    print(system.source_grid.shape)
    star_field = sources.UniformStarField(physics_utils.inverse_cubic_meters_from_inverse_cubic_parsecs(0.14),
                                          physics_utils.meters_from_parsecs(5),
                                          physics_utils.meters_from_parsecs(15000),
                                          seed=42)
    imager = imagers.FraunhoferImager(aperture_diameter, focal_length)
    aperture = apertures.CircularAperture(aperture_diameter, inner_diameter=secondary_diameter, spider_wane_width=spider_wane_width)
    seeing = turbulence.AveragedKolmogorovTurbulence(0.08)
    filter_set = filters.FilterSet(filters.Filter('red', 595e-9, 680e-9),
                                   filters.Filter('green', 500e-9, 575e-9),
                                   filters.Filter('blue', 420e-9, 505e-9))

    system.set_imager(imager)
    system.set_aperture(aperture)
    system.set_filter_set(filter_set)
    system.add_source('star_field', star_field, store_field=True)
    system.add_image_postprocessor('seeing', seeing)

    print('Seeing FWHM: {:g} focal lengths'.format(np.sin(seeing.compute_approximate_time_averaged_FWHM(wavelengths[20]))))
    print('Rayleigh limit: {:g} focal lengths'.format(np.sin(system.compute_rayleigh_limit(wavelengths[20]))))

    system.run_full_propagation()

    #system.visualize_energy_conservation()

    #system.visualize_aperture_modulation_field('seeing', only_window=0)
    #system.visualize_total_aperture_modulation_field(only_window=1)
    #system.visualize_source_field('star_field', use_log=1)
    #system.visualize_total_source_field()
    #system.visualize_aperture_field()
    #system.visualize_modulated_aperture_field()
    white_point_scale = 0.01
    system.visualize_image_field(use_autostretch=1, white_point_scale=white_point_scale)
    system.visualize_postprocessed_image_field(use_autostretch=1, white_point_scale=white_point_scale)
    system.visualize_filtered_image_field(use_autostretch=1, white_point_scale=white_point_scale)

    #math_utils.save_pyfftw_wisdom()

if __name__ == '__main__':
    test()
    #math_utils.test_pyfftw(use_pyfftw=1, save=0, load=0)

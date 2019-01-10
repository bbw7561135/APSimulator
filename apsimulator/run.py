import numpy as np
import math_utils
import physics_utils
import imaging_system
import sources
import imagers
import apertures
import turbulence
import filters
import cameras
import parallel_utils

def test():
    field_of_view_x = math_utils.radian_from_arcsec(300)
    field_of_view_y = math_utils.radian_from_arcsec(300)
    angular_coarseness = math_utils.radian_from_arcsec(0.2)
    wavelengths = np.linspace(400, 700, 30)*1e-9
    aperture_diameter = 0.15
    secondary_diameter = 0.03
    spider_wane_width = 0.006
    focal_length = 0.75

    system = imaging_system.ImagingSystem(field_of_view_x, field_of_view_y, angular_coarseness, wavelengths)
    print(system.source_grid.shape, system.source_grid.window.shape)
    star_field = sources.UniformStarField(stellar_density=math_utils.inverse_cubic_meters_from_inverse_cubic_parsecs(0.14),
                                          near_distance=math_utils.meters_from_parsecs(2),
                                          far_distance=math_utils.meters_from_parsecs(15000),
                                          seed=42)
    skyglow = sources.UniformBlackbodySkyglow(bortle_class=7, color_temperature=2800)
    imager = imagers.FraunhoferImager(aperture_diameter=aperture_diameter, focal_length=focal_length)
    aperture = apertures.CircularAperture(diameter=aperture_diameter)#, inner_diameter=secondary_diameter, spider_wane_width=spider_wane_width)
    seeing = turbulence.AveragedKolmogorovTurbulence(reference_fried_parameter=0.08, minimum_psf_extent=30)
    filter_set = filters.FilterSet(filters.Filter('red', 595e-9, 680e-9),
                                   filters.Filter('green', 500e-9, 575e-9),
                                   filters.Filter('blue', 420e-9, 505e-9))
    camera = cameras.Camera(filter_set=filter_set)

    system.set_imager(imager)
    system.set_aperture(aperture)
    system.set_camera(camera)
    system.add_source('star_field', star_field, store_field=True)
    system.add_source('skyglow', skyglow, store_field=True)
    system.add_image_postprocessor('seeing', seeing)

    print('Seeing FWHM: {:g} focal lengths'.format(np.sin(seeing.compute_approximate_time_averaged_FWHM(wavelengths[20]))))
    print('Rayleigh limit: {:g} focal lengths'.format(np.sin(system.compute_rayleigh_limit(wavelengths[20]))))

    parallel_utils.set_number_of_threads('auto')

    system.run_full_propagation()
    #system.visualize_energy_conservation()

    #star_field.plot_HR_diagram(absolute=0)

    #system.visualize_energy_conservation()

    #system.visualize_aperture_modulation_field('seeing', only_window=0)
    #system.visualize_total_aperture_modulation_field(only_window=1)
    #system.visualize_source_field('star_field')
    #system.visualize_source_field('skyglow')
    #system.visualize_total_source_field(use_autostretch=1, only_window=0)
    #system.visualize_aperture_field(use_autostretch=1, only_window=0)
    #system.visualize_modulated_aperture_field(use_autostretch=1, only_window=0)
    #system.visualize_image_field(use_autostretch=0)
    #system.visualize_postprocessed_image_field(use_autostretch=0)
    system.visualize_camera_signal_field(use_autostretch=1)#, filter_label='green'#, white_point_scale=0.01)

    for exposure_time in [1, 10, 100]:
        system.capture_exposure(exposure_time)
        system.visualize_captured_camera_signal_field(use_autostretch=1)#, filter_label='green'#, white_point_scale=0.01)

    #math_utils.save_pyfftw_wisdom()

if __name__ == '__main__':
    test()

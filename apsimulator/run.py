import numpy as np
import math_utils
import imaging_system
import sources
import imagers
import apertures

def test():
    field_of_view_x = math_utils.radian_from_arcsec(50)
    field_of_view_y = math_utils.radian_from_arcsec(50)
    angular_coarseness = math_utils.radian_from_arcsec(0.1)
    wavelengths = np.linspace(400, 700, 50)*1e-9
    aperture_diameter = 0.15
    secondary_diameter = 0.03
    spider_wane_width = 0.006
    focal_length = 0.75

    system = imaging_system.ImagingSystem(field_of_view_x, field_of_view_y, angular_coarseness, wavelengths)
    star_field = sources.StarField(1e9, seed=42)
    imager = imagers.FraunhoferImager(aperture_diameter, focal_length)
    aperture = apertures.CircularAperture(aperture_diameter, inner_diameter=secondary_diameter, spider_wane_width=spider_wane_width)

    system.set_imager(imager)
    system.set_aperture(aperture)
    system.add_source('star_field', star_field, keep_field=True)

    system.run_full_propagation()

    #system.visualize_combined_aperture_modulation_field()
    system.visualize_source_field('star_field', only_window=True)
    #system.visualize_combined_source_field()
    #system.visualize_aperture_field()
    system.visualize_modulated_aperture_field(only_window=True)
    system.visualize_image_field(only_window=True)

if __name__ == '__main__':
    test()

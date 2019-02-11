# -*- coding: utf-8 -*-
# This file is part of the APSimulator API.
# Author: Lars Frogner
import numpy as np
import io_utils
import filters
import physics_utils


CIE_X_filter = None
CIE_Y_filter = None
CIE_Z_filter = None
CIE_X_norm = 16684241.637886828
CIE_Y_norm = 16685557.111685598
CIE_Z_norm = 16680098.720950313


def get_CIE_X_filter(filter_label='X'):
    global CIE_X_filter
    if CIE_X_filter is None:
        CIE_X_filter = filters.create_filter_from_file(io_utils.get_path_relative_to_root('data', 'CIE_X.txt'),
                                                       filter_label=filter_label, wavelength_scale=1e-9)
    else:
        CIE_X_filter.set_label(filter_label)
    return CIE_X_filter


def get_CIE_Y_filter(filter_label='Y'):
    global CIE_Y_filter
    if CIE_Y_filter is None:
        CIE_Y_filter = filters.create_filter_from_file(io_utils.get_path_relative_to_root('data', 'CIE_Y.txt'),
                                                       filter_label=filter_label, wavelength_scale=1e-9)
    else:
        CIE_Y_filter.set_label(filter_label)
    return CIE_Y_filter


def get_CIE_Z_filter(filter_label='Z'):
    global CIE_Z_filter
    if CIE_Z_filter is None:
        CIE_Z_filter = filters.create_filter_from_file(io_utils.get_path_relative_to_root('data', 'CIE_Z.txt'),
                                                       filter_label=filter_label, wavelength_scale=1e-9)
    else:
        CIE_Z_filter.set_label(filter_label)
    return CIE_Z_filter


def clamp(value, lower=0, upper=1):
    return np.maximum(lower, np.minimum(upper, value))


def RGB_from_tristimulus(X, Y, Z):
    '''
    Converts the given tristimulus X, Y and Z values to RGB values.
    '''
    R =  3.240479*X - 1.537150*Y - 0.498535*Z
    G = -0.969256*X + 1.875991*Y + 0.041556*Z
    B =  0.055648*X - 0.204043*Y + 1.057311*Z
    return R, G, B


def tristimulus_from_RGB(R, G, B):
    '''
    Converts the given RGB values to tristimulus X, Y and Z values.
    '''
    X = 0.412453*R + 0.357580*G + 0.180423*B
    Y = 0.212671*R + 0.715160*G + 0.072169*B
    Z = 0.019334*R + 0.119193*G + 0.950227*B
    return X, Y, Z


def HSV_from_RGB(R, G, B):

    assert R >= 0 and R <= 1
    assert G >= 0 and G <= 1
    assert B >= 0 and B <= 1

    max_RGB = max((R, G, B))
    min_RGB = min((R, G, B))
    max_min_diff = max_RGB - min_RGB

    if max_min_diff == 0:
        H = 0
    elif max_RGB == R:
        H = 60*(0 + (G - B)/max_min_diff)
    elif max_RGB == G:
        H = 60*(2 + (B - R)/max_min_diff)
    elif max_RGB == B:
        H = 60*(4 + (R - G)/max_min_diff)
    if H < 0:
        H += 360

    S = 0 if max_RGB == 0 else max_min_diff/max_RGB

    V = max_RGB

    return H, S, V


def RGB_from_HSV(H, S, V):
    assert H >= 0 and H <= 360
    assert S >= 0 and S <= 1
    assert V >= 0 and V <= 1

    def f(n):
        k = (n + H/60) % 6
        return V*(1 - S*max(0, min((1, k, 4 - k))))

    R = f(5)
    G = f(3)
    B = f(1)

    return R, G, B


def RGB_from_spectrum(wavelengths, spectrum, spectrum_scale=1):
    CIE_X_filter = get_CIE_X_filter()
    CIE_Y_filter = get_CIE_Y_filter()
    CIE_Z_filter = get_CIE_Z_filter()
    scaled_spectrum = spectrum if spectrum_scale == 1 else spectrum*spectrum_scale
    X = CIE_X_filter.compute_integrated_flux(wavelengths, scaled_spectrum)*CIE_X_norm
    Y = CIE_Y_filter.compute_integrated_flux(wavelengths, scaled_spectrum)*CIE_Y_norm
    Z = CIE_Z_filter.compute_integrated_flux(wavelengths, scaled_spectrum)*CIE_Z_norm
    return RGB_from_tristimulus(X, Y, Z)


def RGB_from_blackbody_temperature(temperature):
    wavelengths = get_CIE_X_filter().get_wavelengths()
    normalized_blackbody_spectral_fluxes = physics_utils.compute_flux_normalized_blackbody_spectral_fluxes(wavelengths, temperature)
    R, G, B = RGB_from_spectrum(wavelengths, normalized_blackbody_spectral_fluxes)
    max_RGB = np.maximum(R, np.maximum(G, B))
    # Rescale so max(R, G, B) = 1, meaning that colors will differ in hue and saturation but have the same intensity
    return np.maximum(0, R/max_RGB), np.maximum(0, G/max_RGB), np.maximum(0, B/max_RGB)

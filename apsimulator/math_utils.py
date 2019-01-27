# -*- coding: utf-8 -*-
# This file is part of the APSimulator API.
# Author: Lars Frogner
import numpy as np
import numba


@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2


@numba.jit
def is_sorted(array):
    for i in range(len(array)-1):
        if array[i+1] < array[i] :
            return False
    return True


def linear_extrapolation(new_x_values, x_values, y_values, side='left'):
    assert side in ('left', 'right')
    left_idx = 0 if side == 'left' else -2
    right_idx = left_idx + 1
    slope = (edge_y_values[right_idx] - edge_y_values[left_idx])/(edge_x_values[right_idx] - edge_x_values[left_idx])
    return edge_y_values[left_idx] + slope*(new_x_values - edge_x_values[left_idx])


def interp_extrap_scalar(new_x_value, x_values, y_values):
    if new_x_value < x_values[0]:
        new_y_value = linear_extrapolation(new_x_value, x_values, y_values, side='left')
    elif new_x_value > x_values[-1]:
        new_y_value = linear_extrapolation(new_x_value, x_values, y_values, side='right')
    else:
        new_y_value = np.interp(new_x_value, x_values, y_values)
    return new_y_value


def interp_extrap_array(new_x_values, x_values, y_values):
    new_y_values = np.interp(new_x_values, x_values, y_values)
    new_y_values[new_x_values < x_values[0]] = linear_extrapolation(new_x_values[new_x_values < x_values[0]], x_values, y_values, side='left')
    new_y_values[new_x_values > x_values[-1]] = linear_extrapolation(new_x_values[new_x_values > x_values[-1]], x_values, y_values, side='right')
    return new_y_values


def interp_extrap(new_x_values, x_values, y_values):
    if isinstance(new_x_values, np.ndarray):
        return interp_extrap_array(new_x_values, x_values, y_values)
    else:
        return interp_extrap_scalar(new_x_values, x_values, y_values)


def meters_from_parsecs(parsecs):
    return 3.0857e16*parsecs


def inverse_cubic_meters_from_inverse_cubic_parsecs(inverse_cubic_parsecs):
    return 3.404e-50*inverse_cubic_parsecs


def arcsec_from_radian(angle):
    return angle*206264.806


def radian_from_arcsec(angle):
    return angle*4.84813681e-6


def square_arcsec_from_steradians(steradians):
    return 4.255e10*steradians


def steradians_from_square_arcsec(square_arcsec):
    return 2.350e-11*square_arcsec


def is_power_of_two(number):
    return number != 0 and ((number & (number - 1)) == 0)


def nearest_power_of_2_exponent(number):
    return int(round(np.log2(number)))


def nearest_power_of_2(number):
    return 2**nearest_power_of_2_exponent(number)


def nearest_lower_power_of_2_exponent(number):
    return int(np.floor(np.log2(number)))


def nearest_lower_power_of_2(number):
    return 2**nearest_lower_power_of_2_exponent(number)


def nearest_higher_power_of_2_exponent(number):
    return int(np.ceil(np.log2(number)))


def nearest_higher_power_of_2(number):
    return 2**nearest_higher_power_of_2_exponent(number)


def angular_distances(angles_1, angles_2):
    differences = (angles_1 - angles_2) % (2*np.pi)
    differences[differences > np.pi] -= 2*np.pi
    return np.abs(differences)


def angular_coordinates_from_spherical_angles(polar_angle, azimuth_angle):
    '''
    Computes the angular x- and y-coordinate of a direction given by a
    spherical polar and azimuth angle.
    '''
    angular_x_coordinate = polar_angle*np.cos(azimuth_angle)
    angular_y_coordinate = polar_angle*np.sin(azimuth_angle)
    return angular_x_coordinate, angular_y_coordinate


def direction_vector_from_spherical_angles(polar_angle, azimuth_angle):
    '''
    Computes the x- and y-component of the unit vector in the direction given by
    a spherical polar and azimuth angle.
    '''
    sin_polar_angle = np.sin(polar_angle)
    direction_vector_x = sin_polar_angle*np.cos(azimuth_angle)
    direction_vector_y = sin_polar_angle*np.sin(azimuth_angle)
    return direction_vector_x, direction_vector_y


def polar_angle_from_direction_vector(direction_vector_x, direction_vector_y):
    '''
    Computes the spherical polar angle corresponding to the given unit direction vector.
    '''
    direction_vector_z_squared = 1 - direction_vector_x**2 - direction_vector_y**2
    polar_angle = np.arccos(np.sqrt(direction_vector_z_squared))
    return polar_angle

def azimuth_angle_from_direction_vector(direction_vector_x, direction_vector_y):
    '''
    Computes the spherical azimuth angle corresponding to the given unit direction vector.
    '''
    azimuth_angle = np.arctan2(direction_vector_y, direction_vector_x)
    return azimuth_angle


def spherical_angles_from_direction_vector(direction_vector_x, direction_vector_y):
    '''
    Computes the spherical polar and azimuth angle corresponding to the given
    unit direction vector.
    '''
    polar_angle = polar_angle_from_direction_vector(direction_vector_x, direction_vector_y)
    azimuth_angle = azimuth_angle_from_direction_vector(direction_vector_x, direction_vector_y)
    return polar_angle, azimuth_angle


def direction_vector_extent_from_polar_angle(polar_angle):
    '''
    Computes the value of a component of the unit direction vector given by
    a polar angle along the direction of that component.
    '''
    return np.sin(polar_angle)


def polar_angle_from_direction_vector_extent(direction_vector_extent):
    '''
    Computes the polar angle along the direction of a component of the unit direction
    vector, given the value of that component.
    '''
    return np.arcsin(direction_vector_extent)


def direction_vector_from_angular_coordinates(angular_x_coordinate, angular_y_coordinate):
    '''
    Computes the x- and y-component of the unit vector in the direction given by
    angular x- and y-coordinates.
    '''
    polar_angle = np.sqrt(angular_x_coordinate**2 + angular_y_coordinate**2)
    scale = np.sinc(polar_angle/np.pi) # This is just sin(polar_angle)/polar_angle, but this form handles zero limit automatically
    direction_vector_x = angular_x_coordinate*scale
    direction_vector_y = angular_y_coordinate*scale
    return direction_vector_x, direction_vector_y

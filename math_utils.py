# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np


def arcsec_from_radian(angle):
    return angle*206264.806


def radian_from_arcsec(angle):
    return angle*4.84813681e-6


def nearest_higher_power_of_2(number):
    return 2**int(np.ceil(np.log2(number)))


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

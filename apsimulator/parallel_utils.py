# -*- coding: utf-8 -*-
# This file is part of the APSimulator API.
# Author: Lars Frogner
import numpy as np
import scipy.signal
import joblib
import multiprocessing as mp
import time


n_threads = 1


def set_number_of_threads(n_threads_argument):
    '''
    Determines a valid number of threads based on the given argument and assigns
    to the global n_threads variable. If n_threads='auto', the number of threads
    will correspond to the total number of logical cpu cores of the machine.
    '''
    global n_threads
    max_threads = mp.cpu_count()
    n_threads = max_threads if n_threads_argument == 'auto' else min(max_threads, max(1, int(n_threads_argument)))


def get_number_of_threads():
    return n_threads


def subranges(n_elements):
    '''
    Iterator that returns n_threads consecutive subranges in the total range [0, n_elements].
    The length of some of the first subranges may be 1 larger than for the later subranges
    in order to match the total number of elements.
    '''
    chunk_size = n_elements//n_threads
    remaining = n_elements % n_threads
    for i in range(remaining):
        yield i*(chunk_size+1), (i+1)*(chunk_size+1)
    for i in range(remaining, n_threads):
        yield remaining + i*chunk_size, remaining + (i+1)*chunk_size


def parallelize_over_axis(input_values, shape, axis, output_dtype, parallel_job, timing=True):
    '''
    Runs the given job concurrently with the given number of threads, parallelizing over the given
    axis of one or more input arrays of the given shape and returning an output array of the given
    shape and dtype. The input_values argument is passed directly to the job function and can be
    of any form that the job function accepts.
    '''

    output_values = np.empty(shape, dtype=output_dtype)

    if timing:
        print('Parallel job started with {:d} threads'.format(n_threads))
        start_time = time.time()

    joblib.Parallel(n_jobs=n_threads, prefer='threads', require='sharedmem')(
        joblib.delayed(parallel_job)(output_values, input_values, start_idx, end_idx) for start_idx, end_idx in subranges(shape[axis]))

    if timing:
        end_time = time.time()
        print('Parallel job ended after time {:g} s'.format(end_time - start_time))

    return output_values


def parallel_fft2_job(output_values, input_values, start_idx, end_idx):
    output_values[start_idx:end_idx, :, :] = np.fft.fft2(input_values[start_idx:end_idx, :, :], axes=(1, 2))


def parallel_centered_fft2_job(output_values, input_values, start_idx, end_idx):
    shifted_values = np.fft.ifftshift(input_values[start_idx:end_idx, :, :], axes=(1, 2))
    fourier_coefficients = np.fft.fft2(shifted_values, axes=(1, 2))
    output_values[start_idx:end_idx, :, :] = np.fft.fftshift(fourier_coefficients, axes=(1, 2))


def parallel_ifft2_job(output_values, input_values, start_idx, end_idx):
    output_values[start_idx:end_idx, :, :] = np.fft.ifft2(input_values[start_idx:end_idx, :, :], axes=(1, 2))


def parallel_centered_ifft2_job(output_values, input_values, start_idx, end_idx):
    shifted_values = np.fft.ifftshift(input_values[start_idx:end_idx, :, :], axes=(1, 2))
    fourier_coefficients = np.fft.ifft2(shifted_values, axes=(1, 2))
    output_values[start_idx:end_idx, :, :] = np.fft.fftshift(fourier_coefficients, axes=(1, 2))


def parallel_fft2(values, centered=True, inverse=False):
    '''
    Computes the 2D FFT of the given 3D array of values in parallel over the first axis.
    '''
    if centered:
        parallel_job = parallel_centered_ifft2_job if inverse else parallel_centered_fft2_job
    else:
        parallel_job = parallel_ifft2_job if inverse else parallel_fft2_job
    return parallelize_over_axis(values, values.shape, 0, 'complex128', parallel_job)


def parallel_fftconvolve_job(output_values, input_values, start_idx, end_idx):
    output_values[start_idx:end_idx, :, :] = scipy.signal.fftconvolve(input_values[0][start_idx:end_idx, :, :],
                                                                      input_values[1][start_idx:end_idx, :, :],
                                                                      mode='same', axes=(1, 2))


def parallel_fftconvolve(values, kernel):
    '''
    Computes the 2D convolution of the given 3D array of values with the given 3D kernel
    in parallel over the first axis. The first axis must be the same size for values and
    kernel but the shapes of the remaining two axes can differ.
    '''
    return parallelize_over_axis((values, kernel), values.shape, 0, values.dtype, parallel_fftconvolve_job)

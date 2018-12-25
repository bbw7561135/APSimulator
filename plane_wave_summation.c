#include <Python.h>
#include <arrayobject.h>
#include <assert.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

// Function declarations
static PyObject* sum_plane_waves(PyObject*, PyObject*);

// Method definition object for this extension, these argumens mean:
// ml_name: The name of the method
// ml_meth: Function pointer to the method implementation
// ml_flags: Flags indicating special features of this method, such as
//          accepting arguments, accepting keyword arguments, being a
//          class method, or being a static method of a class.
// ml_doc:  Contents of this method's docstring
static PyMethodDef plane_wave_summation_methods[] = {
    { "sum_plane_waves", sum_plane_waves, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

// Module definition
// The arguments of this structure tell Python what to call your extension,
// what it's methods are and where to look for it's method definitions
static struct PyModuleDef plane_wave_summation_definition = {
    PyModuleDef_HEAD_INIT,
    "plane_wave_summation",
    NULL,
    -1,
    plane_wave_summation_methods
};

// Module initialization
// Python calls this function when importing your extension. It is important
// that this function is named PyInit_[[your_module_name]] exactly, and matches
// the name keyword argument in setup.py's setup() call.
PyMODINIT_FUNC PyInit_plane_wave_summation(void) {
    Py_Initialize();
    import_array();
    return PyModule_Create(&plane_wave_summation_definition);
}

static PyObject* sum_plane_waves(PyObject* self, PyObject* args)
{
    /*
    Takes a set of plane waves coming from point sources in various directions, and computes the
    combined light field incident on a wavelength-normalized spatial grid. The plane waves are
    assumed to have no initial phase difference, and the (extremely fast) temporal component due
    to the oscillation of the electromagnetic field is neglected.

    Arguments: Array of light field values to update    (complex[n_y_coordinates, n_x_coordinates, n_wavelengths])
               Amplitudes of the plane waves            (float[n_sources, n_wavelengths])
               Array of scaled x-coordinates            (float[n_x_coordinates])
               Array of scaled y-coordinates            (float[n_y_coordinates])
               x- and y-components of direction vectors (float[n_sources])
    */

    // Input arguments
    PyArrayObject* field_values_array;
    PyArrayObject* wave_amplitudes_array;
    PyArrayObject* scaled_x_coordinates_array;
    PyArrayObject* scaled_y_coordinates_array;
    PyArrayObject* direction_vectors_x_array;
    PyArrayObject* direction_vectors_y_array;

    // Dimensions of input data
    int n_x_coordinates;
    int n_y_coordinates;
    int n_wavelengths;
    int n_sources;

    // Pointers to input data
    double complex* field_values;
    double* wave_amplitudes;
    double* scaled_x_coordinates;
    double* scaled_y_coordinates;
    double* direction_vectors_x;
    double* direction_vectors_y;

    // Local variables

    int source_idx;
    int x_idx;
    int y_idx;
    int wavelength_idx;

    int wave_amplitudes_idx_offset;
    int field_values_x_idx_offset;
    int field_values_wavelength_idx_offset;

    double direction_vector_x;
    double direction_vector_y;
    double phase_shift_x;
    double* cos_phase_shifts_x;
    double* sin_phase_shifts_x;
    double phase_shift_y;
    double cos_phase_shift_y;
    double sin_phase_shift_y;
    double complex modulation;

    // Get argument data
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!",
                          &PyArray_Type, &field_values_array,
                          &PyArray_Type, &wave_amplitudes_array,
                          &PyArray_Type, &scaled_x_coordinates_array,
                          &PyArray_Type, &scaled_y_coordinates_array,
                          &PyArray_Type, &direction_vectors_x_array,
                          &PyArray_Type, &direction_vectors_y_array))
    {
        return NULL;
    }

    n_sources = wave_amplitudes_array->dimensions[0];
    n_wavelengths = wave_amplitudes_array->dimensions[1];
    n_x_coordinates = scaled_x_coordinates_array->dimensions[0];
    n_y_coordinates = scaled_y_coordinates_array->dimensions[0];

    assert(field_values_array->dimensions[0] == n_y_coordinates);
    assert(field_values_array->dimensions[1] == n_x_coordinates);
    assert(field_values_array->dimensions[2] == n_wavelengths);
    assert(direction_vectors_x_array->dimensions[0] == n_sources);
    assert(direction_vectors_y_array->dimensions[0] == n_sources);

    field_values = (double complex*)field_values_array->data;
    wave_amplitudes = (double*)wave_amplitudes_array->data;
    scaled_x_coordinates = (double*)scaled_x_coordinates_array->data;
    scaled_y_coordinates = (double*)scaled_y_coordinates_array->data;
    direction_vectors_x = (double*)direction_vectors_x_array->data;
    direction_vectors_y = (double*)direction_vectors_y_array->data;

    // Allocate arrays for precomputed numbers
    cos_phase_shifts_x = (double*)malloc(sizeof(double)*2*n_x_coordinates);
    sin_phase_shifts_x = cos_phase_shifts_x + n_x_coordinates;

    #pragma omp parallel default(shared) private(source_idx, x_idx, y_idx, wavelength_idx, \
                                                wave_amplitudes_idx_offset, field_values_x_idx_offset, field_values_wavelength_idx_offset, \
                                                phase_shift_x, phase_shift_y, cos_phase_shift_y, sin_phase_shift_y, \
                                                modulation)
    {
    for (source_idx = 0; source_idx < n_sources; source_idx++)
    {
        wave_amplitudes_idx_offset = source_idx*n_wavelengths;

        // Precompute cosines and sines of phase shifts in x-direction
        for (x_idx = 0; x_idx < n_x_coordinates; x_idx++)
        {
            phase_shift_x = scaled_x_coordinates[x_idx]*direction_vectors_x[source_idx];
            cos_phase_shifts_x[x_idx] = cos(phase_shift_x);
            sin_phase_shifts_x[x_idx] = sin(phase_shift_x);
        }

        #pragma omp for schedule(static)
        for (y_idx = 0; y_idx < n_y_coordinates; y_idx++)
        {
            field_values_x_idx_offset = y_idx*n_x_coordinates;

            // Compute cosines and sines of phase shifts in y-direction
            phase_shift_y = scaled_y_coordinates[y_idx]*direction_vectors_y[source_idx];
            cos_phase_shift_y = cos(phase_shift_y);
            sin_phase_shift_y = sin(phase_shift_y);

            for (x_idx = 0; x_idx < n_x_coordinates; x_idx++)
            {
                field_values_wavelength_idx_offset = (field_values_x_idx_offset + x_idx)*n_wavelengths,

                // Compute exp(-i*(phase_shift_x + phase_shift_y)).
                modulation = (cos_phase_shifts_x[x_idx]*cos_phase_shift_y - sin_phase_shifts_x[x_idx]*sin_phase_shift_y) -
                             (sin_phase_shifts_x[x_idx]*cos_phase_shift_y + cos_phase_shifts_x[x_idx]*sin_phase_shift_y)*I;

                // Add modulated amplitudes for each wavelength
                for (wavelength_idx = 0; wavelength_idx < n_wavelengths; wavelength_idx++)
                    field_values[field_values_wavelength_idx_offset + wavelength_idx] += wave_amplitudes[wave_amplitudes_idx_offset + wavelength_idx]*modulation;
            }
        }
    }
    }

    free(cos_phase_shifts_x);

    Py_RETURN_NONE;
}

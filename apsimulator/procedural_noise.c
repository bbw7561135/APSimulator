#include <Python.h>
#include <arrayobject.h>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include "open-simplex-noise.h"

// Function declarations
static PyObject* generate_noise_pattern(PyObject*, PyObject*);

// Method definition object for this extension, these argumens mean:
// ml_name: The name of the method
// ml_meth: Function pointer to the method implementation
// ml_flags: Flags indicating special features of this method, such as
//          accepting arguments, accepting keyword arguments, being a
//          class method, or being a static method of a class.
// ml_doc:  Contents of this method's docstring
static PyMethodDef procedural_noise_methods[] = {
    { "generate_noise_pattern", generate_noise_pattern, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

// Module definition
// The arguments of this structure tell Python what to call your extension,
// what it's methods are and where to look for it's method definitions
static struct PyModuleDef procedural_noise_definition = {
    PyModuleDef_HEAD_INIT,
    "procedural_noise",
    NULL,
    -1,
    procedural_noise_methods
};

// Module initialization
// Python calls this function when importing your extension. It is important
// that this function is named PyInit_[[your_module_name]] exactly, and matches
// the name keyword argument in setup.py's setup() call.
PyMODINIT_FUNC PyInit_procedural_noise(void) {
    Py_Initialize();
    import_array();
    return PyModule_Create(&procedural_noise_definition);
}

static PyObject* generate_noise_pattern(PyObject* self, PyObject* args)
{
    /*
    Generates coherent noise values for the 3D mesh defined by the given 1D coordinate
    arrays and inserts them into the give 3D pattern_values array.

    Usage:

    generate_noise_pattern(pattern_values,
                           x_coordinates, y_coordinates, z_coordinates,
                           number_of_octaves,
                           initial_frequency,
                           frequency_scale,
                           spectral_index,
                           exponential_factor,
                           invert_octaves,
                           seed,
                           n_threads)

    The noise value for point (x, y, z) is computed as

    Phi(x, y, z, omega, k0, fs, nu, alpha) = (sum_{k=1,..,omega} Theta((k0 + fs*k)**(-nu)*P((k0 + fs*k)*(x, y, z))))**alpha

    where omega is the number of octaves (frequencies), k0 is the initial frequency,
    fs is the frequency scale, nu is the spectral index, alpha is an exponential
    factor, Theta(x) = x or 1/z and P(x, y, z) is the simplex noise function.
    */

    // Input arguments
    PyArrayObject* pattern_value_array;
    PyArrayObject* x_coordinate_array;
    PyArrayObject* y_coordinate_array;
    PyArrayObject* z_coordinate_array;
    int number_of_octaves;
    double initial_frequency;
    double frequency_scale;
    double spectral_index;
    double exponential_factor;
    int invert_octaves;
    int seed;
    int n_threads;

    double* pattern_values = NULL;
    double* x_coordinates = NULL;
    double* y_coordinates = NULL;
    double* z_coordinates = NULL;

    int size_x, size_y, size_z;
    int i, j, k, f;
    int z_offset, zy_offset;
    double scaled_frequency;
    double octave_value;
    double pattern_value;

    struct osn_context* context;

    // Get argument data
    if (!PyArg_ParseTuple(args, "O!O!O!O!iddddiii",
                          &PyArray_Type, &pattern_value_array,
                          &PyArray_Type, &x_coordinate_array,
                          &PyArray_Type, &y_coordinate_array,
                          &PyArray_Type, &z_coordinate_array,
                          &number_of_octaves,
                          &initial_frequency,
                          &frequency_scale,
                          &spectral_index,
                          &exponential_factor,
                          &invert_octaves,
                          &seed,
                          &n_threads))
    {
        return NULL;
    }

    size_x = x_coordinate_array->dimensions[0];
    size_y = y_coordinate_array->dimensions[0];
    size_z = z_coordinate_array->dimensions[0];

    assert(pattern_value_array->dimensions[0] == size_z);
    assert(pattern_value_array->dimensions[1] == size_y);
    assert(pattern_value_array->dimensions[2] == size_x);

    assert(number_of_octaves > 0);
    assert(initial_frequency >= 0);
    assert(frequency_scale > 0);
    assert(spectral_index > 0);
    assert(exponential_factor != 0);
    assert(invert_octaves == 0 || invert_octaves == 1);
    assert(seed >= 0);
    assert(n_threads > 0);

    pattern_values = (double*)pattern_value_array->data;
    x_coordinates = (double*)x_coordinate_array->data;
    y_coordinates = (double*)y_coordinate_array->data;
    z_coordinates = (double*)z_coordinate_array->data;

    open_simplex_noise(seed, &context);

    #pragma omp parallel for num_threads(n_threads) schedule(static) default(shared) private(i, j, k, f, z_offset, zy_offset, pattern_value, scaled_frequency, octave_value)
    for (k = 0; k < size_z; k++)
    {
        z_offset = k*size_y;

        for (j = 0; j < size_y; j++)
        {
            zy_offset = (z_offset + j)*size_x;

            for (i = 0; i < size_x; i++)
            {
                pattern_value = 0;

                for (f = 1; f < number_of_octaves+1; f++)
                {
                    scaled_frequency = initial_frequency + frequency_scale*f;
                    octave_value = open_simplex_noise3(context, scaled_frequency*x_coordinates[i],
                                                                scaled_frequency*y_coordinates[j],
                                                                scaled_frequency*z_coordinates[k]);
                    octave_value *= pow(1/scaled_frequency, spectral_index); // Could be precomputed
                    pattern_value += (invert_octaves)? 1/octave_value : octave_value;
                }

                pattern_value = pow(pattern_value, exponential_factor);
                pattern_values[zy_offset + i] = pattern_value;
            }
        }
    }

    open_simplex_noise_free(context);

    Py_RETURN_NONE;
}

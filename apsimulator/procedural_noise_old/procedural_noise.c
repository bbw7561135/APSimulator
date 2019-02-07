#include <Python.h>
#include <arrayobject.h>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include "open-simplex-noise.h"

// Function declarations
static PyObject* generate_fractal_noise(PyObject*, PyObject*);

// Method definition object for this extension, these argumens mean:
// ml_name: The name of the method
// ml_meth: Function pointer to the method implementation
// ml_flags: Flags indicating special features of this method, such as
//          accepting arguments, accepting keyword arguments, being a
//          class method, or being a static method of a class.
// ml_doc:  Contents of this method's docstring
static PyMethodDef procedural_noise_methods[] = {
    { "generate_fractal_noise", generate_fractal_noise, METH_VARARGS, NULL},
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

static PyObject* generate_fractal_noise(PyObject* self, PyObject* args)
{
    /*
    Generates coherent fractal noise for the 3D mesh defined by the given 3D coordinate
    arrays and inserts them into the give 3D pattern_values array.
    Usage:
    generate_fractal_noise(noise_values,
                           x_coordinates, y_coordinates, z_coordinates,
                           mask,
                           number_of_octaves,
                           initial_frequency,
                           persistence,
                           seed,
                           n_threads)
    */

    // Input arguments
    PyArrayObject* noise_value_array;
    PyArrayObject* x_coordinate_array;
    PyArrayObject* y_coordinate_array;
    PyArrayObject* z_coordinate_array;
    PyArrayObject* mask_array;
    int number_of_octaves;
    float initial_frequency;
    float persistence;
    int seed;
    int n_threads;

    float* noise_values = NULL;
    float* x_coordinates = NULL;
    float* y_coordinates = NULL;
    float* z_coordinates = NULL;
    unsigned char* mask = NULL;

    int size_x, size_y, size_z;
    int i, j, k, f;
    int x_offset, xy_offset;
    int idx;
    float noise_value;
    float total_amplitude;
    float frequency;
    float amplitude;

    struct osn_context* context;

    // Get argument data
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iffii",
                          &PyArray_Type, &noise_value_array,
                          &PyArray_Type, &x_coordinate_array,
                          &PyArray_Type, &y_coordinate_array,
                          &PyArray_Type, &z_coordinate_array,
                          &PyArray_Type, &mask_array,
                          &number_of_octaves,
                          &initial_frequency,
                          &persistence,
                          &seed,
                          &n_threads))
    {
        return NULL;
    }

    size_x = noise_value_array->dimensions[0];
    size_y = noise_value_array->dimensions[1];
    size_z = noise_value_array->dimensions[2];

    assert(x_coordinate_array->dimensions[0] == size_x);
    assert(x_coordinate_array->dimensions[1] == size_y);
    assert(x_coordinate_array->dimensions[2] == size_z);

    assert(y_coordinate_array->dimensions[0] == size_x);
    assert(y_coordinate_array->dimensions[1] == size_y);
    assert(y_coordinate_array->dimensions[2] == size_z);

    assert(z_coordinate_array->dimensions[0] == size_x);
    assert(z_coordinate_array->dimensions[1] == size_y);
    assert(z_coordinate_array->dimensions[2] == size_z);

    assert(mask_array->dimensions[0] == size_x);
    assert(mask_array->dimensions[1] == size_y);
    assert(mask_array->dimensions[2] == size_z);

    assert(number_of_octaves > 0);
    assert(initial_frequency > 0);
    assert(persistence > 0);
    assert(seed >= 0);
    assert(n_threads > 0);

    noise_values = (float*)noise_value_array->data;
    x_coordinates = (float*)x_coordinate_array->data;
    y_coordinates = (float*)y_coordinate_array->data;
    z_coordinates = (float*)z_coordinate_array->data;
    mask = (unsigned char*)mask_array->data;

    open_simplex_noise(seed, &context);

    #pragma omp parallel for num_threads(n_threads) schedule(static) default(shared) private(i, j, k, f, x_offset, xy_offset, idx, noise_value, total_amplitude, frequency, amplitude)
    for (i = 0; i < size_x; i++)
    {
        x_offset = i*size_y;

        for (j = 0; j < size_y; j++)
        {
            xy_offset = (x_offset + j)*size_z;

            for (k = 0; k < size_z; k++)
            {
                idx = xy_offset + k;

                if (mask[idx] > 0)
                {
                    noise_value = 0;

                    total_amplitude = 0;
                    frequency = initial_frequency;
                    amplitude = 1;

                    for (f = 0; f < number_of_octaves; f++)
                    {
                        noise_value += (float)open_simplex_noise3(context, frequency*x_coordinates[idx],
                                                                           frequency*y_coordinates[idx],
                                                                           frequency*z_coordinates[idx])
                                       *amplitude;

                        total_amplitude += amplitude;
                        frequency *= 2;
                        amplitude *= persistence;
                    }

                    noise_values[idx] = noise_value/total_amplitude;
                }
            }
        }
    }

    open_simplex_noise_free(context);

    Py_RETURN_NONE;
}

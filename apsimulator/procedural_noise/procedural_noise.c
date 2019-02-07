#include <Python.h>
#include <arrayobject.h>
#include <assert.h>
#include "noise_generation.h"

// Function declarations
static PyObject* generate_perlin_noise(PyObject*, PyObject*);

// Method definition object for this extension, these argumens mean:
// ml_name: The name of the method
// ml_meth: Function pointer to the method implementation
// ml_flags: Flags indicating special features of this method, such as
//          accepting arguments, accepting keyword arguments, being a
//          class method, or being a static method of a class.
// ml_doc:  Contents of this method's docstring
static PyMethodDef procedural_noise_methods[] = {
    { "generate_perlin_noise", generate_perlin_noise, METH_VARARGS, NULL},
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

static PyObject* generate_perlin_noise(PyObject* self, PyObject* args)
{
    /*
    Generates coherent fractal noise for the given 3D coordinates. The noise value
    for coordinate i is inserted into the noise_values array at index value_indices[i].
    Usage:
    generate_perlin_noise(noise_values,
                          value_indices,
                          x_coordinates,
                          y_coordinates,
                          z_coordinates,
                          number_of_octaves,
                          initial_frequency,
                          lacunarity,
                          persistence,
                          seed,
                          number_of_threads)
    */

    // Input arguments
    PyArrayObject* noise_value_array;
    PyArrayObject* value_indices_array;
    PyArrayObject* x_coordinate_array;
    PyArrayObject* y_coordinate_array;
    PyArrayObject* z_coordinate_array;
    int number_of_coordinates;
    int number_of_octaves;
    float initial_frequency;
    float lacunarity;
    float persistence;
    int seed;
    int number_of_threads;

    float* noise_values = NULL;
    unsigned int* value_indices = NULL;
    float* x_coordinates = NULL;
    float* y_coordinates = NULL;
    float* z_coordinates = NULL;

    // Get argument data
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!ifffii",
                          &PyArray_Type, &noise_value_array,
                          &PyArray_Type, &value_indices_array,
                          &PyArray_Type, &x_coordinate_array,
                          &PyArray_Type, &y_coordinate_array,
                          &PyArray_Type, &z_coordinate_array,
                          &number_of_octaves,
                          &initial_frequency,
                          &lacunarity,
                          &persistence,
                          &seed,
                          &number_of_threads))
    {
        return NULL;
    }

    number_of_coordinates = value_indices_array->dimensions[0];

    assert(x_coordinate_array->dimensions[0] == number_of_coordinates);
    assert(y_coordinate_array->dimensions[0] == number_of_coordinates);
    assert(z_coordinate_array->dimensions[0] == number_of_coordinates);

    noise_values = (float*)noise_value_array->data;
    value_indices = (unsigned int*)value_indices_array->data;
    x_coordinates = (float*)x_coordinate_array->data;
    y_coordinates = (float*)y_coordinate_array->data;
    z_coordinates = (float*)z_coordinate_array->data;

    generate_perlin_noise(noise_values,
                          value_indices,
                          x_coordinates,
                          y_coordinates,
                          z_coordinates,
                          number_of_coordinates,
                          number_of_octaves,
                          initial_frequency,
                          lacunarity,
                          persistence,
                          seed,
                          number_of_threads);

    Py_RETURN_NONE;
}

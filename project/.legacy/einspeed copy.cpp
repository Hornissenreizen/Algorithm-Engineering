#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "parse_tensor.h"
#include "tensor.h"

#include <omp.h>
#include <iostream>

template <typename T>
static PyObject* einsum(PyObject* self, PyObject* args) {
// *** <Parsing Arguments> ***
    PyObject *t1, *t2;
    const char *s;
    // Parse the input argument as a generic PyObject
    if (!PyArg_ParseTuple(args, "sOO", &s, &t1, &t2)) {
        return NULL;
    }
    // TODO: Check for nullptr
    auto tensor1 = parse_tensor<T>(t1);
    auto tensor2 = parse_tensor<T>(t2);
// *** </Parsing Arguments>


    std::cout << tensor1->data[3] << "\n";

    Py_RETURN_NONE;
}


static PyObject* einsumd(PyObject* self, PyObject* args) {
    return einsum<double>(self, args);
}

static PyObject* einsum(PyObject* self, PyObject* args) {
    return einsum<float>(self, args);
}

static PyMethodDef MyMethods[] = {
    {"einsumd", einsumd, METH_VARARGS, "einsum two double tensors"},
    {"einsum", einsum, METH_VARARGS, "einsum two float tensors"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef einspeed = {
    PyModuleDef_HEAD_INIT,
    "einspeed",
    NULL,
    -1,
    MyMethods
};

PyMODINIT_FUNC PyInit_einspeed(void) {
    import_array(); // Initialize NumPy C API
    return PyModule_Create(&einspeed);
}

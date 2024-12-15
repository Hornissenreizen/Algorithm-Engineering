#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include "compute_einsum.h"
#include "parse_tensor.h"
#include "numpy_types.h"
#include "unroll_loop.h"

static PyObject* einsum(PyObject* const self, PyObject* const args) {
    PyObject *lhs_o, *rhs_o;
    const char *s;

    // Parse the input argument as a generic PyObject
    if (!PyArg_ParseTuple(args, "sOO", &s, &lhs_o, &rhs_o)) {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments are passed to einsum.");
        Py_RETURN_NONE;   
    }

    const PyArrayObject* const a = PyObject_to_PyArrayObject(lhs_o);
    const PyArrayObject* const b = PyObject_to_PyArrayObject(rhs_o);

    const int a_type = PyArray_TYPE(a);
    const int b_type = PyArray_TYPE(b);

    if (a_type != b_type) {
        PyErr_SetString(PyExc_TypeError, "The numpy arrays are expected to have the same data type.");
        Py_RETURN_NONE;
    }

    // pick the correct generic based on a_type
    PyObject *result = nullptr;
    for_<NO_SUPPORTED_NUMPY_TYPES>([&] (auto i) {
        if (SUPPORTED_NUMPY_TYPES[i.value] == a_type) {
            using cpp_type = typename numpy_to_cpp_type<SUPPORTED_NUMPY_TYPES[i.value]>::type;
            result = compute_einsum<cpp_type>(s, parse_tensor<cpp_type>(a), parse_tensor<cpp_type>(b)).to_PyObject();
        }
    });

    if (result) return result;  
    PyErr_SetString(PyExc_TypeError, "Unsupported data type in numpy array.");
    Py_RETURN_NONE;
}

static PyMethodDef MyMethods[] = {
    {"einsum", einsum, METH_VARARGS, "Fast einsum implementation."},
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
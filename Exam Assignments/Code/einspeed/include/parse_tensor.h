#ifndef PARSE_TENSOR_H
#define PARSE_TENSOR_H

#include <Python.h>
#include <numpy/arrayobject.h>
#include "tensor.h"
#include "func.h"

PyArrayObject* PyObject_to_PyArrayObject(PyObject * const _t) {
    // Ensure the input is a NumPy array
    if (!PyArray_Check(_t)) {
        PyErr_SetString(PyExc_TypeError, "Expected a NumPy array");
        return nullptr;
    }
    // Cast to PyArrayObject and check type
    return reinterpret_cast<PyArrayObject*>(_t);
}


template <typename T>
const Tensor<T> parse_tensor(PyArrayObject * const _a) {
    size_t ndim = PyArray_NDIM(_a);
    return Tensor<T>(PyArray_NDIM(_a), cast_all<npy_intp, size_t>(ndim, PyArray_DIMS(_a)), static_cast<T*>(PyArray_DATA(_a)));
};

#endif
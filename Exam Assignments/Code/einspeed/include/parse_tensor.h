#ifndef PARSE_TENSOR_H
#define PARSE_TENSOR_H

#include <Python.h>
#include <numpy/arrayobject.h>
#include "tensor.h"
#include "func.h"

const PyArrayObject* PyObject_to_PyArrayObject(const PyObject * const _t) {
    // Ensure the input is a NumPy array
    if (!PyArray_Check(_t)) {
        PyErr_SetString(PyExc_TypeError, "Expected a NumPy array.");
        return nullptr;
    }
    return reinterpret_cast<const PyArrayObject*>(_t);
}


template <typename T>
const Tensor<T> parse_tensor(const PyArrayObject * const _a) {
    size_t ndim = PyArray_NDIM(_a);
    return Tensor<T>(ndim, cast_all<npy_intp, size_t>(ndim, PyArray_DIMS(const_cast<PyArrayObject*>(_a))), static_cast<T*>(PyArray_DATA(const_cast<PyArrayObject*>(_a))));
};

#endif
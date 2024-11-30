#ifndef TENSOR_H
#define TENSOR_H

#include <Python.h>
#include <numpy/arrayobject.h>
#include <cstddef>
#include "numpy_types.h"
// #include <stdexcept>
// #include <cstring> // For memcpy

template <typename T>
class Tensor {
public:
    size_t ndim;    // Number of dimensions
    size_t* shape;  // Pointer to array defining the shape
    T* data;        // Pointer to the actual data

    Tensor(size_t ndim, size_t* shape, T* data)
        : ndim(ndim), shape(shape), data(data) {}

    ~Tensor() {
        delete[] shape; // It is okay to delete this, since when casting to PyObject, we will create a copy
        // delete[] data; // Don't delete data as it belongs to Python
    }

    PyObject* to_PyObject() const {
        // Initialize NumPy (if not already done)
        if (!PyArray_API) {
            import_array(); // Necessary to initialize NumPy C API
        }

        // Create a NumPy array
        npy_intp* np_shape = new npy_intp[ndim];
        for (size_t i = 0; i < ndim; ++i) {
            np_shape[i] = static_cast<npy_intp>(shape[i]);
        }

        int numpy_type = numpy_type_of<T>();
        
        PyObject* numpy_array = PyArray_SimpleNewFromData(
            static_cast<int>(ndim), np_shape, numpy_type, data
        );

        // Set ownership of the data to NumPy
        PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(numpy_array), NPY_ARRAY_OWNDATA);

        delete[] np_shape; // Cleanup temporary shape array
        return numpy_array;
    }
};

#endif
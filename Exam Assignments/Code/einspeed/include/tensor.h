#ifndef TENSOR_H
#define TENSOR_H

#include <Python.h>
#include <numpy/arrayobject.h>
// #include <cstddef>
#include <vector>
#include "numpy_types.h"
// #include <stdexcept>
#include <cstring> // For memcpy

#include <iostream>

template <typename T>
class Tensor {
public:
    size_t ndim;    // Number of dimensions
    size_t* shape;  // Pointer to array defining the shape
    T* data;        // Pointer to the actual data

    Tensor(size_t ndim, size_t* shape, T* data)
        : ndim(ndim), shape(shape), data(data) {}
    
    // Tensor(size_t ndim, size_t* shape, T* data)
    //     : ndim(ndim), shape(new size_t[ndim]), data(new T[calculate_size(shape, ndim)]) {
    //     std::copy(shape, shape + ndim, this->shape);
    //     std::copy(data, data + calculate_size(shape, ndim), this->data);
    // }

    ~Tensor() {
        delete[] shape; // It is okay to delete this, since when casting to PyObject, we will create a copy (which we will later also delete though)
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

        delete[] np_shape; // it is safe to delete it, as python will make a copy
        return numpy_array;
    }





    // **********
    // Core logic
    // **********

    // Transpose method
    Tensor<T> transpose(const std::vector<size_t>& perm) const {
        if (perm.size() != ndim) {
            throw std::invalid_argument("Permutation size must match the number of dimensions.");
        }

        // Check if perm is a valid permutation
        std::vector<bool> seen(ndim, false);
        for (size_t i : perm) {
            if (i >= ndim || seen[i]) {
                throw std::invalid_argument("Invalid permutation array.");
            }
            seen[i] = true;
        }

        // Compute the new shape
        size_t* new_shape = new size_t[ndim];
        for (size_t i = 0; i < ndim; ++i) {
            new_shape[i] = shape[perm[i]];
        }

        // Allocate memory for the transposed data
        T* new_data = new T[calculate_size(new_shape, ndim)];

        // Compute strides for both old and new tensor layouts
        std::vector<size_t> old_strides = compute_strides(shape, ndim);
        std::vector<size_t> new_strides = compute_strides(new_shape, ndim);

        // Fill the new_data array
        std::vector<size_t> old_indices(ndim, 0);
        for (size_t new_index = 0; new_index < calculate_size(new_shape, ndim); ++new_index) {
            // Map the flat new_index to multi-dimensional indices in the new layout
            size_t remaining = new_index;
            for (size_t i = 0; i < ndim; ++i) {
                old_indices[perm[i]] = remaining / new_strides[i];
                remaining %= new_strides[i];
            }

            // Compute the flat index in the original tensor
            size_t old_index = 0;
            for (size_t i = 0; i < ndim; ++i) {
                old_index += old_indices[i] * old_strides[i];
            }

            // Copy the data
            new_data[new_index] = data[old_index];
        }

        // Create and return the new transposed tensor
        return Tensor<T>(ndim, new_shape, new_data);
    }


    // Reshape method
    Tensor<T>& reshape(const std::vector<size_t>& new_shape) {
        size_t new_size = calculate_size(new_shape.data(), new_shape.size());
        size_t old_size = calculate_size(shape, ndim);

        // Ensure the total number of elements matches
        if (new_size != old_size) {
            std::invalid_argument("New shape of tensor must have the same size as old one.");
        }

        // Update shape and dimensions
        delete[] this->shape; // Free old shape memory
        ndim = new_shape.size();
        this->shape = new size_t[ndim];
        std::copy(new_shape.begin(), new_shape.end(), shape);

        return *this;
    }


private:
    static size_t calculate_size(const size_t* shape, size_t ndim) {
        size_t size = 1;
        for (size_t i = 0; i < ndim; ++i) {
            size *= shape[i];
        }
        return size;
    }

    static std::vector<size_t> compute_strides(const size_t* shape, size_t ndim) {
        std::vector<size_t> strides(ndim);
        strides[ndim - 1] = 1;
        for (int i = static_cast<int>(ndim) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }
};

#endif
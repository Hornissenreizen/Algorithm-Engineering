#ifndef TENSOR_H
#define TENSOR_H

#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <iostream>
#include <cstring>
#include <vector>

#include "numpy_types.h"

template <typename T>
class Tensor {
public:
    size_t ndim = 0;          // Number of dimensions
    size_t* shape = nullptr;  // Pointer to array defining the shape
    T* data = nullptr;        // Pointer to the actual data
                              // IMPORTANT: Tensor class does NOT free the data!

    Tensor(size_t ndim, size_t* shape, T* data)
        : ndim(ndim), data(data) {
            this->shape = new size_t[ndim];
            std::memcpy(this->shape, shape, ndim * sizeof(size_t));
        }
    Tensor() {}
    
    Tensor(Tensor<T> &&other)
        : ndim(other.ndim), shape(other.shape), data(other.data) {
        // Nullify the moved-from object to avoid double deletion
        other.ndim = 0;
        other.shape = nullptr;
        other.data = nullptr;
    }

    Tensor<T>& operator=(Tensor<T> &&other) {
        if (this != &other) {  // Prevent self-assignment
            // Clean up any existing resources (if necessary)
            delete[] shape;
            // Don't delete data as it belongs to Python or is managed elsewhere.

            // Move the resources
            ndim = other.ndim;
            shape = other.shape;
            data = other.data;

            // Nullify the moved-from object
            other.ndim = 0;
            other.shape = nullptr;
            other.data = nullptr;
        }
        return *this;
    }


    ~Tensor() {
        if (this->shape) delete[] this->shape; // It is okay to delete this, since when casting to PyObject, we will create a copy (which we will later also delete though, Python seems to make a copy)
        // Don't delete data as it might belong to Python
    }

    PyObject* to_PyObject() const {
        // Initialize NumPy (if not already done)
        if (!PyArray_API) {
            import_array(); // Necessary to initialize NumPy C API
        }

        int numpy_type = numpy_type_of<T>();

        // If ndim == 0, return a scalar
        if (ndim == 0) {
            if (data) return PyArray_Scalar(data, PyArray_DescrFromType(numpy_type), nullptr);
            else Py_RETURN_NONE;
        }

        // Create a NumPy array
        npy_intp* np_shape = new npy_intp[ndim];
        for (size_t i = 0; i < ndim; ++i) {
            np_shape[i] = static_cast<npy_intp>(shape[i]);
        }

        PyObject* numpy_array = PyArray_SimpleNewFromData(
            static_cast<int>(ndim), np_shape, numpy_type, data
        );

        // Set ownership of the data to NumPy
        PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(numpy_array), NPY_ARRAY_OWNDATA);

        delete[] np_shape; // it is safe to delete it, as Python will make a copy
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
        if (ndim == 0) {
            // needs special treatment
            if (data) {
                T* new_data =  new T[1];
                *new_data = this->data[0];
                return Tensor<T>(ndim, shape, new_data);
            } else return Tensor<T>();
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
        size_t new_size = calculate_size(new_shape);
        size_t old_size = calculate_size(shape, ndim);

        // Ensure the total number of elements matches
        if (new_size != old_size) {
            throw std::invalid_argument("New shape of tensor must have the same size as old one.");
        }

        // Update shape and dimensions
        delete[] this->shape; // Free old shape memory
        this->ndim = new_shape.size();
        this->shape = new size_t[ndim];
        std::copy(new_shape.begin(), new_shape.end(), this->shape);

        return *this;
    }


    // Reduce method
    Tensor<T> reduce() const {
        // reduces over the last dimension
        const size_t last_dimension = this->shape[this->ndim-1];
        const size_t target_size = this->calculate_size(this->shape, this->ndim) / last_dimension;
        T* target_data = new T[target_size];

        for (size_t i = 0; i < target_size; i++) {
            T sum = 0;
            for (size_t k = 0; k < last_dimension; k++) {
                sum += this->data[i * last_dimension + k];
            }
            target_data[i] = sum;
        }
        return Tensor<T>(this->ndim-1, this->shape, target_data);
    }




    // Utility Methods

    bool inline is_scalar() const {
        return ndim == 0 && data;
    }
    
    static size_t calculate_size(const size_t* shape, size_t ndim) {
        size_t size = 1;
        for (size_t i = 0; i < ndim; ++i) {
            size *= shape[i];
        }
        return size;
    }

    static inline size_t calculate_size(const std::vector<size_t>& shape) {
        return Tensor<T>::calculate_size(shape.data(), shape.size());
    }

    size_t inline size() const {
        return Tensor<T>::calculate_size(this->shape, this->ndim);
    }

    void print() const {
        std::cout << "Tensor(";
        for (size_t i = 0; i < ndim; ++i) {
            std::cout << shape[i];
            if (i < ndim - 1) std::cout << ", ";
        }
        std::cout << "):" << std::endl;

        // Print data in a formatted way
        if (ndim) Tensor<T>::print_data_recursive(data, shape, ndim, 0);
    }

private:
    static void print_data_recursive(const T* data, const size_t* shape, size_t ndim, size_t dim) {
        if (dim == ndim - 1) {
            // Base case: print the last dimension as a flat list
            std::cout << "[ ";
            for (size_t i = 0; i < shape[dim]; ++i) {
                std::cout << data[i] << " ";
            }
            std::cout << "]" << std::endl;
        } else {
            // Recursive case: iterate over the current dimension
            std::cout << "[" << std::endl;
            size_t stride = calculate_stride(shape, ndim, dim);
            for (size_t i = 0; i < shape[dim]; ++i) {
                print_data_recursive(data + i * stride, shape, ndim, dim + 1);
            }
            std::cout << "]" << std::endl;
        }
    }

    static size_t calculate_stride(const size_t* shape, size_t ndim, size_t dim) {
        size_t stride = 1;
        for (size_t i = dim + 1; i < ndim; ++i) {
            stride *= shape[i];
        }
        return stride;
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
#ifndef TENSOR_H
#define TENSOR_H

#include <Python.h>
#include <numpy/arrayobject.h>
// #include <cstddef>
#include <vector>
#include <algorithm>
#include "numpy_types.h"
// #include <stdexcept>
#include <cstring> // For memcpy

#include <iostream>

template <typename T>
class Tensor {
public:
    size_t ndim = 0;          // Number of dimensions
    std::vector<size_t> shape = nullptr;  // Pointer to array defining the shape
    T* data = nullptr;        // Pointer to the actual data

    Tensor(size_t ndim, std::vector<size_t> shape, T* data)
        : ndim(ndim), shape(shape), data(data) {};
    Tensor(size_t ndim, size_t *shape, T *data) 
        : ndim(ndim), shape(shape, shape + ndim), data(data) {}
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
        std::cout << "DESTROYING TENSOR:\n";
        // this->print();
        if (this->shape) delete[] this->shape; // It is okay to delete this, since when casting to PyObject, we will create a copy (which we will later also delete though)
        // if (this->data) delete[] this->data; // Don't delete data as it belongs to Python
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
        std::vector<size_t> new_shape = std::vector<size_t>(ndim);
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
        this->shape = new_shape;

        return *this;
    }


    Tensor<T> reduce(const std::vector<size_t> &axes_to_reduce) const {
        // Determine the new shape
        // TODO: implement this method
        return Tensor<T>();
        std::vector<size_t> new_shape;
        std::vector<size_t> reduce_axes_set(axes_to_reduce.begin(), axes_to_reduce.end());

        for (size_t i = 0; i < ndim; i++) {
            if (std::find(reduce_axes_set.begin(), reduce_axes_set.end(), i) == reduce_axes_set.end()) {
                new_shape.push_back(shape[i]);
            }
        }

        if (new_shape.empty()) {
            new_shape.push_back(1); // Reduction to scalar
        }

        size_t new_size = calculate_size(new_shape.data(), new_shape.size());
        size_t original_size = calculate_size(shape, ndim);

        // Create new tensor
        Tensor<T> result(new_shape.size(), new_shape.data(), nullptr);
        std::fill(result.data, result.data + new_size, static_cast<T>(0));

        // Perform reduction
        for (size_t idx = 0; idx < original_size; idx++) {
            // Compute the multi-dimensional index for the original tensor
            size_t multi_idx[ndim];
            size_t temp = idx;
            for (size_t i = ndim; i-- > 0;) {
                multi_idx[i] = temp % shape[i];
                temp /= shape[i];
            }

            // Compute the corresponding index in the reduced tensor
            size_t reduced_idx = 0;
            size_t stride = 1;
            for (size_t i = new_shape.size(); i-- > 0;) {
                if (std::find(reduce_axes_set.begin(), reduce_axes_set.end(), i) == reduce_axes_set.end()) {
                    reduced_idx += multi_idx[i] * stride;
                    stride *= new_shape[i];
                }
            }

            // Add value to the reduced tensor
            result.data[reduced_idx] += data[idx];
        }

        return result;
    }






    // Utility Methods

    static size_t calculate_size(const size_t* shape, size_t ndim) {
        size_t size = 1;
        for (size_t i = 0; i < ndim; ++i) {
            size *= shape[i];
        }
        return size;
    }

    static inline size_t calculate_size(const std::vector<size_t> &shape) {
        return Tensor<T>::calculate_size(shape.data(), shape.size());
    }

    void print() const {
        std::cout << "Tensor(";
        for (size_t i = 0; i < ndim; ++i) {
            std::cout << shape[i];
            if (i < ndim - 1) std::cout << ", ";
        }
        std::cout << "):" << std::endl;

        // Print data in a formatted way
        Tensor<T>::print_data_recursive(data, shape, ndim, 0);
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
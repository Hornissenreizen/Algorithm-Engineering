#ifndef TENSOR_H
#define TENSOR_H

#include <cstddef> // For size_t

template <typename T>
class Tensor {
public:
    size_t ndim;    // Number of dimensions
    size_t* shape;  // Pointer to array defining the shape
    T* data;        // Pointer to the actual data

    Tensor(size_t ndim, size_t* shape, T* data)
        : ndim(ndim), shape(shape), data(data) {}

    ~Tensor() {
        delete[] shape;
        // delete[] data; Don't delete data as it belongs to Python
    }
};

#endif
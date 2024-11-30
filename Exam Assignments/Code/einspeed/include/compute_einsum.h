#ifndef COMPUTE_EINSUM_H
#define COMPUTE_EINSUM_H

#include "tensor.h"

// ******************************************
// The core functionality is implemented HERE
// ******************************************

template <typename T>
PyObject* compute_einsum(const char * const s, const Tensor<T> lhs_tensor, const Tensor<T> rhs_tensor) {
    if constexpr (std::is_same_v<T, double>) {
        lhs_tensor.data[0] = 42;
    }
    return lhs_tensor.to_PyObject();
}

#endif
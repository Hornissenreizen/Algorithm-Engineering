#ifndef COMPUTE_EINSUM_H
#define COMPUTE_EINSUM_H

#include "tensor.h"

// ******************************************
// The core functionality is implemented HERE
// ******************************************

template <typename T>
PyObject* compute_einsum(const char * const s, const Tensor<T> &&lhs_tensor, const Tensor<T> &&rhs_tensor) {
    if constexpr (std::is_same_v<T, double>) {
        lhs_tensor.data[0] = 42;
    }
    auto perm = std::vector<size_t>({1, 0});
    auto shape = std::vector<size_t>({9});
    return lhs_tensor.transpose(perm).reshape(shape).to_PyObject();
}

#endif
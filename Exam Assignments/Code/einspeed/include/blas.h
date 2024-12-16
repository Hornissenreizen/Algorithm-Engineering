#ifndef BLAS_H
#define BLAS_H

#include "tensor.h"

template <typename T>
Tensor<T> batch_matmul(const Tensor<T>& lhs, const Tensor<T>& rhs) {
    // Check dimensions
    if (lhs.ndim != 3 || rhs.ndim != 3) {
        throw std::invalid_argument("Both tensors must have 3 dimensions for batch matrix multiplication.");
    }
    if (lhs.shape[0] != rhs.shape[0]) {
        throw std::invalid_argument("Batch sizes must match for batch matrix multiplication.");
    }
    if (lhs.shape[2] != rhs.shape[1]) {
        throw std::invalid_argument("Inner matrix dimensions must match for batch matrix multiplication.");
    }

    // Shapes
    const size_t batch_size = lhs.shape[0];
    const size_t rows = lhs.shape[1];
    const size_t cols = rhs.shape[2];
    const size_t inner_dim = lhs.shape[2]; // Same as rhs.shape[1]

    // Allocate the result tensor
    size_t result_shape[] = {batch_size, rows, cols};
    T* result_data = new T[batch_size * rows * cols];

    // Perform batch matrix multiplication
    for (size_t b = 0; b < batch_size; ++b) {
        // Get pointers to the current batch matrices
        const T* lhs_batch = lhs.data + b * rows * inner_dim;
        const T* rhs_batch = rhs.data + b * inner_dim * cols;
        T* result_batch = result_data + b * rows * cols;

        // Multiply the two matrices for the current batch
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result_batch[i * cols + j] = 0;
                for (size_t k = 0; k < inner_dim; ++k) {
                    result_batch[i * cols + j] += lhs_batch[i * inner_dim + k] * rhs_batch[k * cols + j];
                }
            }
        }
    }

    return Tensor<T>(3, result_shape, result_data);
}


template <typename T>
Tensor<T> _batch_matrix_times_transpose_matrix(const Tensor<T>& lhs, const Tensor<T>& rhs) {
    // Check dimensions
    if (lhs.ndim != 3 || rhs.ndim != 3) {
        throw std::invalid_argument("Both tensors must have 3 dimensions for batch matrix multiplication.");
    }
    if (lhs.shape[0] != rhs.shape[0]) {
        throw std::invalid_argument("Batch sizes must match for batch matrix multiplication.");
    }
    if (lhs.shape[2] != rhs.shape[2]) {
        throw std::invalid_argument("Inner matrix dimensions must match for batch multiplication with transpose.");
    }

    // Shapes
    const size_t batch_size = lhs.shape[0];
    const size_t rows = lhs.shape[1];
    const size_t cols = rhs.shape[1]; // Since rhs is transposed
    const size_t inner_dim = lhs.shape[2];

    // Allocate the result tensor
    size_t result_shape[] = {batch_size, rows, cols};
    T* result_data = new T[batch_size * rows * cols];

    // Perform batch matrix multiplication with transpose
    #pragma omp parallel for
    for (size_t b = 0; b < batch_size; ++b) {
        // Get pointers to the current batch matrices
        const T* lhs_batch = lhs.data + b * rows * inner_dim;
        const T* rhs_batch = rhs.data + b * cols * inner_dim; // Use rhs.shape[1] because it's transposed
        T* result_batch = result_data + b * rows * cols;

        // Multiply lhs with transposed rhs for the current batch
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result_batch[i * cols + j] = 0;
                for (size_t k = 0; k < inner_dim; ++k) {
                    result_batch[i * cols + j] += lhs_batch[i * inner_dim + k] * rhs_batch[j * inner_dim + k];
                }
            }
        }
    }

    return Tensor<T>(3, result_shape, result_data);
}


template <typename T>
Tensor<T> batch_matrix_times_transpose_matrix(const Tensor<T>& lhs, const Tensor<T>& rhs) {
    // Check dimensions
    if (lhs.ndim != 3 || rhs.ndim != 3) {
        throw std::invalid_argument("Both tensors must have 3 dimensions for batch matrix multiplication.");
    }
    if (lhs.shape[0] != rhs.shape[0]) {
        throw std::invalid_argument("Batch sizes must match for batch matrix multiplication.");
    }
    if (lhs.shape[2] != rhs.shape[2]) {
        throw std::invalid_argument("Inner matrix dimensions must match for batch multiplication with transpose.");
    }

    // Shapes
    const size_t batch_size = lhs.shape[0];
    const size_t rows = lhs.shape[1];
    const size_t cols = rhs.shape[1];
    const size_t inner_dim = lhs.shape[2];

    // Allocate the result tensor
    size_t result_shape[] = {batch_size, rows, cols};
    T* result_data = new T[batch_size * rows * cols];

    // Perform batch matrix multiplication with transpose using OpenMP
    #pragma omp parallel for collapse(3)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                const T* lhs_row = lhs.data + b * rows * inner_dim + i * inner_dim;
                const T* rhs_col = rhs.data + b * cols * inner_dim + j * inner_dim;
                T* result_row = result_data + b * rows * cols + i * cols;

                T sum = T(); // Initialize sum as zero for type T
                for (size_t k = 0; k < inner_dim; ++k) {
                    sum += lhs_row[k] * rhs_col[k];
                }
                result_row[j] = sum;
            }
        }
    }

    return Tensor<T>(3, result_shape, result_data);
}


template <>
Tensor<float> batch_matrix_times_transpose_matrix<float>(const Tensor<float>& lhs, const Tensor<float>& rhs) {
    // Check dimensions
    if (lhs.ndim != 3 || rhs.ndim != 3) {
        throw std::invalid_argument("Both tensors must have 3 dimensions for batch matrix multiplication.");
    }
    if (lhs.shape[0] != rhs.shape[0]) {
        throw std::invalid_argument("Batch sizes must match for batch matrix multiplication.");
    }
    if (lhs.shape[2] != rhs.shape[2]) {
        throw std::invalid_argument("Inner matrix dimensions must match for batch multiplication with transpose.");
    }

    // Shapes
    const size_t batch_size = lhs.shape[0];
    const size_t rows = lhs.shape[1];
    const size_t cols = rhs.shape[1];
    const size_t inner_dim = lhs.shape[2];

    // Allocate the result tensor
    size_t result_shape[] = {batch_size, rows, cols};
    float* result_data = new float[batch_size * rows * cols];

    // Perform batch matrix multiplication with transpose using OpenMP
    #pragma omp parallel for collapse(3)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                const float* lhs_row = lhs.data + b * rows * inner_dim + i * inner_dim;
                const float* rhs_col = rhs.data + b * cols * inner_dim + j * inner_dim;
                float* result_row = result_data + b * rows * cols + i * cols;

                float sum = 0.0f;
                #pragma omp simd reduction(+:sum)
                for (size_t k = 0; k < inner_dim; ++k) {
                    sum += lhs_row[k] * rhs_col[k];
                }
                result_row[j] = sum;
            }
        }
    }

    return Tensor<float>(3, result_shape, result_data);
}


#endif
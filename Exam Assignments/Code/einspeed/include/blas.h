#ifndef BLAS_H
#define BLAS_H

#include "tensor.h"


template <typename T>
Tensor<T> batch_matrix_matmul_transpose(const Tensor<T>& lhs, const Tensor<T>& rhs, T* available_memory, size_t available_memory_size) {
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
    T* result_data;
    size_t result_size = batch_size * rows * cols;
    if (available_memory_size >= result_size) {
#ifdef DEBUG
        std::cout << "Reusing memory\n";
#endif
        result_data = (T*)std::realloc(available_memory, result_size * sizeof(T));
    }    
    else {
        if (available_memory) delete[] available_memory;
        result_data = new T[result_size];
    }

    // Perform batch matrix multiplication with transpose using OpenMP
    #pragma omp parallel for collapse(3)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                const T* lhs_row = lhs.data + b * rows * inner_dim + i * inner_dim;
                const T* rhs_col = rhs.data + b * cols * inner_dim + j * inner_dim;
                T* result_row = result_data + b * rows * cols + i * cols;
                T sum = T(); // Initialize sum as zero for type T

                #pragma omp simd
                for (size_t k = 0; k < inner_dim; ++k) {
                    sum += lhs_row[k] * rhs_col[k];
                }
                result_row[j] = sum;
            }
        }
    }

    return Tensor<T>(3, result_shape, result_data);
}


// template <>
// Tensor<float> batch_matrix_matmul_transpose<float>(const Tensor<float>& lhs, const Tensor<float>& rhs) {
//     // Check dimensions
//     if (lhs.ndim != 3 || rhs.ndim != 3) {
//         throw std::invalid_argument("Both tensors must have 3 dimensions for batch matrix multiplication.");
//     }
//     if (lhs.shape[0] != rhs.shape[0]) {
//         throw std::invalid_argument("Batch sizes must match for batch matrix multiplication.");
//     }
//     if (lhs.shape[2] != rhs.shape[2]) {
//         throw std::invalid_argument("Inner matrix dimensions must match for batch multiplication with transpose.");
//     }

//     // Shapes
//     const size_t batch_size = lhs.shape[0];
//     const size_t rows = lhs.shape[1];
//     const size_t cols = rhs.shape[1];
//     const size_t inner_dim = lhs.shape[2];

//     // Allocate the result tensor
//     size_t result_shape[] = {batch_size, rows, cols};
//     float* result_data = new float[batch_size * rows * cols];

//     // Perform batch matrix multiplication with transpose using OpenMP
//     #pragma omp parallel for collapse(3)
//     for (size_t b = 0; b < batch_size; ++b) {
//         for (size_t i = 0; i < rows; ++i) {
//             for (size_t j = 0; j < cols; ++j) {
//                 const float* lhs_row = lhs.data + b * rows * inner_dim + i * inner_dim;
//                 const float* rhs_col = rhs.data + b * cols * inner_dim + j * inner_dim;
//                 float* result_row = result_data + b * rows * cols + i * cols;

//                 float sum = 0.0f;
//                 #pragma omp simd reduction(+:sum)
//                 for (size_t k = 0; k < inner_dim; ++k) {
//                     sum += lhs_row[k] * rhs_col[k];
//                 }
//                 result_row[j] = sum;
//             }
//         }
//     }

//     return Tensor<float>(3, result_shape, result_data);
// }

// template <>
// Tensor<double> batch_matrix_matmul_transpose<double>(const Tensor<double>& lhs, const Tensor<double>& rhs) {
//     // Check dimensions
//     if (lhs.ndim != 3 || rhs.ndim != 3) {
//         throw std::invalid_argument("Both tensors must have 3 dimensions for batch matrix multiplication.");
//     }
//     if (lhs.shape[0] != rhs.shape[0]) {
//         throw std::invalid_argument("Batch sizes must match for batch matrix multiplication.");
//     }
//     if (lhs.shape[2] != rhs.shape[2]) {
//         throw std::invalid_argument("Inner matrix dimensions must match for batch multiplication with transpose.");
//     }

//     // Shapes
//     const size_t batch_size = lhs.shape[0];
//     const size_t rows = lhs.shape[1];
//     const size_t cols = rhs.shape[1];
//     const size_t inner_dim = lhs.shape[2];

//     // Allocate the result tensor
//     size_t result_shape[] = {batch_size, rows, cols};
//     double* result_data = new double[batch_size * rows * cols];

//     // Perform batch matrix multiplication with transpose using OpenMP
//     #pragma omp parallel for collapse(3)
//     for (size_t b = 0; b < batch_size; ++b) {
//         for (size_t i = 0; i < rows; ++i) {
//             for (size_t j = 0; j < cols; ++j) {
//                 const double* lhs_row = lhs.data + b * rows * inner_dim + i * inner_dim;
//                 const double* rhs_col = rhs.data + b * cols * inner_dim + j * inner_dim;
//                 double* result_row = result_data + b * rows * cols + i * cols;

//                 double sum = 0.0d;
//                 #pragma omp simd reduction(+:sum)
//                 for (size_t k = 0; k < inner_dim; ++k) {
//                     sum += lhs_row[k] * rhs_col[k];
//                 }
//                 result_row[j] = sum;
//             }
//         }
//     }

//     return Tensor<double>(3, result_shape, result_data);
// }


#endif
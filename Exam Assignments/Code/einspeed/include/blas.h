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
        posix_memalign((void**)&result_data, 32, result_size * sizeof(T));
    }

    // data of lhs and rhs used for the omp clause
    const T* lhs_data = lhs.data;
    const T* rhs_data = rhs.data;

    // Perform batch matrix multiplication with transpose using OpenMP
    #pragma omp parallel for simd aligned(result_data: 32) aligned(lhs_data: 32) aligned(rhs_data: 32) collapse(3)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                const T* lhs_row = lhs_data + b * rows * inner_dim + i * inner_dim;
                const T* rhs_col = rhs_data + b * cols * inner_dim + j * inner_dim;
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


template <>
Tensor<float> batch_matrix_matmul_transpose<float>(const Tensor<float>& lhs, const Tensor<float>& rhs, float* available_memory, size_t available_memory_size) {
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
    float* result_data;
    size_t result_size = batch_size * rows * cols;
    if (available_memory_size >= result_size) {
#ifdef DEBUG
        std::cout << "Reusing memory\n";
#endif
        result_data = (float*)std::realloc(available_memory, result_size * sizeof(float));
    }    
    else {
        if (available_memory) delete[] available_memory;
        posix_memalign((void**)&result_data, 32, result_size * sizeof(float));
    }

    // data of lhs and rhs used for the omp clause
    const float* lhs_data = lhs.data;
    const float* rhs_data = rhs.data;

    // Perform batch matrix multiplication with transpose using OpenMP
    #pragma omp parallel for collapse(3)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                const float* lhs_row = lhs_data + b * rows * inner_dim + i * inner_dim;
                const float* rhs_col = rhs_data + b * cols * inner_dim + j * inner_dim;
                float* result_row = result_data + b * rows * cols + i * cols;

                // Initialize sum with AVX
                __m256 vec_sum = _mm256_setzero_ps();

                size_t k = 0;
                for (; k + 7 < inner_dim; k += 8) { // Process 8 elements at a time
                    // Load 8 floats from lhs_row and rhs_col
                    __m256 vec_lhs = _mm256_load_ps(lhs_row + k);
                    __m256 vec_rhs = _mm256_load_ps(rhs_col + k);

                    // Multiply and accumulate
                    __m256 vec_product = _mm256_mul_ps(vec_lhs, vec_rhs);
                    vec_sum = _mm256_add_ps(vec_sum, vec_product);
                }

                // Horizontal sum of vec_sum
                float sum = 0.0f;
                for (int x = 0; x < 8; ++x) {
                    sum += vec_sum[x];
                }

                // Handle the remaining elements (if inner_dim is not a multiple of 8)
                for (; k < inner_dim; ++k) {
                    sum += lhs_row[k] * rhs_col[k];
                }

                result_row[j] = sum;
            }
        }
    }

    return Tensor<float>(3, result_shape, result_data);
}


template <>
Tensor<double> batch_matrix_matmul_transpose<double>(const Tensor<double>& lhs, const Tensor<double>& rhs, double* available_memory, size_t available_memory_size) {
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
    double* result_data;
    size_t result_size = batch_size * rows * cols;
    if (available_memory_size >= result_size) {
#ifdef DEBUG
        std::cout << "Reusing memory\n";
#endif
        result_data = (double*)std::realloc(available_memory, result_size * sizeof(double));
    }    
    else {
        if (available_memory) delete[] available_memory;
        posix_memalign((void**)&result_data, 32, result_size * sizeof(double));
    }

    // data of lhs and rhs used for the omp clause
    const double* lhs_data = lhs.data;
    const double* rhs_data = rhs.data;

    // Perform batch matrix multiplication with transpose using OpenMP
    #pragma omp parallel for collapse(3)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                const double* lhs_row = lhs_data + b * rows * inner_dim + i * inner_dim;
                const double* rhs_col = rhs_data + b * cols * inner_dim + j * inner_dim;
                double* result_row = result_data + b * rows * cols + i * cols;

                // Initialize sum with AVX
                __m256d vec_sum = _mm256_setzero_pd();

                size_t k = 0;
                for (; k + 3 < inner_dim; k += 4) { // Process 4 elements at a time
                    // Load 4 floats from lhs_row and rhs_col
                    __m256d vec_lhs = _mm256_load_pd(lhs_row + k);
                    __m256d vec_rhs = _mm256_load_pd(rhs_col + k);

                    // Multiply and accumulate
                    __m256d vec_product = _mm256_mul_pd(vec_lhs, vec_rhs);
                    vec_sum = _mm256_add_pd(vec_sum, vec_product);
                }

                // Horizontal sum of vec_sum
                double sum = 0.0d;
                for (int x = 0; x < 4; ++x) {
                    sum += vec_sum[x];
                }

                // Handle the remaining elements (if inner_dim is not a multiple of 4)
                for (; k < inner_dim; ++k) {
                    sum += lhs_row[k] * rhs_col[k];
                }

                result_row[j] = sum;
            }
        }
    }

    return Tensor<double>(3, result_shape, result_data);
}

#endif
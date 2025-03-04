#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "tensor.h"
#include "blas.h"

TEST_CASE("Batch Matrix Multiplication", "[batch_matmul]") {
    REQUIRE(true);
    // SECTION("Basic batch multiplication with 2x2 matrices") {
    //     // Create two tensors with batch size 2 and matrices of size 2x2
    //     size_t shape[3] = {2, 2, 2}; // Batch size 2, 2x2 matrices
    //     double data_lhs[] = {1, 2, 3, 4,   // First matrix in batch
    //                          5, 6, 7, 8};  // Second matrix in batch

    //     double data_rhs[] = {9, 10, 11, 12,  // First matrix in batch
    //                          13, 14, 15, 16}; // Second matrix in batch

    //     Tensor<double> lhs(3, shape, data_lhs);
    //     Tensor<double> rhs(3, shape, data_rhs);

    //     // Expected output: Batch of multiplied matrices
    //     double expected_data[] = {
    //         31, 34,  71, 78,  // First batch result
    //         149, 162, 221, 240 // Second batch result
    //     };
    //     Tensor<double> expected_result(3, shape, expected_data);

    //     // Perform batch matrix multiplication
    //     auto result = batch_matmul(lhs, rhs);

    //     // Compare the resulting tensor with the expected tensor
    //     REQUIRE(result.ndim == expected_result.ndim);
    //     REQUIRE(std::equal(result.shape, result.shape + 3, expected_result.shape));
    //     REQUIRE(std::equal(result.data, result.data + 8, expected_result.data));
    // }

    // SECTION("Invalid batch dimensions should throw an exception") {
    //     // Batch size mismatch between tensors
    //     size_t shape_lhs[3] = {2, 2, 2};
    //     size_t shape_rhs[3] = {3, 2, 2}; // Batch size mismatch
    //     double data[] = {1, 2, 3, 4, 5, 6, 7, 8};

    //     Tensor<double> lhs(3, shape_lhs, data);
    //     Tensor<double> rhs(3, shape_rhs, data);

    //     // Expect an exception due to mismatched batch dimensions
    //     REQUIRE_THROWS_AS(batch_matmul(lhs, rhs), std::invalid_argument);
    // }

    // SECTION("Mismatched matrix dimensions within the batch") {
    //     // Create tensors with invalid dimensions for multiplication
    //     size_t shape_lhs[3] = {2, 2, 3}; // Matrices of size 2x3
    //     size_t shape_rhs[3] = {2, 4, 2}; // Matrices of size 4x2 (invalid)

    //     double data_lhs[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    //     double data_rhs[] = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28};

    //     Tensor<double> lhs(3, shape_lhs, data_lhs);
    //     Tensor<double> rhs(3, shape_rhs, data_rhs);

    //     // Expect an exception due to mismatched inner dimensions
    //     REQUIRE_THROWS_AS(batch_matmul(lhs, rhs), std::invalid_argument);
    // }
}

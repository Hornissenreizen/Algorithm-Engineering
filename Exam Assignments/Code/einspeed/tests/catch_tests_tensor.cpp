#include "catch.hpp"
#include "tensor.h"

// Helper function to create a Tensor object with random data for testing
template <typename T>
Tensor<T> create_tensor(const std::vector<size_t>& shape) {
    size_t size = 1;
    for (auto s : shape) size *= s;
    T* data = nullptr;
    posix_memalign((void**)&data, 32, size * sizeof(T));

    // Fill tensor with some test data (e.g., sequential integers)
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<T>(i + 1);
    }

    return Tensor<T>(shape.size(), shape.data(), data);
}

TEST_CASE("Tensor Transpose", "[transpose]") {
    SECTION("Transpose 2D Tensor") {
        std::vector<size_t> shape = {2, 3};
        Tensor<float> tensor = create_tensor<float>(shape);
        std::vector<size_t> perm = {1, 0};  // Swap rows and columns
        Tensor<float> transposed = tensor.transpose(perm);

        REQUIRE(transposed.get_shape() == std::vector<size_t>{3, 2});
        REQUIRE(transposed.data[0] == tensor.data[0]);
        REQUIRE(transposed.data[1] == tensor.data[3]);
        REQUIRE(transposed.data[2] == tensor.data[1]);
        REQUIRE(transposed.data[3] == tensor.data[4]);
        REQUIRE(transposed.data[4] == tensor.data[2]);
        REQUIRE(transposed.data[5] == tensor.data[5]);
    }

    SECTION("Transpose 3D Tensor") {
        std::vector<size_t> shape = {2, 3, 4};
        Tensor<float> tensor = create_tensor<float>(shape);
        std::vector<size_t> perm = {2, 0, 1};  // Move third dim to first place
        Tensor<float> transposed = tensor.transpose(perm);

        REQUIRE(transposed.get_shape() == std::vector<size_t>{4, 2, 3});
    }
}

TEST_CASE("Tensor Reshape", "[reshape]") {
    SECTION("Reshape 1D to 2D Tensor") {
        std::vector<size_t> shape = {6};
        Tensor<float> tensor = create_tensor<float>(shape);
        std::vector<size_t> new_shape = {2, 3};  // Reshape into a 2x3 matrix
        tensor.reshape(new_shape);

        REQUIRE(tensor.get_shape() == new_shape);
    }

    SECTION("Reshape 2D to 1D Tensor") {
        std::vector<size_t> shape = {2, 3};
        Tensor<float> tensor = create_tensor<float>(shape);
        std::vector<size_t> new_shape = {6};  // Flatten it back into a 1D tensor
        tensor.reshape(new_shape);

        REQUIRE(tensor.get_shape() == new_shape);
    }

    SECTION("Reshape with incompatible dimensions") {
        std::vector<size_t> shape = {2, 3};
        Tensor<float> tensor = create_tensor<float>(shape);
        std::vector<size_t> new_shape = {5};  // Invalid reshape
        REQUIRE_THROWS_AS(tensor.reshape(new_shape), std::invalid_argument);
    }
}

TEST_CASE("Tensor Reduce", "[reduce]") {
    SECTION("Reduce 2D Tensor") {
        std::vector<size_t> shape = {3, 4};
        Tensor<double> tensor = create_tensor<double>(shape);
        Tensor<double> reduced = tensor.reduce();  // Reduce over the last dimension

        REQUIRE(reduced.get_shape() == std::vector<size_t>{3});
        REQUIRE(reduced.data[0] == 10);  // 1+2+3+4 = 10
        REQUIRE(reduced.data[1] == 26);  // 5+6+7+8 = 26
        REQUIRE(reduced.data[2] == 42);  // 9+10+11+12 = 42
    }

    SECTION("Reduce 3D Tensor") {
        std::vector<size_t> shape = {3, 3, 4};
        Tensor<double> tensor = create_tensor<double>(shape);
        Tensor<double> reduced = tensor.reduce();  // Reduce over the last dimension

        REQUIRE(reduced.get_shape() == std::vector<size_t>{3, 3});
        for (size_t i = 0; i < 9; i++) {
            REQUIRE(reduced.data[i] == 10 + i * 16);
        }
    }
}

TEST_CASE("Tensor Lazy Transpose and Return PyObject", "[lazy_transpose_and_return_PyObject]") {
    Py_Initialize();
    SECTION("Lazy transpose 2D Tensor") {
        std::vector<size_t> shape = {2, 3};
        Tensor<double> tensor = create_tensor<double>(shape);
        std::vector<size_t> perm = {1, 0};  // Swap rows and columns
        PyObject* numpy_array = tensor.lazy_transpose_and_return_PyObject(perm);

        REQUIRE(numpy_array != nullptr);
    }

    SECTION("Lazy transpose 3D Tensor") {
        std::vector<size_t> shape = {2, 3, 4};
        Tensor<double> tensor = create_tensor<double>(shape);
        std::vector<size_t> perm = {2, 0, 1};  // Move third dim to first place
        PyObject* numpy_array = tensor.lazy_transpose_and_return_PyObject(perm);

        REQUIRE(numpy_array != nullptr);
    }
    Py_Finalize();
}


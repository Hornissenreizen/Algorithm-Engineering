#ifndef COMPUTE_EINSUM_H
#define COMPUTE_EINSUM_H

// uncomment the following line to show debug information
// #define DEBUG

#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <string>

#include "tensor.h"
#include "blas.h"
#include "func.h"


// **************************************************
// *** The core functionality is implemented HERE ***
// **************************************************


// Some declarations (implementation can be found below)
std::vector<size_t> column_identifiers_like(const std::vector<size_t>& target_column_identifiers, const std::string& target_identifier_string, const std::string& origin_identifier_string);
template <typename T> bool is_valid_einsum_expression(const char* const s, const Tensor<T>& lhs_tensor, const Tensor<T>& rhs_tensor, std::string& lhs_string, std::string& rhs_string, std::string& target_string);


template <typename T>
PyObject* compute_einsum(const char* const s, const Tensor<T>& lhs_tensor, const Tensor<T>& rhs_tensor) {

    // *************************
    // *** Function Overview ***
    // *************************

    // We categorize each column identifier c in string lhs_string into one of the following four categories:
    //
    // present in   |  rhs_string  | target_string |
    // ---------------------------------------------
    // batch        |      yes     |      yes      |
    // kept left    |      no      |      yes      |
    // contracted   |      yes     |      no       |
    // summed left  |      no      |      no       |

    // We do the same for rhs_string. Then, we transpose and reshape lhs_tensor and rhs_tensor in the following manner:
    //
    // lhs_tensor[batch, kept left, contracted, summed left]
    // rhs_tensor[batch, kept right, contracted, summed right]
    //
    // IMPORTANT: the column identifiers of batch and contracted of both tensors are obviously the same.
    //            We have to ensure that they also have the same dimensions and ordering!

    // Now, we are ready to perform an axis summation on lhs_tensor over the last dimension (= summed left). The same for rhs_tensor.
    // Then, we can finally use batch matrix matrix multiplication
    // The resulting matrix bmm_result will be of the form
    //
    // bmm_result[batch, kept left, kept right]
    //
    // To transform it back to the desired shape specified by target_string, we need to track the column identifiers.

    std::string lhs_string, rhs_string, target_string;
    is_valid_einsum_expression(s, lhs_tensor, rhs_tensor, lhs_string, rhs_string, target_string); // throws error if not

    
    // Vectors to store categorized index types
    std::vector<size_t> batch_dims_lhs, kept_left_dims_lhs, contracted_dims_lhs, summed_left_dims_lhs;
    std::vector<size_t> batch_dims_rhs, kept_right_dims_rhs, contracted_dims_rhs, summed_right_dims_rhs;

    // Split lhs_string and rhs_string by identifying the roles of each index (batch, contracted, etc.)
    // Staring with lhs_string
    for (size_t i = 0; i < lhs_string.length(); i++) {
        char lhs_char = lhs_string[i];
        if (contains(rhs_string, lhs_char)) {
            if (contains(target_string, lhs_char)) batch_dims_lhs.push_back(i);
            else contracted_dims_lhs.push_back(i);
        } else {
            if (contains(target_string, lhs_char)) kept_left_dims_lhs.push_back(i);
            else summed_left_dims_lhs.push_back(i);
        }
    }

    // now rhs_string
    for (size_t i = 0; i < rhs_string.length(); i++) {
        char rhs_char = rhs_string[i];
        if (!contains(lhs_string, rhs_char)) {
            if (contains(target_string, rhs_char)) kept_right_dims_rhs.push_back(i);
            else summed_right_dims_rhs.push_back(i);  // Summed right dimension
        }
    }

    // We need to reorder the rhs dimensions to match lhs
    batch_dims_rhs = column_identifiers_like(batch_dims_lhs, lhs_string, rhs_string);
    contracted_dims_rhs = column_identifiers_like(contracted_dims_lhs, lhs_string, rhs_string);

    // Now that we have categorized the dimensions, we can begin transposing, reshaping, and performing the axis summation

    // Reshaping the lhs and rhs tensors as described:
    // lhs_tensor[batch, kept left, contracted, summed left]
    // rhs_tensor[batch, kept right, contracted, summed right]
    Tensor<T> transposed_lhs = lhs_tensor.transpose(merge_vectors({batch_dims_lhs, kept_left_dims_lhs, contracted_dims_lhs, summed_left_dims_lhs}));
    Tensor<T> transposed_rhs = rhs_tensor.transpose(merge_vectors({batch_dims_rhs, kept_right_dims_rhs, contracted_dims_rhs, summed_right_dims_rhs}));

#ifdef DEBUG
    std::cout << "Transposed Tensors:" << '\n';
    transposed_lhs.print();
    transposed_rhs.print();
    std::cout << "New Shape of LHS:" << '\n';
    print_vector<size_t>(std::vector<size_t>({Tensor<T>::calculate_size(multi_index(batch_dims_lhs, lhs_tensor.shape)),
                                                Tensor<T>::calculate_size(multi_index(kept_left_dims_lhs, lhs_tensor.shape)),
                                                Tensor<T>::calculate_size(multi_index(contracted_dims_lhs, lhs_tensor.shape)),
                                                Tensor<T>::calculate_size(multi_index(summed_left_dims_lhs, lhs_tensor.shape))}));
#endif

    transposed_lhs.reshape(std::vector<size_t>({Tensor<T>::calculate_size(multi_index(batch_dims_lhs, lhs_tensor.shape)),
                                                Tensor<T>::calculate_size(multi_index(kept_left_dims_lhs, lhs_tensor.shape)),
                                                Tensor<T>::calculate_size(multi_index(contracted_dims_lhs, lhs_tensor.shape)),
                                                Tensor<T>::calculate_size(multi_index(summed_left_dims_lhs, lhs_tensor.shape))}));
    transposed_rhs.reshape(std::vector<size_t>({Tensor<T>::calculate_size(multi_index(batch_dims_rhs, rhs_tensor.shape)),
                                                Tensor<T>::calculate_size(multi_index(kept_right_dims_rhs, rhs_tensor.shape)),
                                                Tensor<T>::calculate_size(multi_index(contracted_dims_rhs, rhs_tensor.shape)),
                                                Tensor<T>::calculate_size(multi_index(summed_right_dims_rhs, rhs_tensor.shape))}));

#ifdef DEBUG
    std::cout << "Tensors after reshaping:\n";
    transposed_lhs.print();
    transposed_rhs.print();
#endif

    // Perform axis summation on lhs_tensor and rhs_tensor for the summed dimensions (summed left, summed right)
    // If, however, the last dimension is one, we simply need to reshape the matrix (there is no need for a reduction)
    Tensor<T> reduced_lhs, reduced_rhs;
    T* available_memory = nullptr; // keeping track of available memory in order to reuse it later
    size_t available_memory_size = 0;
    if (transposed_lhs.shape[3] == 1) {
        reduced_lhs = std::move(transposed_lhs.reshape(std::vector<size_t>(transposed_lhs.shape, transposed_lhs.shape + 3)));
    } else {
        reduced_lhs = std::move(transposed_lhs.reduce());
        available_memory = transposed_lhs.data;
        available_memory_size = transposed_lhs.size();
    }
    // now for rhs
    if (transposed_rhs.shape[3] == 1) {
        reduced_rhs = std::move(transposed_rhs.reshape(std::vector<size_t>(transposed_rhs.shape, transposed_rhs.shape + 3)));
    } else {
        reduced_rhs = std::move(transposed_rhs.reduce());
        size_t transposed_rhs_size = transposed_rhs.size();
        if (transposed_rhs_size > available_memory_size) {
            if (available_memory) delete[] available_memory;
            available_memory = transposed_rhs.data;
            available_memory_size = transposed_rhs_size;
        } else delete[] transposed_rhs.data; // no longer needed
    }

#ifdef DEBUG
    std::cout << "Reduced Matrices:" << '\n';
    reduced_lhs.print();
    reduced_rhs.print();
#endif

    // Finally, perform the batch matrix multiplication.
    Tensor<T> bmm_result;
    {
    if (reduced_rhs.size() == 1 && reduced_rhs.data[0] == T(1)) { // little heuristic to improve performance
        bmm_result = std::move(reduced_lhs);
        delete[] reduced_rhs.data;
#ifdef DEBUG
        std::cout << "skipped batch_matmul!\n";
#endif
    } else if (reduced_lhs.size() == 1 && reduced_lhs.data[0] == T(1)) {
        bmm_result = std::move(reduced_rhs);
        delete[] reduced_rhs.data;
#ifdef DEBUG
        std::cout << "skipped batch_matmul!\n";
#endif
    }
    else {
        bmm_result = batch_matrix_matmul_transpose(reduced_lhs, reduced_rhs, available_memory, available_memory_size);
        // Now we free the data of reduced_lhs and reduced_rhs, as it is no longer needed
        delete[] reduced_lhs.data;
        delete[] reduced_rhs.data;
    }
    }

#ifdef DEBUG
    std::cout << "Result of Batch Matrix Multiplication:\n";
    bmm_result.print();
#endif

    // We are almost done, we only have to reshape the result to the desired output.
    // The current form of bmm_result is

    //     bmm_result = [batch | kept_left | kept_right]
    
    // where batch corresponds to the order present in lhs_string, as well as  kept_left. kept_right corresponds to rhs_string.
    // Hence, we need to firstly reshape bmm_result into the individual dimensions, and then find the correct permutation to transpose it back.

    // Let's start with reshaping it:
    std::vector<size_t> target_shape;
    target_shape.reserve(target_string.size());
    for (auto& i : batch_dims_lhs) target_shape.push_back(lhs_tensor.shape[i]);
    for (auto& i : kept_left_dims_lhs) target_shape.push_back(lhs_tensor.shape[i]);
    for (auto& i : kept_right_dims_rhs) target_shape.push_back(rhs_tensor.shape[i]);

#ifdef DEBUG
    std::cout << "New shape of target matrix:\n";
    print_vector(target_shape);
#endif

    bmm_result.reshape(target_shape);

#ifdef DEBUG
    std::cout << "bmm_result after reshaping:\n";
    bmm_result.print();
#endif

    // Now, we have to transpose it into the correct shape.
    // For that, we have to calculate the permutation in the following manner:

    const std::vector<size_t> target_permutation = get_permutation<char>(merge_vectors<char>({
        multi_index(batch_dims_lhs, lhs_string.data()),
        multi_index(kept_left_dims_lhs, lhs_string.data()),
        multi_index(kept_right_dims_rhs, rhs_string.data()),
    }), std::vector<char>(target_string.begin(), target_string.end()));

    return bmm_result.lazy_transpose_and_return_PyObject(target_permutation);
}


std::vector<size_t> column_identifiers_like(const std::vector<size_t>& target_column_identifiers, const std::string& target_identifier_string, const std::string& origin_identifier_string) {
    // it is assumed that every character in target_identifier_string accessed by target_column_identifiers is also present in origin_identifier_string
    std::vector<size_t> origin_column_identifiers;
    for (size_t i = 0; i < target_column_identifiers.size(); i++) {
        origin_column_identifiers.push_back(origin_identifier_string.find(target_identifier_string[target_column_identifiers[i]]));
    }
    return origin_column_identifiers;
}


template <typename T>
bool is_valid_einsum_expression(const char* const s, const Tensor<T>& lhs_tensor, const Tensor<T>& rhs_tensor, std::string& lhs_string, std::string& rhs_string, std::string& target_string) {
    // we split the string into
    // s = lhs_string , rhs_string -> target_string
    //
    // s hast to satisfy the following criteria:
    // -----------------------------------------
    // 1. every char in target_string must be present in either lhs_string or rhs_string
    // 2. there are no duplicate characters in lhs_string, rhs_string or target_string
    // 3. the size of lhs_string must match the number of dimensions of lhs_tensor. The same for rhs_string and rhs_tensor
    // 4. common column identifiers in lhs_string and rhs_string must correspond to matching dimension sizes in lhs_tensor and rhs_tensor
    
    // Split the string into lhs_string, rhs_string, and target_string
    std::string expr(s);
    size_t comma_pos = expr.find(",");
    size_t arrow_pos = expr.find("->");
    if (comma_pos == std::string::npos || arrow_pos == std::string::npos || arrow_pos < comma_pos) {
        throw std::invalid_argument("Invalid einsum expression: Missing or misplaced ',' or '->' to separate subscripts and target_string.");
    }

    lhs_string = trim_copy((expr.substr(0, comma_pos)));
    rhs_string = trim_copy(expr.substr(comma_pos + 1, arrow_pos - (comma_pos + 1)));
    target_string = trim_copy(expr.substr(arrow_pos + 2));

    // Criterion 1: Ensure every character in target_string appears in either lhs_string or rhs_string
    std::unordered_set<char> valid_chars(lhs_string.begin(), lhs_string.end());
    valid_chars.insert(rhs_string.begin(), rhs_string.end());

    for (char c : target_string) {
        if (valid_chars.find(c) == valid_chars.end()) {
            throw std::invalid_argument("Invalid einsum expression: Target subscript contains invalid characters.");
        }
    }

    // Criterion 2: Ensure there are no duplicate characters in lhs_string, rhs_string, or target_string
    auto has_duplicates = [](const std::string& str) {
        std::unordered_set<char> char_set;
        for (char c : str) {
            if (!char_set.insert(c).second) return true;
        }
        return false;
    };

    if (has_duplicates(lhs_string)) {
        throw std::invalid_argument("Invalid einsum expression: LHS subscripts contain duplicate characters.");
    }
    if (has_duplicates(rhs_string)) {
        throw std::invalid_argument("Invalid einsum expression: RHS subscripts contain duplicate characters.");
    }
    if (has_duplicates(target_string)) {
        throw std::invalid_argument("Invalid einsum expression: Target subscripts contain duplicate characters.");
    }

    // Criterion 3: Ensure the number of subscripts matches the dimensions of the tensors
    if (lhs_string.size() != lhs_tensor.ndim) {
        throw std::invalid_argument("Invalid einsum expression: LHS number of subscripts does not match the number of dimensions of lhs_tensor.");
    }
    if (rhs_string.size() != rhs_tensor.ndim) {
        throw std::invalid_argument("Invalid einsum expression: RHS number of subscripts does not match the number of dimensions of rhs_tensor.");
    }

    // Criterion 4: Ensure common column identifiers in lhs_string and rhs_string have matching dimension sizes
    std::unordered_map<char, size_t> lhs_dim_map;
    for (size_t i = 0; i < lhs_string.size(); ++i) {
        lhs_dim_map[lhs_string[i]] = lhs_tensor.shape[i];
    }

    for (size_t i = 0; i < rhs_string.size(); ++i) {
        char rhs_char = rhs_string[i];
        if (lhs_dim_map.find(rhs_char) != lhs_dim_map.end()) { // If shared identifier
            if (lhs_dim_map[rhs_char] != rhs_tensor.shape[i]) {
                throw std::invalid_argument("Invalid einsum expression: Mismatched dimension sizes for shared subscript '" +
                                            std::string(1, rhs_char) + "'.");
            }
        }
    }

    // If all criteria are satisfied, return true
    return true;
}

#endif
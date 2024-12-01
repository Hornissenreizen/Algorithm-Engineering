#ifndef COMPUTE_EINSUM_H
#define COMPUTE_EINSUM_H

#include <string>
#include <unordered_set>
#include <stdexcept>
#include "tensor.h"

// ******************************************
// The core functionality is implemented HERE
// ******************************************

template <typename T>
PyObject* compute_einsum(const char * const s, const Tensor<T> &&lhs_tensor, const Tensor<T> &&rhs_tensor) {
    if (!is_valid_einsum_expression(s, lhs_tensor, rhs_tensor)) Py_RETURN_NONE;
    auto target_current_column_identifiers = transform_matrices(s, lhs_tensor, rhs_tensor);

    if constexpr (std::is_same_v<T, double>) {
        lhs_tensor.data[0] = 42;
    }
    auto perm = std::vector<size_t>({1, 0});
    auto shape = std::vector<size_t>({9});
    return lhs_tensor.transpose(perm).reshape(shape).to_PyObject();
}


std::vector<size_t> column_identifiers_like(const std::vector<size_t> &target_column_identifiers, const std::string &target_identifier_string, const std::string &origin_identifier_string) {
    // it is assumed that every character in target_identifier_string accessed by target_column_identifiers is also present in origin_identifier_string
    std::vector<size_t> origin_column_identifiers;
    for (size_t i = 0; i < target_column_identifiers.size(); i++) {
        origin_column_identifiers.push_back(origin_identifier_string.find(target_identifier_string[target_column_identifiers[i]]));
    }
    return origin_column_identifiers;
}


// TODO: reread the entire comments to check for errors (since I changed all occurences of target to target_string, conflicting maybe with target_tensor)
template <typename T>
std::vector<char> transform_matrices(const char * const s, const Tensor<T> &lhs_tensor, const Tensor<T> &rhs_tensor) {
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
    // rhs_tensol[batch, contracted, kept right, summed right]
    //
    // IMPORTANT: the column identifiers of batch and contracted of both tensors are obviously the same.
    //            We have to ensure that they also have the same ordering!

    // Now, we are ready to perform an axis summation on lhs_tensor over the last dimension (= summed left). The same for rhs_tensor.
    // Then, we can finally use batch matrix matrix multiplication
    // The result matrix target_string will be of the form
    //
    // target_tensor[batch, kept left, kept right]
    //
    // To transform it back it to the desired shape specified by target_string, we need to track the column identifiers.
    // We return his list of column identifiers.


    std::string expr(s);
    size_t comma_pos = expr.find(",");
    size_t arrow_pos = expr.find("->");

    // Split the expression into lhs_string, rhs_string, and target_string
    std::string lhs_string = trim_copy((expr.substr(0, comma_pos)));
    std::string rhs_string = trim_copy(expr.substr(comma_pos + 1, arrow_pos - (comma_pos + 1)));
    std::string target_string = trim_copy(expr.substr(arrow_pos + 2));

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

    batch_dims_rhs = column_identifiers_like(batch_dims_lhs, lhs_string, rhs_string);
    contracted_dims_rhs = column_identifiers_like(contracted_dims_lhs, lhs_string, rhs_string);

    // Now that we have categorized the dimensions, we can begin transposing, reshaping, and performing the axis summation

    // Reshaping the lhs and rhs tensors as described:
    // lhs_tensor[batch, kept left, contracted, summed left]
    // rhs_tensor[batch, contracted, kept right, summed right]
    Tensor<T> transposed_lhs = lhs_tensor.transpose(merge_vectors({batch_dims_lhs, kept_left_dims_lhs, contracted_dims_lhs, summed_left_dims_lhs}));
    Tensor<T> transposed_rhs = rhs_tensor.transpose(merge_vectors({batch_dims_rhs, contracted_dims_rhs, kept_right_dims_rhs, summed_right_dims_rhs}));

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
                                                Tensor<T>::calculate_size(multi_index(contracted_dims_rhs, rhs_tensor.shape)),
                                                Tensor<T>::calculate_size(multi_index(kept_right_dims_rhs, rhs_tensor.shape)),
                                                Tensor<T>::calculate_size(multi_index(summed_right_dims_rhs, rhs_tensor.shape))}));

#ifdef DEBUG
    lhs_tensor.print();
    transposed_lhs.print();
    rhs_tensor.print();
    transposed_rhs.print();
#endif

    // Perform axis summation on lhs_tensor and rhs_tensor for the summed dimensions (summed left, summed right)
    // If, however, the last dimension is one, we simply need to reshape the matrix (there is no need for a reduction)
    Tensor<T> reduced_lhs, reduced_rhs;
    transposed_lhs.print();
    std::cout << "shape[3] = " << transposed_lhs.shape[3] << '\n';
    if (transposed_lhs.shape[3] == 1) {
        reduced_lhs = std::move(transposed_lhs.reshape(std::vector<size_t>(transposed_lhs.shape, transposed_lhs.shape + 3)));
    } else {
        reduced_lhs = std::move(transposed_lhs.reduce(std::vector<size_t>({3})));
        delete transposed_lhs.data; // no longer needed
    }
    // now for rhs
    if (transposed_rhs.shape[3] == 1) {
        reduced_rhs = std::move(transposed_rhs.reshape(std::vector<size_t>(transposed_rhs.shape, transposed_rhs.shape + 3)));
    } else {
        reduced_rhs = std::move(transposed_rhs.reduce(std::vector<size_t>({3})));
        delete transposed_rhs.data; // no longer needed
    }

#define DEBUG
#ifdef DEBUG
    std::cout << "Reduced Matrices:" << '\n';
    reduced_lhs.print();
    reduced_rhs.print();
#endif
    return std::vector<char>();

    // Finally, perform the batch matrix multiplication. This can be done with a call to a function
    // like `compute_einsum`, which handles the final multiplication after reshaping and summing.

    // To generate the target_string tensor column identifiers, we combine the kept dimensions from the lhs and rhs tensors
    std::vector<char> target_column_identifiers;

    // Add batch dimensions (these are the same for both tensors)
    for (size_t i = 0; i < batch_dims_lhs.size(); ++i) {
        target_column_identifiers.push_back(lhs_string[batch_dims_lhs[i]]);
    }

    // Add kept left dimensions from lhs
    for (size_t i = 0; i < kept_left_dims_lhs.size(); ++i) {
        target_column_identifiers.push_back(lhs_string[kept_left_dims_lhs[i]]);
    }

    // Add kept right dimensions from rhs
    for (size_t i = 0; i < kept_right_dims_rhs.size(); ++i) {
        target_column_identifiers.push_back(rhs_string[kept_right_dims_rhs[i]]);
    }

    // Return the column identifiers of the target_string tensor
    return target_column_identifiers;
}


template <typename T>
bool is_valid_einsum_expression(const char * const s, const Tensor<T> &lhs_tensor, const Tensor<T> &rhs_tensor) {
    // we split the string into
    // s = lhs_string , rhs_string -> target_string
    //
    // s hast to satisfy the following criteria:
    // -----------------------------------------
    // 1. every char in target_string must be present in either lhs_string or rhs_string
    // TODO: implement 2 and 4
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

    // Split the expression into lhs_string, rhs_string, and target_string
    std::string lhs_string = trim_copy((expr.substr(0, comma_pos)));
    std::string rhs_string = trim_copy(expr.substr(comma_pos + 1, arrow_pos - (comma_pos + 1)));
    std::string target_string = trim_copy(expr.substr(arrow_pos + 2));

    // Criterion 1: Ensure every character in target_string appears in either lhs_string or rhs_string
    std::unordered_set<char> valid_chars(lhs_string.begin(), lhs_string.end());
    valid_chars.insert(rhs_string.begin(), rhs_string.end());

    for (char c : target_string) {
        if (valid_chars.find(c) == valid_chars.end()) {
            throw std::invalid_argument("Invalid einsum expression: Target subscript contains invalid characters.");
        }
    }

    // Criterion 2: Ensure the number of subscripts matches the dimensions of the tensors
    if (lhs_string.size() != lhs_tensor.ndim) {
        throw std::invalid_argument("Invalid einsum expression: LHS number of subscripts does not match the number of dimensions of lhs_tensor.");
    }
    if (rhs_string.size() != rhs_tensor.ndim) {
        throw std::invalid_argument("Invalid einsum expression: RHS number of subscripts does not match the number of dimensions of rhs_tensor.");
    }

    // If both criteria are satisfied, return true
    return true;
}

#endif

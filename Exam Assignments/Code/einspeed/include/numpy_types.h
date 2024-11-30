#ifndef NUMPY_TYPES_H
#define NUMPY_TYPES_H

#include <Python.h>
#include <numpy/arrayobject.h>
#include <complex>

template <typename T>
constexpr int numpy_type_of() {throw std::runtime_error("Unsupported data type for NumPy array");}

template<> constexpr int numpy_type_of<long>() {return NPY_INT64;}
template<> constexpr int numpy_type_of<double>() {return NPY_FLOAT64;}
template<> constexpr int numpy_type_of<std::complex<double>>() {return NPY_COMPLEX128;}


// C++ types corresponding to NumPy types
template <int numpy_type>
struct numpy_to_cpp_type;

template <>
struct numpy_to_cpp_type<NPY_INT64> {
    using type = int64_t;
};

template <>
struct numpy_to_cpp_type<NPY_FLOAT64> {
    using type = double;
};

template <>
struct numpy_to_cpp_type<NPY_COMPLEX128> {
    using type = std::complex<double>;
};


const int NUMPY_TYPES[] = {
    NPY_BOOL,
    NPY_INT8,
    NPY_INT16,
    NPY_INT32,
    NPY_INT64,
    NPY_UINT8,
    NPY_UINT16,
    NPY_UINT32,
    NPY_UINT64,
    NPY_FLOAT32,
    NPY_FLOAT64,
    NPY_COMPLEX64,
    NPY_COMPLEX128
};

constexpr int SUPPORTED_NUMPY_TYPES[] = {
    NPY_INT64,
    NPY_FLOAT64,
    NPY_COMPLEX128
};

const size_t NO_SUPPORTED_NUMPY_TYPES = sizeof(SUPPORTED_NUMPY_TYPES) / sizeof(int);

#endif
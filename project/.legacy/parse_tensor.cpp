// #include "parse_tensor.h"
// #include "func.h"

// // Mapping C++ types to NumPy type constants
// template <typename T>
// constexpr int _numpy_type();
// template <>
// constexpr int _numpy_type<double>() { return NPY_DOUBLE; }
// template <>
// constexpr int _numpy_type<float>() { return NPY_FLOAT; }
// template <>
// constexpr int _numpy_type<int>() { return NPY_INT; }
// template <>
// constexpr int _numpy_type<long>() { return NPY_LONG; }


// template <typename T>
// Tensor<T>* parse_tensor(PyObject *_t) {
//     // Ensure the input is a NumPy array
//     if (!PyArray_Check(_t)) {
//         PyErr_SetString(PyExc_TypeError, "Expected a NumPy array");
//         return nullptr;
//     }

//     // Cast to PyArrayObject and check type
//     PyArrayObject* _a = reinterpret_cast<PyArrayObject*>(_t);
//     if (PyArray_TYPE(_a) != _numpy_type<T>()) {
//         // TODO: customize Error message
//         PyErr_SetString(PyExc_TypeError, "Expected a NumPy array of type double");
//         return NULL;
//     }

//     // Get number of dimensions and shape
//     size_t ndim = (size_t) PyArray_NDIM(_a);
//     // npy_intp* shape = PyArray_DIMS(array);  // Array of dimension sizes

//     // Get a pointer to the data
//     // double* data = static_cast<double*>(PyArray_DATA(array));

//     return new Tensor<T>(PyArray_NDIM(_a), cast_all<npy_intp, size_t>(ndim, PyArray_DIMS(_a)), static_cast<T*>(PyArray_DATA(_a)));
// }
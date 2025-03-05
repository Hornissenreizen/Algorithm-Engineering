** Requirements **

    The following tools need to be installed in order to compile and use the implementation:
        - python3
        - numpy
        - make
        - cmake


** Synopsis **

    Build the einspeed.so python module with:

        $ make

    In case it does not work one might try to remove the cache files from cmake.

    The module will be created in the current directory. After that, one can execute the python file einspeed.py to use the module.

    To run the unit tests found in ./tests/catch_tests_tensor.cpp just run:

        $ make test

    The test cases cover subroutines used by the main method. The main method compute_einsum in ./include/compute_einsum.h itself does not have unit tests, as its correctness can directly be verified by the outputs of that method. Though, there is the option to define DEBUG such that debug information will be printed along the execution of the function.


** Project Structure **

    - top level
        We have the final einspeed.so module and the top level python file to execute the code, as well as CMakeLists.txt and Makefile files to build the project.
    
    - build
        Used by cmake to build the project.
    
    - external
        External C++ library to transpose tensors. Source: https://github.com/springer13/hptt
    
    - include
        This is were most of the source code is located. The C++ files are implemented as .h files because of the excessive use of templates and in order to keep the project simple, note that most non template implementations are relatively small anyways. Let's quickly cover each file in there:

            - blas.h
                Implements the batch matrix multiplication method, were the second operand batch of matrices is interpreted as transposed. The implementation is very efficient and vectorized, specialized methods can be found as well.
            
            - compute_einsum.h
                This file is used to implement the core functionality of einsum, it is the top level function and acts as the main execution flow. It calls various subroutines and implements functionality itself in order to correctly compute the einsum expression passed to it. It is rather long and might be a bit tougher to read. It helps to read the comments and trying to understand the steps necessary to implement einsum.
            
            - func.h
                This file just includes some useful smaller subroutines used by other methods in the project.
            
            - numpy_types.h
                This file is used to implement the conversion from NumPy types to native C++ types.
            
            - parse_tensor.h
                In there is just a small casting method casting from the NumPy array to the native C++ type to then initialize a tensor defined in:
            
            - tensor.h
                Defines the tensor data structure. It is implemented as a multi indexed flat array. The most important methods are transpose and reduce which implement the usual tensor functionality. Those implementations and specializations can be found at the bottom of the file. The external library for transposing tensors is used for the transpose implementation.
            
            - unroll_loop.h
                This is very hard to read, but what it essentially does is to facilitate the type conversion (in ./src/einspeed.cpp) from Python types to C++ types. This way, we don't need to specify each type manually, but instead we can loop over an array using this special for_ implementation. This is very useful, because now we just have to alter numpy_types.h accordingly in order to augment our data type support.
    
    - src
        We have a CMakeLists.txt to build the project and the entry point file einspeed.cpp. This file uses the Python C++ interface in order to provide the implemented einsum method to Python. It also does some very basic parsing, and already checks for the NumPy type in order to call the correct template method.


** Specification **

    Import the module in Python using:

        import einspeed
    
    This module provides only one method:

        einspeed.einsum(format_string, lhs_tensor, rhs_tensor)
    
    This is a very basic einsum implementation and works similar to numpy.einsum, though not all features are implemented. The format string and both operand tensors specified by multi-dimensional NumPy arrays must be compatible, this includes their dimensions, the dimension sizes, and their data types must match. Supported data types are np.float32, np.float64, np.complex64, and np.complex128. The format string can be used to specify batch, contracted, kept, and reduced dimensions of the operand tensors, as well as the shape of the output tensor. This fully describes the functionality of this method.

    It returns a multi-dimensional NumPy array representing the result of the tensor operations specified by the format string on the operand tensors. In case of a scalar output, an actual scalar is returned.

    For further analysis please view the paper.
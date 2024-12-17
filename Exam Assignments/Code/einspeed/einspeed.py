import einspeed
import numpy as np
import time

dtype = np.complex128
epsilon = 1e-2 # we need to be generous here, as there is a big computing error on large tensors
size = 50
matrix_lhs = (1+np.random.rand(size, size, size)).astype(dtype)
matrix_rhs = (1+np.random.rand(size, size, size)).astype(dtype)
# matrix_lhs = np.array([[[1,2],[3,4]],[[5,6], [7,8]]], dtype)
# matrix_rhs = matrix_lhs
# matrix_rhs = np.array([1], dtype)
copy = np.copy(matrix_lhs)
vector = np.array([1, 2, 3], dtype)
# string = "abc,def->" # einsum implementation is HORRENDOUS at reducing
string = "abc,def->"

start_time = time.time()
correct = np.einsum(string, matrix_lhs, matrix_rhs)
print(f"correct result in {time.time() - start_time} seconds")
print(correct.strides)

start_time = time.time()
result = einspeed.einsum(string, matrix_lhs, matrix_rhs)
print(f"result in {time.time() - start_time} seconds")
print("Correct: ", (np.abs((result-correct)) < epsilon).all() and (matrix_lhs==copy).all())
print(result.strides)
print(correct, result)
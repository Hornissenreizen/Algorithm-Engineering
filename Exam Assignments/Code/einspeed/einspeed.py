import einspeed
import numpy as np
import time

dtype = np.complex128
matrix_lhs = np.random.rand(2, 2, 2).astype(dtype)
# matrix_rhs = np.random.rand(100, 100, 100).astype(dtype)
matrix_rhs = np.array([1], dtype)
copy = np.copy(matrix_lhs)
vector = np.array([1, 2, 3], dtype)
string = "bik,a->kib"

start_time = time.time()
correct = np.einsum(string, matrix_lhs, matrix_rhs)
print(f"correct result in {time.time() - start_time} seconds")

start_time = time.time()
result = einspeed.einsum(string, matrix_lhs, matrix_rhs)
print(f"result in {time.time() - start_time} seconds")

print("Correct: ", (result==correct).all() and (matrix_lhs==copy).all())
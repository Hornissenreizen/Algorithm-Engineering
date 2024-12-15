import einspeed
import numpy as np

dtype = np.complex128
matrix = np.array([[1.0, 2.0, 3.0+4j], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype)
copy = np.copy(matrix)
vector = np.array([1, 2, 3], dtype)
string = "ab,cd->dcba"

result = einspeed.einsum(string, matrix, matrix)
correct = np.einsum(string, matrix, matrix)

print("correct result:")
print(correct)

print("result:")
print(result)

print("Correct: ", (result==correct).all() and (matrix==copy).all())
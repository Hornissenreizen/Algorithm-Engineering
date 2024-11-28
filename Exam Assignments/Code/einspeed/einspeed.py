import einspeed
import numpy as np

# Pass and process a large matrix
matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)
einspeed.einsumd("", matrix, matrix)

matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
einspeed.einsum("", matrix, matrix)

print("Updated matrix:")
print(matrix)
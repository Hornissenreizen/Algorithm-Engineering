import einspeed
import numpy as np
import time
import csv

def benchmark_einsum(format_string, lhs_shape, rhs_shape, dtype = np.float32, test_case = "Default", epsilon = 1e-4):
    lhs_tensor = (1+np.random.rand(*lhs_shape)).astype(dtype)
    rhs_tensor = (1+np.random.rand(*rhs_shape)).astype(dtype)
    
    lhs_copy = np.copy(lhs_tensor)
    rhs_copy = np.copy(rhs_tensor)

    start_time = time.time()
    correct = np.einsum(format_string, lhs_tensor, rhs_tensor)
    numpy_time = time.time() - start_time

    start_time = time.time()
    result = einspeed.einsum(format_string, lhs_copy, rhs_copy)
    einspeed_time = time.time() - start_time

    maximum_relative_discrepancy = np.max(np.abs((result - correct) / correct))

    print(f"Test Case: {test_case} {str(dtype)}")
    print("Correct: ", (maximum_relative_discrepancy < epsilon) and (lhs_tensor==lhs_copy).all() and (rhs_tensor==rhs_copy).all())
    print(f"Maximum relative discrepancy: {maximum_relative_discrepancy}")
    print(f"NumPy: {numpy_time} s\neinspeed: {einspeed_time} s")
    print(f"Speedup: {numpy_time / einspeed_time}")
    print("\n")

    return (numpy_time, einspeed_time, maximum_relative_discrepancy)


def create_csv(file_name, *args):
    """
    Creates a CSV file from column data provided as tuples.

    Parameters:
        file_name (str): The name of the CSV file to create.
        *args (tuple): Each tuple contains a column name as the first element
                       and a list or numpy array of column data as the second element.
    """
    # Ensure all column data lengths are consistent
    num_rows = len(args[0][1]) if args else 0
    for _, data in args:
        if len(data) != num_rows:
            raise ValueError("All columns must have the same number of rows.")
    
    # Write to the CSV file
    with open(file_name, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # Write header
        headers = [col_name for col_name, _ in args]
        writer.writerow(headers)
        
        # Write rows
        for row_idx in range(num_rows):
            row = [data[row_idx] for _, data in args]
            writer.writerow(row)


def run_test_cases():
    DTYPES = [np.float32, np.float64, np.complex128]
    for dtype in DTYPES:
        # There will be a big discrepancy for np.float32, but einspeed's result is actually CLOSER to the correct result!

        # Reduction
        # benchmark_einsum("abc,def->", [50, 42, 55], [63, 69, 127], dtype, "Multidimensional Reduction")

        # # Contraction
        # benchmark_einsum("i,i->", [600000000], [600000000], dtype, "Vector Dot Product")
        # benchmark_einsum("ij,ij->", [9000, 80000], [9000, 80000], dtype, "Element-wise Contraction")
        # benchmark_einsum("ij,ji->", [9000, 80000], [80000, 9000], dtype, "Element-wise Contraction with Transpose")
        # benchmark_einsum("ijk,ijk->", [1000, 2000, 300], [1000, 2000, 300], dtype, "High-Dimensional Contraction")

        # # Outer product
        # benchmark_einsum("i,j->ij", [100000], [20000], dtype, "Outer Product")
        # benchmark_einsum("ij,k->ijk", [300, 2000], [500], dtype, "Outer Product 3D")
        
        # Transpose-like operations
        benchmark_einsum("ij,jk->ik", [300, 400], [400, 500], dtype, "Matrix Multiplication")
        benchmark_einsum("ijk,jl->ikjl", [100, 300, 300], [300, 400], dtype, "Tensor Contraction")
        
        # Broadcasting
        benchmark_einsum("ij,j->ij", [300, 400], [400], dtype, "Broadcasting along one axis")
        benchmark_einsum("ij,kl->ijkl", [100, 200], [300, 40], dtype, "Full Tensor Broadcasting")

        # Permutations
        benchmark_einsum("ij,k->ji", [3000, 4000], [1], dtype, "Transpose Simulation")
        benchmark_einsum("ijk,l->ikj", [100, 200, 300], [1], dtype, "3D Permutation")

        # Mixed operations
        benchmark_einsum("ijk,il->jkl", [100, 200, 300], [100, 150], dtype, "Mixed Contraction")
        # benchmark_einsum("ij,jk,kl->il", [30, 40], [40, 50], [50, 60], dtype, "Chained Multiplication")
        
        # Single-dimension cases (simulating single operand)
        benchmark_einsum("i,i->i", [1000], [1000], dtype, "Element-wise Multiplication")
        benchmark_einsum("i,j->", [100000], [1], dtype, "Sum Reduction")
        
        # Miscellaneous
        benchmark_einsum("abc,ab->c", [200, 300, 400], [200, 300], dtype, "Reduction along last axis")
        # benchmark_einsum("a,b,c->abc", [10], [20], [30], dtype, "Tensor Construction")
        # benchmark_einsum("ij,jk,kl->il", [20, 30], [30, 40], [40, 50], dtype, "Complex Chain Contraction")


def scaling_study(format_string, lhs_shape, rhs_shape, dtype = np.float32, test_case = "Default", epsilon = 1e-4):
    lhs_shape = np.array(lhs_shape, np.int32)
    rhs_shape = np.array(rhs_shape, np.int32)
    alpha = 1
    problem_sizes = []
    numpy_times = []
    einspeed_times = []
    speedups = []
    for i in range(7):
        (numpy_time, einspeed_time, _) = benchmark_einsum(format_string, alpha * lhs_shape, alpha * rhs_shape, dtype, f"{test_case} ; alpha = {alpha}", epsilon)
        problem_sizes.append(np.prod(alpha * lhs_shape) + np.prod(alpha * rhs_shape)) # this is not very accurate for every numpy expression, but for the tested reduction operation it is
        numpy_times.append(numpy_time)
        einspeed_times.append(einspeed_time)
        speedups.append(numpy_time / einspeed_time)
        alpha = alpha * 2
    create_csv("scaling_study.csv", ("Problem size", problem_sizes), ("Numpy time (s)", numpy_times), ("Einspeed time (s)", einspeed_times), ("Speedup", speedups))


# scaling_study("abc,def->", [1, 2, 3], [3, 2, 1], np.float32, "Multidimensional Reduction")
benchmark_einsum("i,i->", [100000000], [100000000], np.complex128, "Test")
benchmark_einsum("i,i->", [100000000], [100000000], np.float64, "Test")
benchmark_einsum("i,i->", [10], [10], np.float64, "Test")

# Test Correctness
# lhs_vector = np.random.rand(100)
# rhs_vector = np.random.rand(100)
# print(np.einsum("i,i->", lhs_vector, rhs_vector))
# print(einspeed.einsum("i,i->", lhs_vector, rhs_vector))
# print(lhs_vector @ rhs_vector)

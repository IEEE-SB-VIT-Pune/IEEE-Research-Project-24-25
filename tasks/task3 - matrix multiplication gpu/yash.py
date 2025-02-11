import numpy as np
from numba import cuda

@cuda.jit
def multiply_kernel(A, B, C, m, n, p):

    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if row < m and col < p:
        sum = 0
        for k in range(n):
            sum += A[row, k] * B[k, col]
        C[row, col] = sum

def matrix_multiply(A, B):
    m, n = A.shape
    n, p = B.shape

    C = np.zeros((m, p), dtype=np.float32)

    A_gpu = cuda.to_device(A)
    B_gpu = cuda.to_device(B)
    C_gpu = cuda.to_device(C)

    threads_per_block = (16, 16)
    blocks_per_grid_x = (p + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (m + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    multiply_kernel[blocks_per_grid, threads_per_block](A_gpu, B_gpu, C_gpu, m, n, p)

    C = C_gpu.copy_to_host()
    return C

if __name__ == "__main__":

    A = np.random.rand(4, 3).astype(np.float32)
    B = np.random.rand(3, 5).astype(np.float32)

    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)

    C = matrix_multiply(A, B)

    print("\nResultant Matrix C:")
    print(C)


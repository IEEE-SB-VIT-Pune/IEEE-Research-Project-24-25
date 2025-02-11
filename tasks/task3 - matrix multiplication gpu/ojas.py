import cupy as cp

matrix_mul_kernel = cp.RawKernel(r'''
extern "C" __global__
void matmul(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        double value = 0;
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}
''', 'matmul')

N = int(input("Enter the size of the square matrices: "))

print("Enter values for Matrix A (row by row):")
matrix_a_values = list(map(float, input().split()))
if len(matrix_a_values) != N * N:
    raise ValueError(f"Expected {N * N} values for Matrix A, got {len(matrix_a_values)}")
matrix_a = cp.array(matrix_a_values).reshape(N, N)


print("Enter values for Matrix B (row by row):")
matrix_b_values = list(map(float, input().split()))
if len(matrix_b_values) != N * N:
    raise ValueError(f"Expected {N * N} values for Matrix B, got {len(matrix_b_values)}")
matrix_b = cp.array(matrix_b_values).reshape(N, N)


result_gpu = cp.zeros((N, N), dtype=cp.float64)


threads_per_block = (16, 16)
blocks_per_grid = ((N + threads_per_block[0] - 1) // threads_per_block[0],
                   (N + threads_per_block[1] - 1) // threads_per_block[1])

matrix_mul_kernel((blocks_per_grid), (threads_per_block),
                  (matrix_a, matrix_b, result_gpu, N))

result = cp.asnumpy(result_gpu)
print("\nResultant Matrix (A x B):")
for row in result:
    print(' '.join(map(str, row)))


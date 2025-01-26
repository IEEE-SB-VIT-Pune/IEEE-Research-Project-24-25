import numpy as np
from numba import cuda
import random

def create_matrix(row, column):
    return np.random.randint(1, 21, size=(row, column), dtype=np.int32)

def print_matrix(matrix):
    for row in matrix:
        print(' '.join(map(str, row)))

@cuda.jit
def calculate_distances(chart, const_row, distances, rows, cols):
    row_idx = cuda.grid(1)
    if row_idx < rows and row_idx != const_row:
        dist = 0
        for col_idx in range(cols):
            dist += abs(chart[row_idx, col_idx] - chart[const_row, col_idx])
        distances[row_idx] = dist
    elif row_idx == const_row:
        distances[row_idx] = float('inf')  


m = int(input("Enter no. of rows: "))
n = int(input("Enter no. of columns: "))


chart = create_matrix(m, n)
print("Generated Matrix:")
print_matrix(chart)


const_row = int(input("Enter the row index: "))
distances = np.zeros(m, dtype=np.float32)


d_chart = cuda.to_device(chart)
d_distances = cuda.to_device(distances)

threads_per_block = 256
blocks_per_grid = (m + threads_per_block - 1) // threads_per_block

calculate_distances[blocks_per_grid, threads_per_block](d_chart, const_row, d_distances, m, n)

distances = d_distances.copy_to_host()
sorted_indices = np.argsort(distances)[:3]

print("Top three rows having minimum distance:")
for idx in sorted_indices:
    if distances[idx] != float('inf'): 
        print(f"Index: {idx}, Row: {chart[idx].tolist()}, Distance: {distances[idx]}")

            
            
        
        
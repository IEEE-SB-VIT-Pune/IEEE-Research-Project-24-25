import numpy as np
from numba import cuda
from sklearn.neighbors import NearestNeighbors  
import random

def create_matrix(row, column):
    return np.random.randint(1, 21, size=(row, column), dtype=np.int32)


def calculate_neighbors(matrix, n_neighbors):
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
    knn.fit(matrix)
    dist, indices = knn.kneighbors(matrix)
    return indices


@cuda.jit(device=True)
def binary_search(array, value):
    left= 0
    right =  array.shape[0] - 1
    while left <= right:
        mid = left + (right - left) // 2
        if (array[mid]  == value) :
            return True
        elif (array[mid] < value) :
            left = mid + 1
        else :
            right = mid - 1
    return False

@cuda.jit
def mnn_kernel(neighbor_matrix, mnn_scores):
    thread_id = cuda.grid(1)

    if thread_id < neighbor_matrix.shape[0] :
        current_neighbors = neighbor_matrix[thread_id]

        mnn_score = 0

        for neighbor_id in current_neighbors:
            neighbor_and_neighbor = neighbor_matrix[neighbor_id]

            if binary_search(neighbor_and_neighbor, thread_id):
                mnn_score += 1


        mnn_scores[thread_id] = mnn_score


if __name__ == "__main__":
    row = int(input("Enter the number of rows: "))
    column = int(input("Enter the number of columns: "))
    
    main_matrix = create_matrix(row, column)
    print("Main Matrix:")
    print(main_matrix)
    
    neighbor_matrix = calculate_neighbors(main_matrix, 3)
    print("Indices of 3 Nearest Neighbors for Each Row:")
    print(neighbor_matrix)
    
    threads_per_block = 256  
    blocks_per_grid = (neighbor_matrix.shape[0] + threads_per_block - 1) // threads_per_block 

    neighbor_matrix_gpu = cuda.to_device(neighbor_matrix)
    mnn_scores_gpu = cuda.to_device(np.zeros(neighbor_matrix.shape[0], dtype=np.int32))

    mnn_kernel[blocks_per_grid, threads_per_block](neighbor_matrix_gpu, mnn_scores_gpu)

    
    mnn_scores = mnn_scores_gpu.copy_to_host()

    print("\nMNN Scores for all points:")
    for i in range(len(mnn_scores)):
        score = mnn_scores[i]  
        print(f"Point {i+1}: {score}")
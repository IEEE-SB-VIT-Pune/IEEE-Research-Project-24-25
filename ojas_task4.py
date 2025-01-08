import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def calculate_nn_scores(X, k=1, point_index=None):
    knn_model = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
    knn_model.fit(X)
    _, knn_adjacency_matrix = knn_model.kneighbors(X)

    knn_adjacency_matrix = knn_adjacency_matrix[:, 1:]

    nn_scores = np.zeros(X.shape[0], dtype=int)

    for neighbors in knn_adjacency_matrix:
        for neighbor in neighbors:
            nn_scores[neighbor] += 1

    if point_index is not None:
        if 0 <= point_index < len(nn_scores):
            return nn_scores[point_index]
        else:
            raise ValueError("Invalid point index. It should be within the range of the dataset.")

    return nn_scores


X = pd.DataFrame({
    'x1': [1, 2, 3, 4, 5],
    'x2': [5, 4, 3, 2, 1]
})

k = 3

try:
    point_index = int(input(f"Enter the index of the point (0 to {len(X) - 1}): "))
    result = calculate_nn_scores(X, k=k, point_index=point_index)
    print(f"NN Score for index {point_index}: {result}")
except ValueError as e:
    print(e)

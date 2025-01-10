import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def calculate_nn_and_mnn_scores(X, k=1, point_index=None):
    knn_model = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
    knn_model.fit(X)
    _, knn_adjacency_matrix = knn_model.kneighbors(X)

    # Remove self-reference (each point is its own first neighbor)
    knn_adjacency_matrix = knn_adjacency_matrix[:, 1:]

    nn_scores = np.zeros(X.shape[0], dtype=int)
    mnn_scores = np.zeros(X.shape[0], dtype=int)

    # Store nearest neighbors in a dictionary for mutual check
    neighbors_dict = {i: set(knn_adjacency_matrix[i]) for i in range(X.shape[0])}

    # Calculate NN and MNN scores
    for i, neighbors in enumerate(knn_adjacency_matrix):
        for neighbor in neighbors:
            # Increment NN score for each neighbor
            nn_scores[neighbor] += 1

            # Check mutuality for MNN score
            if i in neighbors_dict[neighbor]:  # Mutual nearest neighbor check
                mnn_scores[i] += 1

    if point_index is not None:
        if 0 <= point_index < len(nn_scores):
            return {
                "NN Score": nn_scores[point_index],
                "MNN Score": mnn_scores[point_index]
            }
        else:
            raise ValueError("Invalid point index. It should be within the range of the dataset.")

    return {"NN Scores": nn_scores, "MNN Scores": mnn_scores}


# Example data
X = pd.DataFrame({
    'x1': [1, 2, 3, 4, 5],
    'x2': [5, 4, 3, 2, 1]
})

k = 3

try:
    point_index = int(input(f"Enter the index of the point (0 to {len(X) - 1}): "))
    results = calculate_nn_and_mnn_scores(X, k=k, point_index=point_index)
    print(f"Results for index {point_index}:")
    print(f"NN Score: {results['NN Score']}")
    print(f"MNN Score: {results['MNN Score']}")
except ValueError as e:
    print(e)

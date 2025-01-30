import numpy as np

def sampler(matrix, nn_scores, mnn_scores, mutual_neighbors):
    train_sample = []
    indices = np.where(nn_scores == np.max(nn_scores))[0]
    
    if len(indices) > 1:
        val =np.max([mnn_scores[x] for x in indices])
        train_index=mnn_scores.index(val)
       
    else:
        train_index = indices[0]
        
    train_sample.append(matrix[train_index])
    
    to_remove = set(mutual_neighbors[train_index])
    to_remove.add(train_index)
    print(to_remove)
    
    matrix = [m for i, m in enumerate(matrix) if i not in to_remove]

    return train_sample,matrix

# Sample Inputs
#matrix = [[1,1,1,1], [2,2,2,2], [3,3,3,3], [4,4,4,4]]
#nn_scores = [2, 3, 3, 1]
#mnn_scores = [1, 2, 1, 1]
#mutual_neighbors = [[1], [0,2], [1,3], [2]]

#result,matrix= sampler(matrix, nn_scores, mnn_scores, mutual_neighbors)
#print("Final Sampled Data:", result,"\n matrix: ",matrix)

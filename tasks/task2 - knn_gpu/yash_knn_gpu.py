import pandas as pd
import numpy as np
import cupy as cp
import random

def generate_random_data(rows, columns, num_classes):

    data = []
    for _ in range(rows):
        point = [random.randint(0, 100) for _ in range(columns)]
        point.append(random.randint(1, num_classes))
        data.append(point)

    columns_names = [f"feature_{i+1}" for i in range(columns)] + ["Class"]
    df = pd.DataFrame(data, columns=columns_names)
    df.to_csv("random_data.csv", index=False)
    print(f"Random data saved to random_data.csv")

def manhattan_distance(features, test_point):

    test_point = test_point.reshape(1, -1)

    differences = cp.abs(features - test_point)

    return cp.sum(differences, axis=1)

def knn():

    rows = int(input("Enter the number of rows for the dataset: "))
    columns = int(input("Enter the number of features (columns) for the dataset: "))
    num_classes = int(input("Enter the number of classes: "))

    generate_random_data(rows, columns, num_classes)
    df = pd.read_csv("random_data.csv")
    print("\nGenerated Dataset:")
    print(df.head())

    print("\nEnter values for the test point:")
    test_point = np.array([float(input(f"Enter value for feature_{i+1}: ")) 
                          for i in range(columns)])

    features = cp.array(df.iloc[:, :-1].values.astype(np.float32))
    test_point_gpu = cp.array(test_point.astype(np.float32))

    distances = manhattan_distance(features, test_point_gpu)
    
    distances_cpu = cp.asnumpy(distances)

    df['Distance'] = distances_cpu
    df_sorted = df.sort_values(by='Distance').reset_index(drop=True)

    k = int(input("Enter the value of k: "))
    knn = df_sorted.head(k)
    print(f"\nK Nearest Neighbors\n{knn}")

    predicted_class = knn['Class'].mode()[0]
    print(f"\nPredicted class for {test_point} is: {predicted_class}")

if __name__ == "__main__":
    knn()
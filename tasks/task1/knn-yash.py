import pandas as pd
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


def manhattan_distance(point1, point2):

    return sum(abs(a - b) for a, b in zip(point1, point2))


def knn():

    rows = int(input("Enter the number of rows for the dataset: "))
    columns = int(input("Enter the number of features (columns) for the dataset: "))
    num_classes = int(input("Enter the number of classes: "))

    generate_random_data(rows, columns, num_classes)

    df = pd.read_csv("random_data.csv")
    print("\nGenerated Dataset:")
    print(df.head())

    print("\nEnter values for the test point:")
    test_point = [int(input(f"Enter value for feature_{i+1}: ")) for i in range(columns)]

    df['Distance'] = df.iloc[:, :-1].apply(lambda row: manhattan_distance(test_point, row), axis=1)

    df_sorted = df.sort_values(by='Distance').reset_index(drop=True)

    k = int(input("Enter the value of k: "))
    knn = df_sorted.head(k)

    print(f"\nK Nearest Neighbors\n{knn}")

    predicted_class = knn['Class'].mode()[0]
    print(f"\nPredicted class{test_point} is: {predicted_class}")


if __name__ == "__main__":
    knn()
